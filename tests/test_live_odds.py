"""Offline checks for benter.live_odds (no network).   python -m tests.test_live_odds

The gateway payload below is synthetic, shaped like a captured bet.hkjc.com
pmPools response (oddsValue strings, "SCR" for scratched runners).
"""
import csv
import tempfile
from pathlib import Path

from benter import live_odds as lo


def payload(race_no, sell="START_SELL", ts="2026-10-14T19:00:00.000+08:00", win1="3.4"):
    nodes = lambda xs: [{"combString": c, "oddsValue": v, "hotFavourite": False, "oddsDropValue": 0, "bankerOdds": []}
                        for c, v in xs]
    pool = lambda t, xs: {"id": f"x{t}{race_no}", "status": sell, "sellStatus": sell, "oddsType": t,
                          "lastUpdateTime": ts, "oddsNodes": nodes(xs)}
    return [{"pmPools": [
        pool("PLA", [("01", "1.6"), ("02", "2.2"), ("03", "SCR"), ("04", "3.1")]),   # PLA before WIN on purpose
        pool("WIN", [("01", win1), ("02", "5.0"), ("03", "SCR"), ("04", "9.9")]),
        pool("QIN", [("01,02", "8.5"), ("04,01", "20"), ("02-04", "---")]),
    ]}]


def test_parse():
    pools = {p.odds_type: p for p in lo.parse_pools(payload(3), 3)}
    assert pools["WIN"].odds == {"1": 3.4, "2": 5.0, "4": 9.9}          # SCR dropped
    assert pools["QIN"].odds == {"1-2": 8.5, "1-4": 20.0}                # sorted pair, '---' dropped
    assert pools["WIN"].selling
    assert lo.selection("11+03") == "3-11" and lo.selection("07") == "7"


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def test_poller_schedule_dedupe_and_close():
    clock, calls, state = FakeClock(), [], {"closed": set(), "ts": {}}

    def fetch(day, venue, race_no, user_agent=None):
        calls.append((clock.t, race_no))
        sell = "STOP_SELL" if race_no in state["closed"] else "START_SELL"
        return lo.parse_pools(payload(race_no, sell, state["ts"].get(race_no, "t0")), race_no)

    with tempfile.TemporaryDirectory() as d:
        book = lo.OddsBook("2026-10-14", "HV", Path(d))
        changes = []
        p = lo.Poller("2026-10-14", "HV", [1, 2, 3], book, interval=10, slow_interval=60, fetch=fetch,
                      clock=clock, sleep=clock.sleep, on_change=lambda b: changes.append(clock.t), log=lambda *_: None)
        p.step()                                   # t=0: all three due
        assert [r for _, r in calls] == [1, 2, 3] and len(changes) == 1
        clock.t = 10; p.step()                     # only the next race (R1) is due at 10s
        assert [r for t, r in calls if t == 10] == [1]
        assert len(changes) == 1                   # same lastUpdateTime -> nothing new stored
        state["ts"][1] = "t1"; clock.t = 20; p.step()
        assert len(changes) == 2                   # odds moved
        state["closed"].add(1); clock.t = 30; p.step()
        assert 1 in p.closed and p.next_race() == 2
        clock.t = 40; p.step()
        assert [r for t, r in calls if t == 40] == [2]   # R2 now polled fast
        rows = list(csv.DictReader(book.latest_path.open()))
        assert {r["bet_type"] for r in rows} == {"WIN", "PLACE", "QUINELLA"}
        assert {r["race_id"] for r in rows} == {"2026-10-14-HV-R01", "2026-10-14-HV-R02", "2026-10-14-HV-R03"}
        snaps = list(csv.DictReader(book.snap_path.open()))
        assert len(snaps) > len(rows)
        state["closed"].update({2, 3})
        p.run(max_steps=20)
        assert p.next_race() is None


def test_race_already_closed_at_start():
    clock = FakeClock()

    def fetch(day, venue, race_no, user_agent=None):
        return lo.parse_pools(payload(race_no, "START_SELL" if race_no > 1 else "STOP_SELL"), race_no)

    with tempfile.TemporaryDirectory() as d:
        p = lo.Poller("2026-10-14", "ST", [1, 2], lo.OddsBook("2026-10-14", "ST", Path(d)), fetch=fetch,
                      clock=clock, sleep=clock.sleep, log=lambda *_: None)
        p.step(); clock.t = 60; p.step()
        assert 1 in p.closed and p.next_race() == 2


def test_errors_back_off():
    clock, n = FakeClock(), {"calls": 0}

    def fetch(*a, **k):
        n["calls"] += 1
        raise lo.GatewayError("WHITELIST_ERROR")

    with tempfile.TemporaryDirectory() as d:
        p = lo.Poller("2026-10-14", "ST", [1], lo.OddsBook("2026-10-14", "ST", Path(d)), fetch=fetch,
                      clock=clock, sleep=clock.sleep, log=lambda *_: None)
        for _ in range(6):
            p.step()
            clock.t = p.due[1]
        assert p.backoff == 300.0 or p.backoff >= 160
        assert n["calls"] == 6


def test_interval_floor():
    try:
        lo.Poller("d", "ST", [1], None, interval=1)
    except ValueError:
        return
    raise AssertionError("1s interval should be refused")


def test_live_odds_feed_the_slip():
    import numpy as np
    from benter import weekend_slip as ws
    cfg = ws.Config(); cfg.gamma_second = cfg.gamma_third = 1.0
    h = ws.make_mock_history(n_meetings=80, seed=4)
    model, card = ws.prepare(h, cfg, verbose=False)
    rid = sorted(card["race_id"].unique())[0]
    race = card[card["race_id"] == rid]
    import pandas as pd
    live = pd.DataFrame([(rid, "WIN", str(n), o * 1.1) for n, o in zip(race["horse_no"], race["win_odds"])][1:],
                        columns=["race_id", "bet_type", "selection", "odds"])   # first runner scratched
    c2, slip = ws.slip_for_card(model, card, live, cfg)
    r2 = c2[c2["race_id"] == rid]
    assert len(r2) == len(race) - 1
    assert np.isclose(r2["model_p"].sum(), 1.0)
    assert slip["stake_hkd"].sum() <= cfg.total_bankroll


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
