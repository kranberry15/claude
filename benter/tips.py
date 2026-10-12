"""One command: today's betting tips for the next HKJC meeting.

    python -m benter.tips             # tips at the current odds, once
    python -m benter.tips --live      # keep watching the odds (every 10 s) and reprint when tips change

It refreshes past results, fetches the next meeting's card and the live Win /
Place / Quinella odds, fits the Benter model and prints a plain betting slip.
Tips are also saved to data/live/tips_<date>_<venue>.txt.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from benter import hkjc_data as hd
from benter import live_odds as lo
from benter import track
from benter import weekend_slip as ws

OUT = hd.ROOT / "data" / "live"


def odds_table(day: str, venue: str, races: list[int], delay: float = 1.0) -> pd.DataFrame:
    rows = []
    for k in races:
        for p in lo.fetch_race(day, venue, k):
            if p.selling:
                rows += [(lo.race_id(day, venue, k), lo.POOL_TO_BET[p.odds_type], s, o) for s, o in p.odds.items()]
        time.sleep(delay)
    return pd.DataFrame(rows, columns=["race_id", "bet_type", "selection", "odds"])


def describe(slip: pd.DataFrame, card: pd.DataFrame, meeting: dict, cfg: ws.Config, model: ws.FittedModel,
             n_open: int) -> str:
    day = datetime.strptime(meeting["date"], "%Y-%m-%d")
    venue = {"ST": "SHA TIN", "HV": "HAPPY VALLEY"}[meeting["venueCode"]]
    post = {lo.race_id(meeting["date"], meeting["venueCode"], int(r["no"])): (r.get("postTime") or "")[11:16]
            for r in meeting.get("races") or []}
    m = model.metrics
    trusted = m["r2_combined"] > m["r2_public"]
    out = [f"{venue} - {day:%a %d %b %Y} - tips at {datetime.now(lo.HKT):%H:%M} HKT",
           f"Bankroll HKD {cfg.total_bankroll:.0f}, races with odds open: {n_open}", ""]
    if not trusted:
        out += ["!! PAPER ONLY: on past races the model was NOT better than the tote odds",
                f"   (score {m['r2_combined']:.4f} vs odds {m['r2_public']:.4f}). Do not bet real money on these.", ""]
    if slip.empty:
        out += ["NO BETS - nothing is good value at the current odds. Keep your money."]
    else:
        for _, r in slip.iterrows():
            rn = int(r.race_id.split("-R")[-1])
            what = {"WIN": "WIN", "PLACE": "PLACE", "QUINELLA": "QUINELLA"}[r.bet_type]
            sel = f"No. {r.selection}" if r.bet_type != "QUINELLA" else f"Nos. {r.selection.replace('-', ' + ')}"
            out.append(f"Race {rn} ({post.get(r.race_id, '')})  {what:8s} {sel:12s} {r['name'][:34]:34s}  bet HKD {r.stake_hkd:.0f}")
            out.append(f"      model chance {r.model_p:.0%}, odds {r.odds:.1f} -> value {r.edge:+.0%}")
        out += ["", f"TOTAL: HKD {slip.stake_hkd.sum():.0f}"]
    out += ["", "Odds keep moving until each race starts. Check again ~5 minutes before the race;",
            "if a tip disappears, skip it."]
    return "\n".join(out)


def replay(history: pd.DataFrame, n_meetings: int, cfg: ws.Config):
    """Paper-test on the last n past meetings: for each, fit on earlier races
    only, bet at the final tote win odds (Win bets only - the results pages'
    Place / Quinella dividends aren't stored) and settle against the result."""
    days = sorted(history["date"].unique())[-n_meetings:]
    staked = returned = 0.0
    lines = []
    for d in days:
        past, meet = history[history["date"] < d], history[history["date"] == d]
        card = meet.assign(finish_position=float("nan"), finish_time=float("nan"))
        model, card_feat = ws.prepare(pd.concat([past, card], ignore_index=True), cfg, verbose=False)
        _, slip = ws.slip_for_card(model, card_feat, None, cfg)
        slip = slip[slip["bet_type"] == "WIN"]
        won = meet.set_index(["race_id", meet["horse_no"].astype(str)])["finish_position"]
        day_s = day_r = 0.0
        for _, r in slip.iterrows():
            ret = r.stake_hkd * r.odds if won.get((r.race_id, r.selection)) == 1 else 0.0
            day_s, day_r = day_s + r.stake_hkd, day_r + ret
            lines.append(f"  {r.race_id}  WIN No. {r.selection:3s} odds {r.odds:5.1f}  HKD {r.stake_hkd:.0f} -> {ret:.0f}")
        staked, returned = staked + day_s, returned + day_r
        m = model.metrics
        print(f"{pd.Timestamp(d):%Y-%m-%d}: {len(slip)} bets, staked {day_s:.0f}, returned {day_r:.0f}   "
              f"(model {m['r2_combined']:.3f} vs odds {m['r2_public']:.3f})")
    print("\n".join(lines))
    print(f"\nTOTAL over {len(days)} meetings: staked HKD {staked:.0f}, returned HKD {returned:.0f}, "
          f"profit HKD {returned - staked:+.0f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true", help="keep polling odds and reprint tips when they change")
    ap.add_argument("--bankroll", type=float, default=200.0)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--min-edge", type=float, default=0.05)
    ap.add_argument("--skip-update", action="store_true", help="don't download new results first")
    ap.add_argument("--details", action="store_true", help="print the model's statistics")
    ap.add_argument("--replay", type=int, metavar="N", help="paper-test the last N past meetings instead")
    a = ap.parse_args(argv)
    cfg = ws.Config(total_bankroll=a.bankroll, kelly_fraction=a.kelly, min_edge=a.min_edge)
    if a.replay:
        return replay(hd.load_results(), a.replay, cfg)

    print("1/4 updating past results ...")
    if not a.skip_update:
        hd.update_results(log=lambda *_: None)
    history = hd.load_results()
    print(f"    {history.race_id.nunique()} past races ({history.date.min():%b %Y} - {history.date.max():%b %Y})")

    print("2/4 fetching the next meeting ...")
    meeting = hd.next_meeting()
    if not meeting:
        sys.exit("HKJC isn't showing an upcoming meeting right now. Try again later.")
    day, venue = meeting["date"], meeting["venueCode"]
    card = hd.card_frame(meeting)
    races = sorted(int(r) for r in card["race_no"].unique())
    print(f"    {venue} {day}: {len(races)} races, {len(card)} runners")

    print("3/4 fetching live odds ...")
    odds = odds_table(day, venue, races)
    open_ids = set(odds.loc[odds["bet_type"] == "WIN", "race_id"])
    if not open_ids:
        sys.exit(f"Betting for {venue} {day} hasn't opened yet (no odds). "
                 "Odds usually open the day before; try again then.")
    card = card[card["race_id"].isin(open_ids)]
    win = odds[odds["bet_type"] == "WIN"].set_index(["race_id", "selection"])["odds"]
    card = card.assign(win_odds=[win.get((r, str(n))) for r, n in zip(card["race_id"], card["horse_no"])])
    card = card[card["win_odds"].notna()]

    print("4/4 fitting the model and sizing bets ...")
    model, card_feat = ws.prepare(pd.concat([history, card], ignore_index=True), cfg, verbose=a.details)
    _, slip = ws.slip_for_card(model, card_feat, odds, cfg)
    text = describe(slip, card_feat, meeting, cfg, model, len(open_ids))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"tips_{day}_{venue}.txt"
    path.write_text(text + "\n")
    trusted = model.metrics["r2_combined"] > model.metrics["r2_public"]
    now = datetime.now(lo.HKT).isoformat(timespec="seconds")
    saved = track.save_tips(slip, day, venue, open_ids, trusted, now)
    print("\n" + text)
    print(f"\n(tips saved for the paper record: {saved.relative_to(hd.ROOT)})")

    if a.live:
        book = lo.OddsBook(day, venue, OUT)
        last = {"text": slip.to_csv(index=False)}

        def on_change(b: lo.OddsBook):
            live = pd.DataFrame(list(b.latest_rows()), columns=["race_id", "bet_type", "selection", "odds"])
            selling = {lo.race_id(day, venue, rn) for (rn, ot), p in b.pools.items() if ot == "WIN" and p.selling}
            c = card_feat[card_feat["race_id"].isin(selling)]
            if c.empty:
                return
            _, s = ws.slip_for_card(model, c, live, cfg)
            if s.to_csv(index=False) != last["text"]:
                last["text"] = s.to_csv(index=False)
                t = describe(s, c, meeting, cfg, model, len(selling))
                path.write_text(t + "\n")
                track.save_tips(s, day, venue, selling, trusted, datetime.now(lo.HKT).isoformat(timespec="seconds"))
                print("\n" + "-" * 70 + "\nTIPS CHANGED\n" + t)

        print(f"\nwatching odds every 10 s until the last race closes (Ctrl-C to stop) ...")
        try:
            lo.Poller(day, venue, races, book, on_change=on_change, log=lambda *_: None).run()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
