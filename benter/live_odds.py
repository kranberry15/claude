"""Poll live HKJC Win / Place / Quinella tote odds and keep the betting slip current.

    python -m benter.live_odds --venue ST                      # log odds only
    python -m benter.live_odds --venue HV --history runs.csv   # + live slip
    python -m benter.live_odds --venue ST --once               # one snapshot, then exit

Source: the GraphQL gateway behind bet.hkjc.com (https://info.cld.hkjc.com/graphql/base/).
The gateway only accepts the exact query strings the bet.hkjc.com app sends
(anything else gets WHITELIST_ERROR), so ODDS_QUERY / CARD_QUERY below are
copied byte for byte.  Don't reformat them.  If HKJC changes its app, capture
the new strings from the browser's network tab (Developer tools -> Network ->
filter "graphql" on bet.hkjc.com) and paste them in.

Polling pattern (one request returns Win, Place and Quinella for one race):
  * the next race to run is polled every --interval seconds (default 10);
  * later races are polled every --slow-interval seconds (default 60);
  * races whose pools have closed are dropped; the loop ends when all are closed.
So a 10-race card costs about one request every 7-8 seconds, not ten per 10 s.

Outputs, under --out-dir (default data/live/):
  <date>_<venue>_card.csv       runners from the card query (once at start)
  <date>_<venue>_snapshots.csv  every odds refresh, append-only, one row per
                                selection, de-duplicated on the pool's lastUpdateTime
  <date>_<venue>_latest.csv     current odds in weekend_slip's --odds format:
                                race_id, bet_type, selection, odds
race_id is "<YYYY-MM-DD>-<venue>-R<nn>" (e.g. 2026-10-14-HV-R03).  Use the same ids
for the card rows in your --history file.

This reads public odds only and never places a bet.  HKJC's website terms
restrict automated access; check them, keep the rate low, and use it for
personal research.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

ENDPOINT = "https://info.cld.hkjc.com/graphql/base/"
HKT = timezone(timedelta(hours=8))
POOL_TO_BET = {"WIN": "WIN", "PLA": "PLACE", "QIN": "QUINELLA"}
MIN_INTERVAL = 5.0  # seconds; refuse anything faster

HEADERS = {
    "accept": "*/*",
    "content-type": "application/json",
    "origin": "https://bet.hkjc.com",
    "referer": "https://bet.hkjc.com/",
    "user-agent": "benter-odds-poller/1.0 (personal research; read-only)",
}

# Exact whitelisted operations used by bet.hkjc.com.  Byte for byte - do not reformat.
CARD_QUERY = "query racing($date: String, $venueCode: String) {\n  raceMeetings(date: $date, venueCode: $venueCode) {\n    status\n    currentNumberOfRace\n    totalInvestment\n    races {\n      no\n      status\n      runners {\n        id\n        no\n        standbyNo\n        status\n        name_ch\n        name_en\n        horse {\n          id\n          code\n        }\n        color\n        barrierDrawNumber\n        handicapWeight\n        currentWeight\n        currentRating\n        internationalRating\n        gearInfo\n        racingColorFileName\n        allowance\n        trainerPreference\n        last6run\n        saddleClothNo\n        trumpCard\n        priority\n        finalPosition\n        deadHeat\n        winOdds\n        jockey {\n          code\n          name_en\n          name_ch\n        }\n        trainer {\n          code\n          name_en\n          name_ch\n        }\n      }\n    }\n  }\n}"  # noqa: E501
ODDS_QUERY = "query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {\n  raceMeetings(date: $date, venueCode: $venueCode) {\n    pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {\n      id\n      status\n      sellStatus\n      oddsType\n      lastUpdateTime\n      guarantee\n      minTicketCost\n      name_en\n      name_ch\n      leg {\n        number\n        races\n      }\n      cWinSelections {\n        composite\n        name_ch\n        name_en\n        starters\n      }\n      oddsNodes {\n        combString\n        oddsValue\n        hotFavourite\n        oddsDropValue\n        bankerOdds {\n          combString\n          oddsValue\n        }\n      }\n    }\n  }\n}"  # noqa: E501


class GatewayError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Transport + parsing
# ---------------------------------------------------------------------------


def post(variables: dict, query: str, timeout: float = 15.0, user_agent: str | None = None,
         operation: str = "racing") -> list[dict]:
    """POST one whitelisted operation; return the raceMeetings list."""
    body = json.dumps({"operationName": operation, "variables": variables, "query": query}).encode()
    headers = dict(HEADERS, **({"user-agent": user_agent} if user_agent else {}))
    req = urllib.request.Request(ENDPOINT, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    if raw[:2] == b"\x1f\x8b":  # the gateway gzips replies even when not asked to
        raw = gzip.decompress(raw)
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("errors"):
        raise GatewayError(str(payload["errors"])[:500])
    return (payload.get("data") or {}).get("raceMeetings") or []


def race_id(day: str, venue: str, race_no: int) -> str:
    return f"{day}-{venue}-R{race_no:02d}"


def selection(comb: str) -> str | None:
    """'07' -> '7';  '03,11' / '03-11' / '11+03' -> '3-11'."""
    nums = sorted(int(x) for x in re.findall(r"\d+", str(comb)))
    return "-".join(map(str, nums)) if nums else None


def parse_odds_value(v) -> float | None:
    """Decimal odds, or None for SCR (scratched), '---' (no bets yet) etc."""
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return x if x >= 1.0 else None


@dataclass
class Pool:
    race_no: int
    odds_type: str
    status: str | None
    sell_status: str | None
    last_update: str | None
    odds: dict  # selection -> decimal odds (scratched / unpriced omitted)

    @property
    def selling(self) -> bool:
        return (self.sell_status or self.status) == "START_SELL"


def parse_pools(meetings: list[dict], race_no: int) -> list[Pool]:
    out = []
    for m in meetings[:1]:
        for p in m.get("pmPools") or []:
            if p.get("oddsType") not in POOL_TO_BET:
                continue
            odds = {}
            for n in p.get("oddsNodes") or []:
                sel, o = selection(n.get("combString")), parse_odds_value(n.get("oddsValue"))
                if sel and o:
                    odds[sel] = o
            out.append(Pool(race_no, p["oddsType"], p.get("status"), p.get("sellStatus"), p.get("lastUpdateTime"), odds))
    return out


def fetch_race(day: str, venue: str, race_no: int, **kw) -> list[Pool]:
    meetings = post({"date": day, "venueCode": venue, "raceNo": race_no, "oddsTypes": ["WIN", "PLA", "QIN"]},
                    ODDS_QUERY, **kw)
    return parse_pools(meetings, race_no)


def fetch_card(day: str, venue: str, **kw) -> list[dict]:
    """Runner rows for the meeting.  The gateway falls back to whatever meeting
    is live when (date, venue) has none, so callers should sanity-check."""
    meetings = post({"date": day, "venueCode": venue}, CARD_QUERY, **kw)
    rows = []
    for m in meetings[:1]:
        for r in m.get("races") or []:
            for h in r.get("runners") or []:
                if h.get("standbyNo"):  # standby horses aren't runners unless promoted
                    continue
                rows.append(dict(
                    race_id=race_id(day, venue, int(r["no"])), race_no=int(r["no"]), race_status=r.get("status"),
                    horse_no=h.get("no"), horse_code=(h.get("horse") or {}).get("code"), horse_name=h.get("name_en"),
                    status=h.get("status"), draw=h.get("barrierDrawNumber"), weight_carried=h.get("handicapWeight"),
                    allowance=h.get("allowance"), body_weight=h.get("currentWeight"), rating=h.get("currentRating"),
                    jockey_id=(h.get("jockey") or {}).get("code"), trainer_id=(h.get("trainer") or {}).get("code"),
                    last6run=h.get("last6run"), win_odds=parse_odds_value(h.get("winOdds")),
                ))
    return rows


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

SNAP_COLS = ["polled_at", "race_id", "bet_type", "selection", "odds", "sell_status", "last_update"]


class OddsBook:
    """Latest odds per (race, pool) plus an append-only snapshot log."""

    def __init__(self, day: str, venue: str, out_dir: Path):
        self.day, self.venue = day, venue
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{day}_{venue}"
        self.snap_path = out_dir / f"{stem}_snapshots.csv"
        self.latest_path = out_dir / f"{stem}_latest.csv"
        self.card_path = out_dir / f"{stem}_card.csv"
        self.pools: dict[tuple[int, str], Pool] = {}

    def update(self, pools: list[Pool], polled_at: str) -> bool:
        """Store pools whose lastUpdateTime moved; True if anything changed."""
        new_rows = []
        for p in pools:
            key = (p.race_no, p.odds_type)
            old = self.pools.get(key)
            if old and old.last_update == p.last_update and old.sell_status == p.sell_status:
                continue
            self.pools[key] = p
            rid, bet = race_id(self.day, self.venue, p.race_no), POOL_TO_BET[p.odds_type]
            new_rows += [(polled_at, rid, bet, s, o, p.sell_status, p.last_update) for s, o in p.odds.items()]
        if new_rows:
            fresh = not self.snap_path.exists()
            with self.snap_path.open("a", newline="") as f:
                w = csv.writer(f)
                if fresh:
                    w.writerow(SNAP_COLS)
                w.writerows(new_rows)
            self.write_latest()
        return bool(new_rows)

    def latest_rows(self):
        for (rn, ot), p in sorted(self.pools.items()):
            for s, o in p.odds.items():
                yield race_id(self.day, self.venue, rn), POOL_TO_BET[ot], s, o

    def write_latest(self):
        tmp = self.latest_path.with_suffix(".tmp")
        with tmp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["race_id", "bet_type", "selection", "odds"])
            w.writerows(self.latest_rows())
        os.replace(tmp, self.latest_path)  # atomic: readers never see a half-written file

    def write_card(self, rows: list[dict]):
        if not rows:
            return
        with self.card_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


# ---------------------------------------------------------------------------
# Polling loop
# ---------------------------------------------------------------------------


class Poller:
    def __init__(self, day: str, venue: str, races: list[int], book: OddsBook, interval: float = 10.0,
                 slow_interval: float = 60.0, fetch=fetch_race, clock=time.monotonic, sleep=time.sleep,
                 on_change=None, log=print, user_agent: str | None = None):
        if interval < MIN_INTERVAL:
            raise ValueError(f"interval must be at least {MIN_INTERVAL:g}s")
        self.day, self.venue, self.book = day, venue, book
        self.interval, self.slow = interval, max(slow_interval, interval)
        self.fetch, self.clock, self.sleep = fetch, clock, sleep
        self.on_change, self.log, self.ua = on_change, log, user_agent
        self.due = {r: 0.0 for r in races}          # race -> next poll time
        self.seen_selling: set[int] = set()
        self.closed: set[int] = set()
        self.backoff = 0.0

    def next_race(self) -> int | None:
        live = sorted(r for r in self.due if r not in self.closed)
        return live[0] if live else None

    def step(self) -> bool:
        """Poll every race that is due.  Returns False when the meeting is over."""
        now, nxt = self.clock(), self.next_race()
        if nxt is None:
            return False
        changed = False
        for r in sorted(self.due):
            if r in self.closed or self.due[r] > now:
                continue
            try:
                pools = self.fetch(self.day, self.venue, r, user_agent=self.ua)
                self.backoff = 0.0
            except (urllib.error.URLError, TimeoutError, GatewayError, ValueError) as e:
                # back off on errors (10 s, 20 s, 40 s ... capped at 5 min) instead of hammering
                self.backoff = min(max(self.backoff * 2, self.interval), 300.0)
                self.log(f"[{_hkt()}] R{r}: {type(e).__name__}: {str(e)[:200]} - retry in {self.backoff:.0f}s")
                self.due[r] = now + self.backoff
                break
            win = next((p for p in pools if p.odds_type == "WIN"), None)
            if win and win.selling:
                self.seen_selling.add(r)
            elif win and (r in self.seen_selling or any(x > r for x in self.seen_selling)):
                # stopped selling after we saw it open, or a later race is open: it has gone
                self.closed.add(r)
                self.log(f"[{_hkt()}] R{r}: pools closed ({win.sell_status})")
                nxt = self.next_race()
                if nxt is not None:
                    self.due[nxt] = min(self.due[nxt], now)  # promote the new next race to fast polling now
            elif not pools:
                self.log(f"[{_hkt()}] R{r}: no pools returned")
            changed |= self.book.update(pools, _hkt())
            if r not in self.closed:
                self.due[r] = now + (self.interval if r == nxt else self.slow)
        if changed and self.on_change:
            self.on_change(self.book)
        return self.next_race() is not None

    def run(self, max_steps: int | None = None):
        steps = 0
        while self.step():
            steps += 1
            if max_steps is not None and steps >= max_steps:
                return
            wake = min(t for r, t in self.due.items() if r not in self.closed)
            self.sleep(max(0.5, wake - self.clock()))
        self.log(f"[{_hkt()}] all pools closed - done")


def _hkt() -> str:
    return datetime.now(HKT).strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Live slip
# ---------------------------------------------------------------------------


def make_slip_callback(history_path: str, cfg_kwargs: dict):
    """Fit the model once on the history; on every odds change, re-score the
    card at the new odds and print the slip if it changed."""
    import pandas as pd

    from benter import weekend_slip as ws

    cfg = ws.Config(**cfg_kwargs)
    model, card = ws.prepare(ws.load_history(history_path), cfg)
    last = {"key": None}

    def on_change(book: OddsBook):
        odds = pd.DataFrame(list(book.latest_rows()), columns=["race_id", "bet_type", "selection", "odds"])
        open_races = {race_id(book.day, book.venue, rn) for (rn, ot), p in book.pools.items()
                      if ot == "WIN" and p.selling}
        c = card[card["race_id"].isin(open_races)]
        if c.empty:
            return
        _, slip = ws.slip_for_card(model, c, odds, cfg)
        key = slip[["race_id", "bet_type", "selection", "stake_hkd"]].to_csv(index=False)
        if key != last["key"]:
            last["key"] = key
            print(f"\n[{_hkt()}] odds moved - slip for races still selling:")
            ws.print_slip(slip, cfg)

    return on_change


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--venue", required=True, choices=["ST", "HV"], help="ST = Sha Tin, HV = Happy Valley")
    ap.add_argument("--date", default=datetime.now(HKT).strftime("%Y-%m-%d"), help="meeting date (HKT), default today")
    ap.add_argument("--races", help="comma-separated race numbers (default: whole card)")
    ap.add_argument("--interval", type=float, default=10.0, help="seconds between polls of the next race")
    ap.add_argument("--slow-interval", type=float, default=60.0, help="seconds between polls of later races")
    ap.add_argument("--out-dir", default="data/live")
    ap.add_argument("--once", action="store_true", help="take one snapshot of every race and exit")
    ap.add_argument("--user-agent", help="override the User-Agent header")
    ap.add_argument("--history", help="runs CSV for weekend_slip: print a live slip as odds move")
    ap.add_argument("--bankroll", type=float, default=200.0)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--min-edge", type=float, default=0.05)
    a = ap.parse_args(argv)

    book = OddsBook(a.date, a.venue, Path(a.out_dir))
    if a.races:
        races = [int(x) for x in a.races.split(",")]
    else:
        try:
            card = fetch_card(a.date, a.venue, user_agent=a.user_agent)
        except (urllib.error.URLError, GatewayError) as e:
            sys.exit(f"could not fetch the card: {e}")
        if not card:
            sys.exit(f"no meeting served for {a.date} {a.venue}")
        book.write_card(card)
        races = sorted({r["race_no"] for r in card})
        print(f"card: {len(races)} races, {len(card)} runners -> {book.card_path}")

    on_change = None
    if a.history:
        on_change = make_slip_callback(a.history, dict(total_bankroll=a.bankroll, kelly_fraction=a.kelly,
                                                       min_edge=a.min_edge))
    poller = Poller(a.date, a.venue, races, book, a.interval, a.slow_interval, on_change=on_change,
                    user_agent=a.user_agent)
    print(f"polling {a.date} {a.venue} races {races}: next race every {a.interval:g}s, others every "
          f"{poller.slow:g}s.  Ctrl-C to stop.  Latest odds -> {book.latest_path}")
    try:
        if a.once:
            poller.slow = poller.interval
            poller.step()
        else:
            poller.run()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
