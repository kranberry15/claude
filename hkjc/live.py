"""Live recommender: polls HKJC odds and prints the best WIN / PLACE /
QUINELLA PLACE bets for the current meeting.

    python -m hkjc.results --start 2016-09-01          # once: build history
    python -m benter.run --data data/hkjc --out data/hkjc/model.json
    python -m hkjc.live                                 # race day, refreshes every 60 s
    python -m hkjc.live --once --race 5 --bankroll 5000

For each open race it:
  1. reads the race card (draw, weight, rating, jockey, trainer) and the
     WIN, PLA and QPL odds from HKJC's GraphQL feed;
  2. builds each runner's features from the scraped history plus today's card;
  3. gets win chances from the market-anchored model, then place and
     quinella-place chances from Benter's discounted Harville formula;
  4. lists every bet whose edge (chance x odds - 1) clears the rule fitted on
     history, with a quarter-Kelly stake capped at 10% of bankroll per race.
Every odds snapshot is appended to data/hkjc/odds/<date>_<venue>.jsonl so
place and QPL rules can later be backtested on real pre-race prices.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from benter.exotics import place_and_qpl
from benter.features import build_features, load_raw
from .graphql import HKJCGraphQL, parse_pools

ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "data" / "hkjc"


def load_model(path: Path | None):
    for p in [path] if path else [HIST / "model.json", ROOT / "tracker" / "model.json"]:
        if p and p.exists():
            return json.loads(p.read_text()), p
    raise SystemExit("no model.json found; run python -m benter.run --data data/hkjc --out data/hkjc/model.json")


def _status_ok(s) -> bool:
    s = str(s or "").lower()
    return not any(k in s for k in ("scr", "standby", "withdr"))


def brand_code(s) -> str:
    """HKJC brand number (e.g. H123) out of whatever id form the feed uses."""
    m = re.search(r"[A-Z]\d{3}$", str(s or "").strip())
    return m[0] if m else str(s)


def card_rows(meeting: dict, race: dict, win_odds: dict) -> pd.DataFrame:
    """Today's runners in the same raw schema as the scraped history."""
    date = pd.Timestamp(meeting["date"][:10])
    venue = meeting["venueCode"]
    track = ((race.get("raceTrack") or {}).get("description_en") or "").upper()
    rows = []
    for r in race.get("runners") or []:
        try:
            no = int(r.get("no"))
        except (TypeError, ValueError):
            continue
        if not _status_ok(r.get("status")) or no not in win_odds:
            continue
        rows.append(dict(
            race_id=f"{meeting['date'][:10]}-{venue}-{race['no']}", date=date, venue=venue, race_no=int(race["no"]),
            distance=float(race.get("distance") or 0), race_class=0, surface=1 if "ALL WEATHER" in track else 0,
            going=race.get("go_en"), horse_no=no, horse_id=brand_code((r.get("horse") or {}).get("code") or r.get("name_en")),
            horse_name=r.get("name_en"), jockey_id=(r.get("jockey") or {}).get("name_en"), trainer_id=(r.get("trainer") or {}).get("name_en"),
            actual_weight=float(r.get("handicapWeight") or 0) or np.nan, declared_weight=float(r.get("currentWeight") or 0) or np.nan,
            draw=float(r.get("barrierDrawNumber") or 0) or np.nan, horse_rating=float(r.get("currentRating") or 60),
            horse_age=4.0, win_odds=win_odds[no], result=np.nan, won=0.0, lengths_behind=np.nan, finish_time=np.nan,
            place_odds=np.nan,
        ))
    df = pd.DataFrame(rows)
    if len(df):
        df["field"] = len(df)
    return df


def win_probs(model: dict, feats: pd.DataFrame) -> np.ndarray:
    m = model["market"]
    X = feats[m["columns"]].astype(float).values
    u = ((X - np.array(m["mean"])) / np.array(m["std"])) @ np.array(m["coef"])
    e = np.exp(u - u.max())
    return e / e.sum()


def score_race(history: pd.DataFrame, model: dict, meeting: dict, race: dict, book: dict, bankroll: float):
    win = book.get("WIN") or {}
    live = card_rows(meeting, race, win)
    if len(live) < 3:
        return None
    feats = build_features(pd.concat([history, live], ignore_index=True))
    feats = feats[feats.race_id == live.race_id.iloc[0]].reset_index(drop=True)
    p = win_probs(model, feats)
    g = model.get("exotics") or {"g2": 0.87, "g3": 0.65}
    place, qpl = place_and_qpl(p, g["g2"], g["g3"])
    nos = feats["horse_no"].astype(int).tolist()
    known = set(history["horse_id"].astype(str))
    jockeys = set(history["jockey_id"].astype(str))
    unknown_jockeys = sum(str(j) not in jockeys for j in feats["jockey_id"])

    rules = model.get("rules") or {}
    bets = []

    def consider(kind, sel, chance, odds, rule):
        if not rule or not odds or odds <= 1:
            return
        edge = chance * odds - 1
        if edge > rule["min_edge"] and odds <= rule["max_odds"]:
            bets.append(dict(type=kind, sel=sel, odds=odds, chance=chance, edge=edge,
                             kelly=model["kelly"]["fraction"] * edge / (odds - 1), validated=rule.get("validated", False)))

    for i, no in enumerate(nos):
        consider("WIN", (no,), p[i], win.get(no), rules.get("win"))
        consider("PLACE", (no,), place[i], (book.get("PLA") or {}).get(no), rules.get("place"))
    for (a, b), odds in (book.get("QPL") or {}).items():
        if a in nos and b in nos:
            consider("QPL", (a, b), qpl[nos.index(a), nos.index(b)], odds, rules.get("qpl"))

    tot = sum(b["kelly"] for b in bets)
    cap = model["kelly"]["max_race_exposure"]
    for b in bets:
        b["stake"] = bankroll * b["kelly"] * (min(1.0, cap / tot) if tot else 0)
    bets.sort(key=lambda b: -b["edge"])
    runners = pd.DataFrame(dict(no=nos, horse=feats["horse_name"], win_odds=feats["win_odds"],
                                public=feats["public_p"], model=p, place=place,
                                place_odds=[(book.get("PLA") or {}).get(n) for n in nos],
                                history=[str(h) in known for h in feats["horse_id"]]))
    return dict(runners=runners.sort_values("model", ascending=False), bets=bets,
                warn=f"{unknown_jockeys} of {len(nos)} jockeys not found in history (name format mismatch?)" if unknown_jockeys > len(nos) / 2 else None)


def fmt_race(race: dict, res: dict, bankroll: float) -> str:
    lines = [f"\nRace {race['no']}  {race.get('distance')}m  {race.get('raceClass_en') or ''}  post {str(race.get('postTime') or '')[11:16]}"]
    if not res:
        return lines[0] + "\n  (no odds yet)"
    if res.get("warn"):
        lines.append("  WARNING " + res["warn"])
    if res["bets"]:
        for b in res["bets"][:8]:
            sel = "-".join(map(str, b["sel"]))
            tag = "" if b["validated"] else "  [rule not backtested]"
            lines.append(f"  BET {b['type']:5s} {sel:>5s}  odds {b['odds']:6.1f}  chance {b['chance']*100:5.1f}%  "
                         f"edge {b['edge']*100:+5.0f}%  stake HK${b['stake']:,.0f}{tag}")
    else:
        lines.append("  no bet clears the rules")
    lines.append("   no  horse                 win    public  model   place%  pla odds")
    for _, r in res["runners"].head(6).iterrows():
        flag = "" if r.history else "  (no history: treated as debut)"
        po = f"{r.place_odds:7.1f}" if r.place_odds else "      -"
        lines.append(f"  {int(r.no):3d}  {str(r.horse)[:20]:20s} {r.win_odds:6.1f}  {r.public*100:5.1f}%  {r.model*100:5.1f}%  "
                     f"{r.place*100:5.1f}%  {po}{flag}")
    return "\n".join(lines)


def log_snapshot(meeting: dict, books: dict):
    path = HIST / "odds" / f"{meeting['date'][:10]}_{meeting['venueCode']}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    ser = {str(r): {k: {("-".join(map(str, s)) if isinstance(s, tuple) else str(s)): o for s, o in v.items()} for k, v in b.items()}
           for r, b in books.items()}
    with path.open("a") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "books": ser}) + "\n")


def run_once(api: HKJCGraphQL, history, model, bankroll, race_filter=None, date=None, venue=None, log=True, out=sys.stdout):
    meeting = api.meeting(date, venue)
    if not meeting:
        print("No local HKJC meeting is open right now.", file=out)
        return {}
    races = [r for r in meeting.get("races") or [] if not race_filter or int(r["no"]) == race_filter]
    dt, vc = meeting["date"][:10], meeting["venueCode"]
    books = parse_pools(api.odds(dt, vc, 0, ["WIN", "PLA"]))
    for r in races:
        books.setdefault(int(r["no"]), {}).update(parse_pools(api.odds(dt, vc, int(r["no"]), ["QPL"])).get(int(r["no"]), {}))
    if log:
        log_snapshot(meeting, books)
    print(f"\n=== {dt} {vc}  odds at {datetime.now():%H:%M:%S}  bankroll HK${bankroll:,.0f} ===", file=out)
    results = {}
    for r in races:
        res = score_race(history, model, meeting, r, books.get(int(r["no"]), {}), bankroll)
        results[int(r["no"])] = res
        print(fmt_race(r, res, bankroll), file=out)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", type=Path, default=HIST)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--bankroll", type=float, default=10000)
    ap.add_argument("--race", type=int)
    ap.add_argument("--date", help="YYYY-MM-DD (default: current meeting)")
    ap.add_argument("--venue", choices=["ST", "HV"])
    ap.add_argument("--interval", type=int, default=60, help="seconds between refreshes")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    if not (a.history / "runs.csv").exists():
        raise SystemExit(f"no history in {a.history}; run python -m hkjc.results --start 2016-09-01 first")
    model, mpath = load_model(a.model)
    print(f"model {mpath}; history {a.history}")
    history = load_raw(a.history)
    api = HKJCGraphQL()
    while True:
        run_once(api, history, model, a.bankroll, a.race, a.date, a.venue)
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
