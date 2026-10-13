"""Real HKJC data for weekend_slip: past results and the upcoming race card.

    python -m benter.hkjc_data            # download / top up past results
    python -m benter.hkjc_data --card     # also show the next meeting's card

Past results come from the public results pages on racing.hkjc.com, one page
per race, at most one request per --delay seconds (default 1).  Each meeting
is saved as data/hkjc/results/<YYYY-MM-DD>.csv and never downloaded again,
so later runs only fetch new meetings.  The first run takes about 30 minutes.

The upcoming card comes from the GraphQL gateway behind bet.hkjc.com, using
the exact query the site sends (benter/queries/race_meetings.graphql, taken
from the MIT-licensed hkjc-api client - don't edit it, the gateway rejects
any other text).

Class proxy: results pages don't show official ratings, so `rating` is set
to the weight carried, in the history and on the card alike.  In a handicap
the weight is the rating plus a constant, so rating_rel still measures class.
"""
from __future__ import annotations

import argparse
import html as htmllib
import io
import json
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from benter import live_odds as lo

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "hkjc"
RESULTS_URL = "https://racing.hkjc.com/en-us/local/information/localresults"
MEETING_QUERY = (Path(__file__).parent / "queries" / "race_meetings.graphql").read_text()
UA = "benter-results/1.0 (personal research; read-only)"
VENUES = {"Sha Tin": "ST", "Happy Valley": "HV"}
COLUMNS = ["race_id", "date", "venue", "race_no", "distance", "horse_id", "horse_no", "horse_name", "jockey_id",
           "trainer_id", "rating", "weight_carried", "finish_position", "finish_time", "win_odds",
           "draw", "body_weight", "lbw", "race_class", "going", "surface", "course"]
GOINGS = ["GOOD TO FIRM", "GOOD TO YIELDING", "YIELDING TO SOFT", "WET FAST", "WET SLOW", "GOOD", "YIELDING",
          "SOFT", "HEAVY", "FAST", "SLOW"]


def http_get(url: str, tries: int = 4) -> str:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError) as e:
            if k == tries - 1:
                raise
            time.sleep(5 * 2 ** k)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Results pages
# ---------------------------------------------------------------------------


def meeting_dates(html: str) -> list[date]:
    """Race dates listed in the results page's date picker (newest first)."""
    out = []
    for v in re.findall(r"<option value='(\{[^']*\})'", html):
        try:
            out.append(datetime.strptime(json.loads(v)["date"], "%d/%m/%Y").date())
        except (ValueError, KeyError):
            pass
    return sorted(set(out), reverse=True)


def _seconds(t) -> float:
    m = re.fullmatch(r"(?:(\d+):)?(\d+(?:\.\d+)?)", str(t).strip())
    return float(m.group(1) or 0) * 60 + float(m.group(2)) if m else np.nan


def _position(p) -> float | None:
    """'1' -> 1, '2 DH' -> 2, 'PU' / 'DISQ' / 'UR' -> 99 (ran, didn't finish), 'WV...' / 'WX' -> None (withdrawn)."""
    s = str(p).strip().upper()
    if s.startswith("W") or s in ("", "NAN"):
        return None
    m = re.match(r"(\d+)", s)
    return float(m.group(1)) if m else 99.0


def _lengths(x) -> float:
    """Lengths behind the winner: '---' 0, 'N' nose, 'SH', 'HD', 'NK', '1-3/4', '3/4', '12'."""
    s = str(x).strip().upper()
    special = {"---": 0.0, "N": 0.05, "SH": 0.1, "HD": 0.2, "NK": 0.3}
    if s in special:
        return special[s]
    m = re.fullmatch(r"(?:(\d+)-)?(\d+)/(\d+)|(\d+(?:\.\d+)?)", s)
    if not m:
        return np.nan
    if m.group(4):
        return min(float(m.group(4)), 30.0)
    return min(float(m.group(1) or 0) + float(m.group(2)) / float(m.group(3)), 30.0)


def race_info(tail: str) -> dict:
    """Class, distance, going, surface and course from the text after 'RACE n (k)'."""
    head = re.match(r"\s*(.*?) - (\d{3,4})M", tail)
    going = re.search(r"Going : (" + "|".join(GOINGS) + r")\b", tail)
    course = re.search(r'Course : (TURF|ALL WEATHER TRACK)(?: - "([^"]+)" Course)?', tail)
    return dict(race_class=head.group(1).strip() if head else None,
                distance=int(head.group(2)) if head else np.nan,
                going=going.group(1) if going else None,
                surface=("AWT" if course.group(1).startswith("ALL") else "TURF") if course else None,
                course=(course.group(2) if course and course.group(2) else None))


def parse_race_page(html: str, day: date) -> dict | None:
    """One race's result.  None if the page holds no local result."""
    text = re.sub(r"\s+", " ", htmllib.unescape(re.sub(r"<[^>]+>", " ", html)))
    venue = next((code for name, code in VENUES.items() if f"{name}:" in text or f"{name} :" in text), None)
    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except ValueError:  # no tables on the page
        return None
    res = next((t for t in tables if "Finish Time" in map(str, t.columns)), None)
    head = re.search(r"RACE (\d+) \(\d+\)", text)
    if venue is None or res is None or head is None:
        return None
    race_no = int(head.group(1))
    info = race_info(text[head.end():head.end() + 500])
    rows = []
    for _, r in res.iterrows():
        pos = _position(r["Pla."])
        if pos is None:
            continue
        m = re.match(r"(.*?)\s*\(([A-Z]\d{3})\)", str(r["Horse"]))
        if not m:
            continue
        odds = pd.to_numeric(r.get("Win Odds"), errors="coerce")
        rows.append(dict(
            race_id=lo.race_id(day.isoformat(), venue, race_no), date=day.isoformat(), venue=venue, race_no=race_no,
            distance=info["distance"], horse_id=m.group(2), horse_no=r["Horse No."],
            horse_name=m.group(1).strip(), jockey_id=str(r["Jockey"]).strip(), trainer_id=str(r["Trainer"]).strip(),
            rating=pd.to_numeric(r["Act. Wt."], errors="coerce"),
            weight_carried=pd.to_numeric(r["Act. Wt."], errors="coerce"),
            finish_position=pos, finish_time=_seconds(r["Finish Time"]) if pos < 99 else np.nan,
            win_odds=odds, draw=pd.to_numeric(r.get("Dr."), errors="coerce"),
            body_weight=pd.to_numeric(r.get("Declar. Horse Wt."), errors="coerce"), lbw=_lengths(r.get("LBW")),
            race_class=info["race_class"], going=info["going"], surface=info["surface"], course=info["course"],
        ))
    races = sorted({int(x) for x in re.findall(r"RaceNo=(\d+)", html)} | {race_no})
    return dict(venue=venue, race_no=race_no, races=races, rows=rows,
                dividends=parse_dividends(html, lo.race_id(day.isoformat(), venue, race_no)))


def parse_dividends(html: str, rid: str) -> list[tuple]:
    """(race_id, bet_type, selection, dividend per HKD 10) for Win / Place / Quinella.
    Read with thousands=None so a quinella '2,3' isn't turned into 23."""
    try:
        t = pd.read_html(io.StringIO(html), flavor="lxml", match="Winning Combination", thousands=None)[0]
    except (ValueError, IndexError):
        return []
    t.columns = [c[-1] if isinstance(c, tuple) else c for c in t.columns]
    t["Pool"] = t["Pool"].ffill()
    out = []
    for _, r in t.iterrows():
        bet = {"WIN": "WIN", "PLACE": "PLACE", "QUINELLA": "QUINELLA"}.get(str(r["Pool"]).strip().upper())
        div = pd.to_numeric(str(r["Dividend (HK$)"]).replace(",", ""), errors="coerce")
        sel = lo.selection(r["Winning Combination"])
        if bet and sel and div == div:
            out.append((rid, bet, sel, float(div)))
    return out


def scrape_meeting(day: date, delay: float = 1.0, log=print) -> pd.DataFrame | None:
    first = parse_race_page(http_get(f"{RESULTS_URL}?racedate={day:%Y/%m/%d}"), day)
    if first is None:
        return None
    rows, divs = list(first["rows"]), list(first["dividends"])
    for k in first["races"]:
        if k == first["race_no"]:
            continue
        time.sleep(delay)
        page = parse_race_page(http_get(f"{RESULTS_URL}?racedate={day:%Y/%m/%d}&Racecourse={first['venue']}&RaceNo={k}"), day)
        if page:
            rows += page["rows"]
            divs += page["dividends"]
    log(f"  {day} {first['venue']}: {len(first['races'])} races, {len(rows)} runners")
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.attrs["dividends"] = pd.DataFrame(divs, columns=["race_id", "bet_type", "selection", "dividend"])
    return df


def _stale(path: Path) -> bool:
    """Saved before the extra columns (draw, going, ...) were collected."""
    with path.open() as f:
        return "draw" not in f.readline()


def update_results(data_dir: Path = DATA, delay: float = 1.0, max_meetings: int | None = None, log=print,
                   since: date | None = None, refresh: bool = False) -> int:
    """Download every listed meeting not already saved.  Returns how many were added.

    The results page only lists about a year of dates.  With `since`, every
    calendar day from `since` up to the oldest listed date is tried as well
    (except the late-July / August break); days without a meeting are
    remembered as <date>.none.  With `refresh`, meetings saved in the old
    format are downloaded again."""
    out = data_dir / "results"
    out.mkdir(parents=True, exist_ok=True)
    dates = meeting_dates(http_get(RESULTS_URL))
    if since and dates:
        d, oldest = min(dates) - timedelta(days=1), set(dates)
        while d >= since:
            if not (d.month == 8 or (d.month == 7 and d.day > 20)) and d not in oldest:
                dates.append(d)
            d -= timedelta(days=1)

    def needed(d):
        f = out / f"{d}.csv"
        if f.exists():
            return refresh and _stale(f)
        return not (out / f"{d}.none").exists()

    todo = [d for d in dates if d < date.today() and needed(d)]
    if max_meetings:
        todo = todo[:max_meetings]
    log(f"results: {len(dates)} meetings listed, {len(todo)} to download")
    added = 0
    for d in todo:
        time.sleep(delay)
        try:
            df = scrape_meeting(d, delay, log)
        except Exception as e:  # one bad page shouldn't stop the run; it is retried next time
            log(f"  {d}: skipped ({type(e).__name__}: {str(e)[:120]})")
            continue
        if df is None or df.empty:
            (out / f"{d}.none").touch()  # e.g. a date with no local results
            continue
        (data_dir / "dividends").mkdir(exist_ok=True)
        df.attrs["dividends"].to_csv(data_dir / "dividends" / f"{d}.csv", index=False)
        df.to_csv(out / f"{d}.csv", index=False)
        added += 1
    return added


def load_results(data_dir: Path = DATA) -> pd.DataFrame:
    files = sorted((data_dir / "results").glob("*.csv"))
    if not files:
        raise FileNotFoundError("no results downloaded yet - run: python -m benter.hkjc_data")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Upcoming card
# ---------------------------------------------------------------------------


def next_meeting(user_agent: str | None = None) -> dict | None:
    """The meeting the gateway is serving (today's, or the next one)."""
    m = lo.post({"date": None, "venueCode": None}, MEETING_QUERY, user_agent=user_agent, operation="raceMeetings")
    return m[0] if m else None


def fetch_card_meeting(day: str, venue: str, user_agent: str | None = None) -> dict | None:
    data = lo.post({"date": day, "venueCode": venue}, MEETING_QUERY, user_agent=user_agent, operation="raceMeetings")
    meetings = [m for m in data if m.get("date") == day and m.get("venueCode") == venue]
    return meetings[0] if meetings else None


def card_frame(meeting: dict) -> pd.DataFrame:
    """Runners of a meeting in weekend_slip's schema (results left empty)."""
    day, venue = meeting["date"], meeting["venueCode"]
    rows = []
    for r in meeting.get("races") or []:
        for h in r.get("runners") or []:
            if (h.get("standbyNo") or "").strip() or "scratch" in (h.get("status") or "").lower():
                continue
            hw = pd.to_numeric(h.get("handicapWeight"), errors="coerce")
            allow = pd.to_numeric((h.get("allowance") or "").strip() or 0, errors="coerce")
            wt = hw - (allow if allow == allow else 0)
            track = ((r.get("raceTrack") or {}).get("description_en") or "").upper()
            rows.append(dict(
                draw=pd.to_numeric(h.get("barrierDrawNumber"), errors="coerce"),
                body_weight=pd.to_numeric(h.get("currentWeight"), errors="coerce"), lbw=np.nan,
                race_class=r.get("raceClass_en") or None, going=(r.get("go_en") or "").upper() or None,
                surface="AWT" if "ALL" in track else ("TURF" if track else None),
                course=(r.get("raceCourse") or {}).get("displayCode") or None,
                race_id=lo.race_id(day, venue, int(r["no"])), date=day, venue=venue, race_no=int(r["no"]),
                distance=pd.to_numeric(r.get("distance"), errors="coerce"),
                horse_id=(h.get("horse") or {}).get("code"), horse_no=int(h["no"]), horse_name=h.get("name_en"),
                jockey_id=(h.get("jockey") or {}).get("name_en"), trainer_id=(h.get("trainer") or {}).get("name_en"),
                rating=wt, weight_carried=wt, finish_position=np.nan, finish_time=np.nan,
                win_odds=lo.parse_odds_value(h.get("winOdds")),
            ))
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between page requests")
    ap.add_argument("--max-meetings", type=int)
    ap.add_argument("--since", type=date.fromisoformat, help="also try every day back to this date (YYYY-MM-DD)")
    ap.add_argument("--refresh", action="store_true", help="re-download meetings saved in the old format")
    ap.add_argument("--card", action="store_true")
    a = ap.parse_args(argv)
    n = update_results(delay=a.delay, max_meetings=a.max_meetings, since=a.since, refresh=a.refresh)
    df = load_results()
    print(f"added {n} meetings; history now {df.race_id.nunique()} races, "
          f"{df.date.min().date()} to {df.date.max().date()}")
    if a.card:
        m = next_meeting()
        print(card_frame(m).to_string() if m else "no meeting served")


if __name__ == "__main__":
    main()
