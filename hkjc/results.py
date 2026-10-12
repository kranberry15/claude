"""Scrape HKJC local results pages into CSVs the model can train on.

    python -m hkjc.results --start 2016-09-01 --end 2026-07-31
    python -m benter.run --data data/hkjc

For each day and course (ST, HV) it fetches race 1's results page; if a
meeting ran it walks races 2..11.  Raw HTML is cached under
data/hkjc/html/ so a rerun or a crash costs nothing.  Writes:

  data/hkjc/races.csv      one row per race (+ place dividends)
  data/hkjc/runs.csv       one row per runner, same columns the Kaggle data uses
  data/hkjc/dividends.csv  every pool's official dividend per HK$10, incl. QPL

Results pages carry no rating, age or pre-race place odds; the live
recommender takes those from the race card and odds feed instead.
Respect HKJC's terms of use: personal research, one request a second.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import date, datetime, timedelta
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlencode

import pandas as pd

from .net import Fetcher

BASE = "https://racing.hkjc.com/en-us/local/information/localresults"
OUT = Path(__file__).resolve().parent.parent / "data" / "hkjc"
VENUES = {"ST": "SHA TIN", "HV": "HAPPY VALLEY"}
NOT_RUN = {"WV", "WV-A", "WX", "WX-A", "WXNR", "WR", "SCR"}       # withdrawn / scratched
DNF_POS = 14                                                         # PU, UR, FE, DNF, DISQ, TNP...
MARGINS = {"N": 0.05, "NOSE": 0.05, "SH": 0.1, "HD": 0.2, "NK": 0.3, "---": 0.0, "": 0.0}


def result_url(d: date, venue: str, race_no: int) -> str:
    return f"{BASE}?" + urlencode({"racedate": d.strftime("%Y/%m/%d"), "Racecourse": venue, "RaceNo": race_no})


class _Tables(HTMLParser):
    """Collects every table row as (cell texts, cell hrefs), tagged with the
    classes of the enclosing elements, plus the page's plain text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, str]] = []
        self.rows: list[dict] = []
        self.text: list[str] = []
        self.links: list[str] = []
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("br", "img", "input", "meta", "link", "hr"):
            return
        self.stack.append((tag, a.get("class") or ""))
        if tag == "a" and a.get("href"):
            self.links.append(a["href"])
            if self._cell is not None:
                self._cell["href"].append(a["href"])
        if tag == "tr":
            self._row = {"cells": [], "ctx": " ".join(c for _, c in self.stack).lower(), "in_tbody": any(t == "tbody" for t, _ in self.stack)}
        elif tag in ("td", "th") and self._row is not None:
            self._cell = {"text": [], "href": [], "th": tag == "th"}

    def handle_endtag(self, tag):
        while self.stack:
            t, _ = self.stack.pop()
            if t == tag:
                break
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row["cells"].append({"text": " ".join("".join(self._cell["text"]).split()), "href": self._cell["href"], "th": self._cell["th"]})
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        self.text.append(data)
        if self._cell is not None:
            self._cell["text"].append(data)


def parse_lbw(s: str) -> float | None:
    s = s.strip().upper()
    if s in MARGINS:
        return MARGINS[s]
    m = re.fullmatch(r"(\d+)-(\d+)/(\d+)", s)
    if m:
        return int(m[1]) + int(m[2]) / int(m[3])
    m = re.fullmatch(r"(\d+)/(\d+)", s)
    if m:
        return int(m[1]) / int(m[2])
    try:
        return float(s)
    except ValueError:
        return None


def parse_time(s: str) -> float | None:
    m = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", s.strip()) or re.fullmatch(r"(\d+)\.(\d{2}\.\d+)", s.strip())
    if m:
        return int(m[1]) * 60 + float(m[2])
    try:
        return float(s)
    except ValueError:
        return None


def _num(s: str):
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return float(m[0]) if m else None


def parse_result_page(html: str, d: date, venue: str, race_no: int) -> dict | None:
    """Parse one results page.  None if it isn't the requested race (HKJC
    serves the latest meeting for dates without racing)."""
    p = _Tables()
    p.feed(html)
    text = " ".join(" ".join(p.text).split())

    # which meeting does this page show?  most common racedate/course in its own links
    seen = Counter()
    for href in p.links:
        if "localresults" in href.lower() and "?" in href:
            q = {k.lower(): v[0] for k, v in parse_qs(unescape(href.split("?", 1)[1])).items()}
            if "racedate" in q:
                seen[(q["racedate"].replace("-", "/"), (q.get("racecourse") or "").upper())] += 1
    if seen:
        (pd_, pv), _ = seen.most_common(1)[0]
        try:
            page_date = datetime.strptime(pd_, "%Y/%m/%d").date() if pd_[:4].isdigit() else datetime.strptime(pd_, "%d/%m/%Y").date()
        except ValueError:
            page_date = None
        if page_date and page_date != d:
            return None
        if pv and pv != venue:
            return None
    title = re.search(r"RACE\s+(\d+)\s*\((\d+)\)", text, re.I)
    if title and int(title[1]) != race_no:
        return None

    runners = []
    for row in p.rows:
        c = [x["text"] for x in row["cells"]]
        if "performance" not in row["ctx"] or not row["in_tbody"] or len(c) < 12 or row["cells"][0]["th"]:
            continue
        place = c[0].upper().replace("DH", "").strip()
        if place in NOT_RUN:
            continue
        horse_id = None
        for h in row["cells"][2]["href"]:
            m = re.search(r"horseid=([^&\s]+)", h, re.I)
            if m:
                horse_id = m[1]
        brand = re.search(r"\(([A-Z]\d{3})\)", c[2])
        result = int(place) if place.isdigit() else DNF_POS
        runners.append(dict(
            horse_no=int(_num(c[1])) if _num(c[1]) is not None else None,
            horse_id=brand[1] if brand else horse_id,
            horse_name=re.sub(r"\s*\([A-Z]\d{3}\)\s*$", "", c[2]).strip(),
            jockey_id=c[3].strip(), trainer_id=c[4].strip(),
            actual_weight=_num(c[5]), declared_weight=_num(c[6]), draw=_num(c[7]),
            lengths_behind=parse_lbw(c[8]) if place.isdigit() else 30.0,
            finish_time=parse_time(c[10]), win_odds=_num(c[11]),
            result=result, dead_heat="DH" in c[0].upper(),
        ))
    runners = [r for r in runners if r["horse_no"] and r["win_odds"]]
    if not runners:
        return None

    dividends, pool = [], None
    for row in p.rows:
        if "dividend" not in row["ctx"] or not row["in_tbody"]:
            continue
        c = [x["text"] for x in row["cells"]]
        if len(c) >= 3:
            pool, comb, div = c[0].upper(), c[1], c[2]
        elif len(c) == 2 and pool:
            comb, div = c
        else:
            continue
        nums = [int(x) for x in re.findall(r"\d+", comb)]
        val = _num(div)
        if nums and val is not None:
            dividends.append(dict(pool=pool, combination="-".join(map(str, nums)), dividend=val))

    cls = re.search(r"Class\s+(\d+)\s*-\s*(\d+)\s*M", text, re.I) or re.search(r"Group\s+\w+\s*-\s*(\d+)\s*M", text, re.I)
    dist = re.search(r"-\s*(\d{3,4})\s*M\b", text)
    going = re.search(r"Going\s*:\s*([A-Z ]+?)(?:\s{2,}|Course|$)", text, re.I)
    course = re.search(r"Course\s*:\s*([^:]+?)(?:\s{2,}|HK\$|Time|$)", text, re.I)
    return dict(
        date=d.isoformat(), venue=venue, race_no=race_no,
        race_index=int(title[2]) if title else None,
        race_class=int(cls[1]) if cls and cls.lastindex == 2 else 0,
        distance=int(dist[1]) if dist else None,
        going=going[1].strip().upper() if going else None,
        surface=1 if course and "ALL WEATHER" in course[1].upper() else 0,
        course=course[1].strip() if course else None,
        runners=runners, dividends=dividends,
    )


def scrape(start: date, end: date, out: Path = OUT, fetcher: Fetcher | None = None, log=print):
    fetcher = fetcher or Fetcher(min_interval=1.0, cache_dir=out / "html")
    out.mkdir(parents=True, exist_ok=True)
    races = []
    d = start
    while d <= end:
        for venue in VENUES:
            for race_no in range(1, 12):
                url = result_url(d, venue, race_no)
                # never cache today's pages: results may still be coming in
                html = fetcher.get_text(url, use_cache=d < date.today())
                r = parse_result_page(html, d, venue, race_no)
                if r is None:
                    break
                races.append(r)
            if races and races[-1]["date"] == d.isoformat() and races[-1]["venue"] == venue:
                log(f"{d} {venue}: {races[-1]['race_no']} races")
        if d.day == 1 and races:
            write(races, out)        # checkpoint monthly
        d += timedelta(days=1)
    write(races, out)
    return races



def write(races: list[dict], out: Path = OUT):
    """Merge into the CSVs (re-scraped races replace older rows)."""
    rr, ru, dv = [], [], []
    for r in races:
        rid = f"{r['date']}-{r['venue']}-{r['race_no']}"
        top3 = [x for x in (d for d in r["dividends"] if d["pool"] == "PLACE")][:3]
        rr.append(dict(race_id=rid, **{k: r[k] for k in ("date", "venue", "race_no", "race_index", "distance", "race_class", "surface", "going", "course")},
                       **{f"place_combination{i + 1}": int(x["combination"]) for i, x in enumerate(top3) if x["combination"].isdigit()},
                       **{f"place_dividend{i + 1}": x["dividend"] for i, x in enumerate(top3)}))
        for x in r["runners"]:
            ru.append(dict(race_id=rid, won=float(x["result"] == 1), **{k: v for k, v in x.items() if k != "dead_heat"}))
        for x in r["dividends"]:
            dv.append(dict(race_id=rid, **x))
    for name, rows in (("races", rr), ("runs", ru), ("dividends", dv)):
        new = pd.DataFrame(rows)
        path = out / f"{name}.csv"
        if path.exists() and len(new):
            old = pd.read_csv(path)
            new = pd.concat([old[~old.race_id.isin(new.race_id)], new])
        if len(new):
            new.to_csv(path, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, default=date.today())
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    a = ap.parse_args(argv)
    scrape(a.start, a.end, a.out, Fetcher(min_interval=a.delay, cache_dir=a.out / "html"))


if __name__ == "__main__":
    main()
