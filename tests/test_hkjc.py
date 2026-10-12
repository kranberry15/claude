"""Offline tests for the scraper and live recommender (no network).

    python -m tests.test_hkjc

1. The results-page parser on a page shaped like HKJC's.
2. The live pipeline end to end: a fake GraphQL feed built from a real
   held-out Kaggle race must give the same win chances as benter.run's
   model, and sane place / QPL chances and bets.
"""
import io
import json
from datetime import date
from pathlib import Path

import numpy as np

from benter.features import load_features, load_raw
from hkjc.graphql import HKJCGraphQL, parse_pools
from hkjc.live import run_once
from hkjc.results import parse_result_page

ROOT = Path(__file__).resolve().parent.parent

PAGE = """<html><body>
<a href="/en-us/local/information/localresults?racedate=2025/03/09&Racecourse=ST&RaceNo=1">1</a>
<a href="/en-us/local/information/localresults?racedate=2025/03/09&Racecourse=ST&RaceNo=2">2</a>
<div class="race_tab"><table><thead><tr><td>RACE 2 (512)</td></tr></thead><tbody>
<tr><td>Class 4 - 1200M - (60-40)</td><td>Going :</td><td>GOOD</td></tr>
<tr><td>Course :</td><td>TURF - "A" COURSE</td></tr></tbody></table></div>
<div class="performance"><table><thead><tr><td>Pla.</td></tr></thead><tbody>
<tr><td>1</td><td>7</td><td><a href="/horse?horseid=HK_2022_H123">LUCKY STAR (H123)</a></td><td>Z Purton</td><td>J Size</td>
<td>128</td><td>1,105</td><td>3</td><td>-</td><td>2 2 1</td><td>1:09.45</td><td>4.2</td></tr>
<tr><td>2 DH</td><td>2</td><td><a href="/horse?horseid=HK_2021_G045">GOLDEN ACE (G045)</a></td><td>H Bowman</td><td>F C Lor</td>
<td>133</td><td>1,180</td><td>9</td><td>1-1/4</td><td>5 4 2</td><td>1:09.65</td><td>6.5</td></tr>
<tr><td>3</td><td>11</td><td><a href="/horse?horseid=HK_2020_E999">SKY HIGH (E999)</a></td><td>A Badel</td><td>P F Yiu</td>
<td>118</td><td>1,050</td><td>1</td><td>SH</td><td>1 1 3</td><td>1:09.66</td><td>23</td></tr>
<tr><td>PU</td><td>4</td><td><a href="/horse?horseid=HK_2022_H555">SLOW ONE (H555)</a></td><td>K Teetan</td><td>D Whyte</td>
<td>125</td><td>1,099</td><td>5</td><td>---</td><td></td><td>---</td><td>40</td></tr>
<tr><td>WV</td><td>9</td><td><a href="/horse?horseid=HK_2022_H777">NON RUNNER (H777)</a></td><td>-</td><td>-</td>
<td>120</td><td>1,000</td><td>8</td><td>---</td><td></td><td>---</td><td>---</td></tr>
</tbody></table></div>
<div class="dividend_tab"><table><thead><tr><td>Pool</td></tr></thead><tbody>
<tr><td>WIN</td><td>7</td><td>42.00</td></tr>
<tr><td>PLACE</td><td>7</td><td>16.50</td></tr><tr><td>2</td><td>21.00</td></tr><tr><td>11</td><td>60.50</td></tr>
<tr><td>QUINELLA PLACE</td><td>2,7</td><td>32.00</td></tr><tr><td>7,11</td><td>140.50</td></tr><tr><td>2,11</td><td>180.00</td></tr>
</tbody></table></div></body></html>"""


def test_results_parser():
    r = parse_result_page(PAGE, date(2025, 3, 9), "ST", 2)
    assert r is not None
    assert (r["distance"], r["race_class"], r["going"], r["surface"]) == (1200, 4, "GOOD", 0), r
    by_no = {x["horse_no"]: x for x in r["runners"]}
    assert set(by_no) == {7, 2, 11, 4}, by_no.keys()          # WV dropped, PU kept as a non-finisher
    assert by_no[7]["horse_id"] == "H123" and by_no[7]["result"] == 1 and by_no[7]["declared_weight"] == 1105
    assert by_no[2]["result"] == 2 and abs(by_no[2]["lengths_behind"] - 1.25) < 1e-9
    assert by_no[11]["lengths_behind"] == 0.1 and abs(by_no[11]["finish_time"] - 69.66) < 1e-9
    assert by_no[4]["result"] == 14
    q = [d for d in r["dividends"] if d["pool"] == "QUINELLA PLACE"]
    assert [d["combination"] for d in q] == ["2-7", "7-11", "2-11"] and q[0]["dividend"] == 32.0
    # a page for another day (HKJC shows the latest meeting) is rejected
    assert parse_result_page(PAGE, date(2025, 3, 12), "ST", 2) is None
    assert parse_result_page(PAGE, date(2025, 3, 9), "HV", 2) is None
    print("results parser ok")


def fake_feed(race):
    """GraphQL responses shaped like HKJC's, built from one historical race."""
    d = race["date"].iloc[0].strftime("%Y-%m-%d")
    venue = race["venue"].iloc[0]
    no = int(race["race_no"].iloc[0])
    runners = [dict(no=str(int(r.horse_no)), status="Declared", name_en=f"HORSE {int(r.horse_no)}", horse=dict(code=str(r.horse_id)),
                    barrierDrawNumber=str(int(r.draw)), handicapWeight=str(int(r.actual_weight)), currentWeight=str(int(r.declared_weight)),
                    currentRating=str(int(r.horse_rating)), jockey=dict(name_en=str(r.jockey_id)), trainer=dict(name_en=str(r.trainer_id)))
               for r in race.itertuples()]
    meeting = dict(date=d + "+08:00", venueCode=venue,
                   races=[dict(no=no, distance=int(race.distance.iloc[0]), raceTrack=dict(description_en="TURF"),
                               postTime=d + "T13:00:00+08:00", raceClass_en="Class 4", runners=runners)])
    nodes = lambda vals: [dict(combString=f"{k:02d}" if isinstance(k, int) else f"{k[0]:02d},{k[1]:02d}", oddsValue=str(v)) for k, v in vals.items()]
    win = {int(r.horse_no): float(r.win_odds) for r in race.itertuples()}
    pla = {int(r.horse_no): float(r.place_odds) for r in race.itertuples()}
    nos = sorted(win)
    qpl = {(a, b): round(3.0 + win[a] * win[b] / 6.0, 1) for i, a in enumerate(nos) for b in nos[i + 1:]}

    def post(query, variables):
        if "activeMeetings" in query:
            return {"data": {"activeMeetings": [meeting], "raceMeetings": [meeting]}}
        types = variables.get("oddsTypes") or []
        pools = []
        if "WIN" in types:
            pools.append(dict(oddsType="WIN", leg=dict(races=[no]), oddsNodes=nodes(win)))
        if "PLA" in types:
            pools.append(dict(oddsType="PLA", leg=dict(races=[no]), oddsNodes=nodes(pla)))
        if "QPL" in types:
            pools.append(dict(oddsType="QPL", leg=dict(races=[no]), oddsNodes=nodes(qpl)))
        return {"data": {"raceMeetings": [{"pmPools": pools}]}}
    return post


def test_live_pipeline():
    model = json.loads((ROOT / "tracker" / "model.json").read_text())
    # HKJC's feed has no age; neutralise it on both sides of the comparison
    model["market"]["coef"][model["market"]["columns"].index("age")] = 0.0
    raw = load_raw(ROOT / "data")
    feats = load_features(ROOT / "data")
    test = feats[(feats.date >= model["periods"]["test"][0]) & feats.place_odds.notna()]
    checked = 0
    for rid in test.race_id.drop_duplicates().sample(5, random_state=3):
        race = raw[raw.race_id == rid]
        d, no = race.date.iloc[0], race.race_no.iloc[0]
        history = raw[(raw.date < d) | ((raw.date == d) & (raw.race_no < no))]
        out = io.StringIO()
        res = run_once(HKJCGraphQL(post=fake_feed(race)), history, model, 10000, log=False, out=out)
        r = res[int(race.race_no.iloc[0])]
        # reference: the model on the features benter.run computed for this race
        ref = feats[feats.race_id == rid]
        m = model["market"]
        X = ref[m["columns"]].astype(float).values
        u = ((X - np.array(m["mean"])) / np.array(m["std"])) @ np.array(m["coef"])
        pref = dict(zip(ref.horse_no.astype(int), np.exp(u) / np.exp(u).sum()))
        got = dict(zip(r["runners"]["no"], r["runners"]["model"]))
        diff = max(abs(got[k] - pref[k]) for k in pref)
        assert diff < 1e-9, (rid, diff)
        assert abs(r["runners"]["place"].sum() - 3.0) < 1e-9
        assert all(b["edge"] > 0 and b["stake"] >= 0 for b in r["bets"])
        assert sum(b["stake"] for b in r["bets"]) <= 10000 * model["kelly"]["max_race_exposure"] + 1e-6
        checked += 1
    print(out.getvalue())
    print(f"live pipeline ok ({checked} races match the backtested model exactly)")


def test_parse_pools():
    pools = [dict(oddsType="QPL", leg=dict(races=[3]), oddsNodes=[dict(combString="05,02", oddsValue="12.5"), dict(combString="01,02", oddsValue="SCR")]),
             dict(oddsType="WIN", leg=dict(number=3, races=[3]), oddsNodes=[dict(combString="02", oddsValue="4.1")])]
    b = parse_pools(pools)
    assert b == {3: {"QPL": {(2, 5): 12.5}, "WIN": {2: 4.1}}}, b
    print("pool parsing ok")


if __name__ == "__main__":
    test_results_parser()
    test_parse_pools()
    test_live_pipeline()
