"""Checks the tracker's JavaScript reproduces the Python model.

Builds race-card inputs (as a user would type them) from the history of a
few held-out races, runs the page's scoring code in node, and compares the
market-anchored probabilities with benter.run's.   python -m tests.parity
"""
import json
import subprocess
from pathlib import Path

import numpy as np

from benter.features import FEATURES, load_features
from benter.model import softmax_by_race

ROOT = Path(__file__).resolve().parent.parent
M = json.loads((ROOT / "tracker" / "model.json").read_text())


def card_inputs(df, race_id, exact_lb):
    race = df[df.race_id == race_id]
    out = []
    for _, r in race.iterrows():
        hist = df[(df.horse_id == r.horse_id) & ((df.date < r.date) | ((df.date == r.date) & (df.race_no < r.race_no)))]
        jk = df[(df.jockey_id == r.jockey_id) & ((df.date < r.date) | ((df.date == r.date) & (df.race_no < r.race_no)))]
        tr = df[(df.trainer_id == r.trainer_id) & ((df.date < r.date) | ((df.date == r.date) & (df.race_no < r.race_no)))]
        trw = tr.groupby(["date", "race_no"]).won.sum().sum()
        trn = len(tr)
        out.append(dict(
            no=int(r.horse_no), name="", draw=int(r.draw), rtg=float(r.horse_rating), wt=float(r.actual_weight),
            bw=float(r.declared_weight), pbw=float(hist.declared_weight.iloc[-1]) if len(hist) else "",
            age=float(r.horse_age), days=int((r.date - hist.date.iloc[-1]).days) if len(hist) else 0,
            rec=f"{len(hist)}-{int(hist.won.sum())}-{int((hist.result <= 3).sum())}",
            last6="/".join(str(int(p)) for p in hist.result.tail(6)),
            lodds="/".join(str(o) for o in hist.win_odds.tail(3)),
            lb3=float(r.avg_lb3) if exact_lb else "",
            jky=f"{int(jk.won.sum())}/{len(jk)}", trn=f"{int(trw)}/{trn}", odds=float(r.win_odds),
        ))
    race_info = dict(venue=race.venue.iloc[0], dist=int(race.distance.iloc[0]))
    return race_info, out, race


def main():
    df = load_features()
    test = df[df.date >= M["periods"]["test"][0]]
    cols = M["market"]["columns"]
    html = (ROOT / "tracker" / "index.html").read_text()
    js = html[html.index("// ---------- feature construction"):html.index("// ---------- race card UI")]
    worst = {True: 0.0, False: 0.0}
    for rid, exact_lb in [(r, e) for r in test.race_id.drop_duplicates().sample(6, random_state=1) for e in (True, False)]:
        info, runners, race = card_inputs(df, rid, exact_lb)
        X = race[cols].values
        u = ((X - np.array(M["market"]["mean"])) / np.array(M["market"]["std"])) @ np.array(M["market"]["coef"])
        py = softmax_by_race(u, race.race_id.values)
        script = f"""
const M = {json.dumps(M)};
let runners = {json.dumps(runners)};
const $ = s => ({{ "#s-edge": {{ value: "0.2" }}, "#s-maxodds": {{ value: "30" }} }})[s];
{js}
console.log(JSON.stringify(score({json.dumps(info)}).map(o => o.model)));
"""
        js_p = np.array(json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout))
        diff = np.abs(js_p - py).max()
        worst[exact_lb] = max(worst[exact_lb], diff)
        print(f"race {rid}: {len(runners)} runners, exact lengths={exact_lb}, max |js - py| = {diff:.2e}")
    # Exact inputs must match; with lengths-beaten imputed from positions the
    # probabilities drift a little.
    assert worst[True] < 1e-6, worst
    assert worst[False] < 1e-2, worst
    print("parity ok")


if __name__ == "__main__":
    main()
