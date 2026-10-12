"""Fit the Benter two-stage model on HKJC history, backtest it, and export
the coefficients + evidence the tracker uses.

    python -m benter.run            # writes tracker/model.json, prints report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import DEBUT_LOG_ODDS, FEATURES, PRIOR, load_features
from .model import ConditionalLogit, pseudo_r2, softmax_by_race

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tracker" / "model.json"

# Seasons run Sep..Jul.  1997/98 is a warm-up season (every horse looks
# like a debutant), fundamental model is fit on the next four, the
# public-odds blend on the one after, and the last two are untouched test.
WARMUP_END = "1998-09-01"
TRAIN_END = "2002-09-01"
BLEND_END = "2003-09-01"

KELLY_FRACTION = 0.25
MAX_RACE_EXPOSURE = 0.10


def split(df):
    d = df["date"]
    return (
        df[(d >= WARMUP_END) & (d < TRAIN_END)],
        df[(d >= TRAIN_END) & (d < BLEND_END)],
        df[d >= BLEND_END],
    )


def blend(f, pi, race_ids, alpha, beta):
    return softmax_by_race(alpha * np.log(f) + beta * np.log(pi), race_ids)


def flat_backtest(t, min_edge, max_odds=99.0):
    b = t[(t["edge"] > min_edge) & (t["win_odds"] <= max_odds)]
    if b.empty:
        return dict(min_edge=min_edge, max_odds=max_odds, bets=0)
    ret = np.where(b["won"] == 1, b["win_odds"] - 1.0, -1.0)
    # bootstrap-by-race 90% interval on ROI so small samples don't mislead
    rng = np.random.default_rng(0)
    races = b["race_id"].values
    uniq, inv = np.unique(races, return_inverse=True)
    per_race_ret = np.bincount(inv, ret)
    per_race_n = np.bincount(inv)
    boots = []
    for _ in range(500):
        s = rng.integers(0, len(uniq), len(uniq))
        boots.append(per_race_ret[s].sum() / per_race_n[s].sum())
    return dict(
        min_edge=min_edge,
        max_odds=max_odds,
        bets=int(len(b)),
        races=int(len(uniq)),
        hit_rate=float((b["won"] == 1).mean()),
        avg_odds=float(b["win_odds"].mean()),
        roi=float(ret.mean()),
        roi_lo=float(np.percentile(boots, 5)),
        roi_hi=float(np.percentile(boots, 95)),
    )


def kelly_backtest(t, min_edge=0.0, max_odds=99.0):
    bank, curve = 1.0, []
    for (date, rid), r in t.groupby(["date", "race_id"], sort=True):
        r = r[(r["edge"] > min_edge) & (r["win_odds"] <= max_odds)]
        if len(r):
            f = KELLY_FRACTION * r["edge"] / (r["win_odds"] - 1.0)
            if f.sum() > MAX_RACE_EXPOSURE:
                f *= MAX_RACE_EXPOSURE / f.sum()
            stake = f * bank
            bank += float((np.where(r["won"] == 1, stake * (r["win_odds"] - 1), -stake)).sum())
        curve.append((str(date.date()), bank))
    # one point per meeting keeps the json small
    c = pd.DataFrame(curve, columns=["date", "bank"]).groupby("date").last().reset_index()
    return c.values.tolist()


def calibration(p, won, bins):
    cut = pd.cut(p, bins)
    g = pd.DataFrame({"p": p, "w": won, "b": cut}).groupby("b", observed=True)
    return [
        dict(lo=float(i.left), hi=float(i.right), n=int(len(x)), pred=float(x["p"].mean()), actual=float(x["w"].mean()))
        for i, x in g
    ]


def evaluate(t, prob_col):
    t = t.assign(edge=t[prob_col] * t["win_odds"] - 1.0)
    grid = [flat_backtest(t, e, mo) for mo in (99.0, 30.0) for e in (-1.0, 0.0, 0.1, 0.2, 0.3, 0.5)]
    for g in grid:
        if g["bets"]:
            print(f"    edge>{g['min_edge']:+.2f} odds<={g['max_odds']:>3.0f}  bets {g['bets']:6d}  hit {g['hit_rate']:.3f}  "
                  f"ROI {g['roi']:+.3f}  90% CI [{g['roi_lo']:+.3f}, {g['roi_hi']:+.3f}]")
        else:
            print(f"    edge>{g['min_edge']:+.2f} odds<={g['max_odds']:>3.0f}  no bets")
    return t, grid


def main():
    df = load_features()
    train, blend_set, test = split(df)
    print(f"races  train {train.race_id.nunique()}  blend {blend_set.race_id.nunique()}  test {test.race_id.nunique()}")
    won, rid = test["won"].values, test["race_id"].values
    t = test.copy()

    # --- A. Benter's two-stage model, as in the 1994 paper ---------------
    fm = ConditionalLogit(l2=1e-4).fit(train[FEATURES].values, train["race_id"].values, train["won"].values)
    fb = fm.predict_proba(blend_set[FEATURES].values, blend_set["race_id"].values)
    comb = ConditionalLogit(l2=0.0, standardize=False).fit(
        np.column_stack([np.log(fb), blend_set["log_public_p"]]), blend_set["race_id"].values, blend_set["won"].values
    )
    alpha, beta = comb.coef_
    t["fund_p"] = fm.predict_proba(t[FEATURES].values, rid)
    t["benter_p"] = blend(t["fund_p"].values, t["public_p"].values, rid, alpha, beta)

    # --- B. Market-anchored model: fundamentals fit jointly with ln(public p)
    # Same functional form as A, but every coefficient now measures what the
    # public *misprices* rather than what predicts winners in general.
    mcols = ["log_public_p"] + FEATURES
    trb = pd.concat([train, blend_set])
    mm = ConditionalLogit(l2=1e-3).fit(trb[mcols].values, trb["race_id"].values, trb["won"].values)
    t["model_p"] = mm.predict_proba(t[mcols].values, rid)

    r2 = {k: pseudo_r2(t[c].values, won, rid) for k, c in
          [("public", "public_p"), ("fundamental", "fund_p"), ("benter_two_stage", "benter_p"), ("market_anchored", "model_p")]}
    print(f"A. Benter two-stage: alpha (fundamental) = {alpha:.3f}  beta (public) = {beta:.3f}")
    print("test pseudo-R2 ", {k: round(v, 4) for k, v in r2.items()})

    print("A. fundamental coefficients (per 1 sd):")
    for k, v in sorted(zip(FEATURES, fm.coef_), key=lambda kv: -abs(kv[1])):
        print(f"    {k:18s} {v:+.3f}")
    print("B. market-anchored coefficients (per 1 sd):")
    for k, v in sorted(zip(mcols, mm.coef_), key=lambda kv: -abs(kv[1])):
        print(f"    {k:18s} {v:+.3f}")

    print("A. Benter two-stage, flat $1 win bets on test seasons:")
    _, grid_a = evaluate(t, "benter_p")
    print("B. market-anchored, flat $1 win bets on test seasons:")
    t, grid_b = evaluate(t, "model_p")

    curve = kelly_backtest(t, 0.2, 30.0)
    print(f"B. quarter-Kelly (edge>0.20, odds<=30) bankroll over test: 1.00 -> {curve[-1][1]:.3f}")

    bins = [0, .02, .05, .1, .15, .2, .3, .4, .6, 1]
    model = dict(
        source="Kaggle gdaley/hkracing - HKJC Sha Tin & Happy Valley, 1997-06..2005-08",
        periods=dict(warmup=["1997-06-02", WARMUP_END], train=[WARMUP_END, TRAIN_END], blend=[TRAIN_END, BLEND_END],
                     test=[BLEND_END, str(df.date.max().date())]),
        races=dict(train=int(train.race_id.nunique()), blend=int(blend_set.race_id.nunique()), test=int(t.race_id.nunique())),
        features=FEATURES,
        prior={k: list(v) for k, v in PRIOR.items()},
        debut_log_odds=DEBUT_LOG_ODDS,
        fundamental=dict(mean=fm.mean_.tolist(), std=fm.std_.tolist(), coef=fm.coef_.tolist(), alpha=float(alpha), beta=float(beta)),
        market=dict(columns=mcols, mean=mm.mean_.tolist(), std=mm.std_.tolist(), coef=mm.coef_.tolist()),
        # used by the tracker to impute lengths-beaten when only positions are known
        lb_by_pos=[float(x) for x in train.assign(lb=train["lengths_behind"].clip(0, 30)).groupby(train["result"].clip(upper=14))["lb"].mean().reindex(range(1, 15)).ffill()],
        kelly=dict(fraction=KELLY_FRACTION, max_race_exposure=MAX_RACE_EXPOSURE),
        r2=r2,
        backtest=dict(benter_two_stage=grid_a, market_anchored=grid_b),
        kelly_curve=curve,
        calibration=dict(
            public=calibration(t["public_p"].values, won, bins),
            market_anchored=calibration(t["model_p"].values, won, bins),
        ),
    )
    OUT.write_text(json.dumps(model, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
