"""Fit the Benter model on HKJC history, backtest win and place betting,
and export everything the tracker and the live recommender use.

    python -m benter.run                      # Kaggle 1997-2005 data in data/
    python -m benter.run --data data/hkjc     # data scraped by hkjc.results

Splits are by date and never shuffled:
  warm-up  history only, so early races don't all look like debuts
  train    fit the fundamental model and the market-anchored model
  blend    fit Benter's alpha/beta and the place discount; choose betting rules
  test     untouched; every number reported for it uses rules fixed on blend
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .exotics import fit_discount, place_and_qpl
from .features import DATA, DEBUT_LOG_ODDS, FEATURES, PRIOR, load_features
from .model import ConditionalLogit, pseudo_r2, softmax_by_race

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tracker" / "model.json"

# Kaggle data: seasons run Sep..Jul.
KAGGLE_SPLITS = ("1998-09-01", "2002-09-01", "2003-09-01")

KELLY_FRACTION = 0.25
MAX_RACE_EXPOSURE = 0.10
MIN_RULE_BETS = 100   # a rule must place this many bets on blend to be chosen
EDGES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
WIN_CAPS = (10.0, 30.0, 99.0)
PLACE_CAPS = (3.0, 5.0, 10.0, 99.0)
# Quinella place cannot be backtested (no historical QPL odds); the live
# recommender uses this conservative rule until logged snapshots allow one.
QPL_RULE = dict(min_edge=0.25, max_odds=60.0, validated=False)

MCOLS = ["log_public_p"] + FEATURES


def split_dates(df):
    if df["date"].min() < pd.Timestamp("2000-01-01"):
        return KAGGLE_SPLITS
    q = df["date"].quantile([0.15, 0.65, 0.80])
    return tuple(str(x.date()) for x in q)


def split(df, dates=KAGGLE_SPLITS):
    w, t, b = dates
    d = df["date"]
    return df[(d >= w) & (d < t)], df[(d >= t) & (d < b)], df[d >= b]


def blend(f, pi, race_ids, alpha, beta):
    return softmax_by_race(alpha * np.log(f) + beta * np.log(pi), race_ids)


def bootstrap_roi(races, ret, n=500):
    """ROI and 90% interval, resampling whole races."""
    uniq, inv = np.unique(races, return_inverse=True)
    pr, pn = np.bincount(inv, ret), np.bincount(inv)
    rng = np.random.default_rng(0)
    boots = [pr[s].sum() / pn[s].sum() for s in (rng.integers(0, len(uniq), len(uniq)) for _ in range(n))]
    return float(ret.mean()), float(np.percentile(boots, 5)), float(np.percentile(boots, 95))


def grid(bets, edge_col, odds_col, ret_col, caps):
    """Flat $1 backtest for every (min edge, max odds) rule."""
    out = []
    for cap in caps:
        for e in (-1.0,) + EDGES:
            b = bets[(bets[edge_col] > e) & (bets[odds_col] <= cap)]
            row = dict(min_edge=e, max_odds=cap, bets=int(len(b)))
            if len(b):
                roi, lo, hi = bootstrap_roi(b["race_id"].values, b[ret_col].values)
                row.update(races=int(b.race_id.nunique()), hit_rate=float((b[ret_col] > -1).mean()),
                           avg_odds=float(b[odds_col].mean()), roi=roi, roi_lo=lo, roi_hi=hi)
            out.append(row)
    return out


def choose(blend_grid):
    ok = [g for g in blend_grid if g["min_edge"] >= 0 and g["bets"] >= MIN_RULE_BETS]
    best = max(ok, key=lambda g: g["roi"]) if ok else None
    return None if best is None else dict(min_edge=best["min_edge"], max_odds=best["max_odds"], blend_roi=best["roi"], validated=True)


def lookup(g, rule):
    return next(x for x in g if x["min_edge"] == rule["min_edge"] and x["max_odds"] == rule["max_odds"])


def place_frame(d, prob_col, g2, g3):
    """One row per runner with model place chance, place odds and the actual return."""
    rows = []
    for rid, r in d.groupby("race_id", sort=False):
        if len(r) < 7 or r["place_odds"].isna().any():
            continue
        place, _ = place_and_qpl(r[prob_col].values, g2, g3)
        for (_, x), pp in zip(r.iterrows(), place):
            div = 0.0
            for i in (1, 2, 3):
                if x["horse_no"] == x.get(f"place_combination{i}"):
                    div = x[f"place_dividend{i}"] / 10.0
            rows.append((rid, x["date"], pp, x["place_odds"], div - 1.0))
    t = pd.DataFrame(rows, columns=["race_id", "date", "place_p", "place_odds", "ret"])
    t["edge"] = t["place_p"] * t["place_odds"] - 1.0
    return t


def kelly_curve(t, edge_col, odds_col, ret_col, rule):
    bank, curve = 1.0, []
    for (date, rid), r in t.groupby(["date", "race_id"], sort=True):
        r = r[(r[edge_col] > rule["min_edge"]) & (r[odds_col] <= rule["max_odds"])]
        if len(r):
            f = KELLY_FRACTION * r[edge_col] / (r[odds_col] - 1.0)
            if f.sum() > MAX_RACE_EXPOSURE:
                f *= MAX_RACE_EXPOSURE / f.sum()
            bank += float((f * bank * r[ret_col]).sum())
        curve.append((str(date.date()), bank))
    return pd.DataFrame(curve, columns=["date", "bank"]).groupby("date").last().reset_index().values.tolist()


def calibration(p, y, bins):
    g = pd.DataFrame({"p": p, "y": y, "b": pd.cut(p, bins)}).groupby("b", observed=True)
    return [dict(lo=float(i.left), hi=float(i.right), n=int(len(x)), pred=float(x.p.mean()), actual=float(x.y.mean())) for i, x in g]


def show(name, g, rule):
    print(name)
    for x in g:
        if x["bets"] and (x["min_edge"] < 0 or x["min_edge"] in (0.0, 0.1, 0.2, 0.3)):
            mark = " <- rule" if rule and x["min_edge"] == rule["min_edge"] and x["max_odds"] == rule["max_odds"] else ""
            e = "all" if x["min_edge"] < 0 else f">{x['min_edge']:.2f}"
            print(f"    edge {e:6s} odds<={x['max_odds']:>3.0f}  bets {x['bets']:6d}  ROI {x['roi']:+.3f}  "
                  f"90% [{x['roi_lo']:+.3f}, {x['roi_hi']:+.3f}]{mark}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    df = load_features(args.data)
    dates = split_dates(df)
    train, bl, test = (x.copy() for x in split(df, dates))
    print(f"races  train {train.race_id.nunique()}  blend {bl.race_id.nunique()}  test {test.race_id.nunique()}")

    # A. Benter two-stage
    fm = ConditionalLogit(l2=1e-4).fit(train[FEATURES].values, train["race_id"].values, train["won"].values)
    for d in (bl, test):
        d["fund_p"] = fm.predict_proba(d[FEATURES].values, d["race_id"].values)
    comb = ConditionalLogit(l2=0.0, standardize=False).fit(
        np.column_stack([np.log(bl["fund_p"]), bl["log_public_p"]]), bl["race_id"].values, bl["won"].values)
    alpha, beta = comb.coef_
    test["benter_p"] = blend(test["fund_p"].values, test["public_p"].values, test["race_id"].values, alpha, beta)

    # B. Market-anchored: fundamentals fit jointly with ln(public chance), on train only
    mm = ConditionalLogit(l2=1e-3).fit(train[MCOLS].values, train["race_id"].values, train["won"].values)
    for d in (bl, test):
        d["model_p"] = mm.predict_proba(d[MCOLS].values, d["race_id"].values)
        d["edge"] = d["model_p"] * d["win_odds"] - 1.0
        d["ret"] = np.where(d["won"] == 1, d["win_odds"] - 1.0, -1.0)

    rid, won = test["race_id"].values, test["won"].values
    r2 = {k: pseudo_r2(test[c].values, won, rid) for k, c in
          [("public", "public_p"), ("fundamental", "fund_p"), ("benter_two_stage", "benter_p"), ("market_anchored", "model_p")]}
    print(f"Benter two-stage alpha {alpha:.3f} beta {beta:.3f};  test pseudo-R2", {k: round(v, 4) for k, v in r2.items()})

    # Discounted Harville for 2nd/3rd, fit on blend
    g2, g3 = fit_discount([(r["model_p"].values, r["result"].values) for _, r in bl.groupby("race_id")])
    print(f"place discount g2 {g2:.3f} g3 {g3:.3f}  (Benter 1994 reports ~0.81, ~0.65)")

    # WIN
    win_rule = choose(grid(bl, "edge", "win_odds", "ret", WIN_CAPS))
    win_test = grid(test, "edge", "win_odds", "ret", WIN_CAPS)
    show("WIN, test seasons (rule chosen on blend):", win_test, win_rule)

    # PLACE
    has_place = test["place_odds"].notna().mean() > 0.5
    place_rule, place_test, place_curve, place_cal = None, [], [], []
    if has_place:
        pb, pt = place_frame(bl, "model_p", g2, g3), place_frame(test, "model_p", g2, g3)
        place_rule = choose(grid(pb, "edge", "place_odds", "ret", PLACE_CAPS))
        place_test = grid(pt, "edge", "place_odds", "ret", PLACE_CAPS)
        show("PLACE, test seasons (rule chosen on blend):", place_test, place_rule)
        place_curve = kelly_curve(pt, "edge", "place_odds", "ret", place_rule) if place_rule else []
        place_cal = calibration(pt["place_p"].values, (pt["ret"] > -1).astype(float).values, [0, .1, .2, .3, .4, .5, .6, .7, .8, 1])
    else:
        print("PLACE: no place odds in this data; backtest skipped")

    win_curve = kelly_curve(test, "edge", "win_odds", "ret", win_rule) if win_rule else []
    for name, rule, g in (("win", win_rule, win_test), ("place", place_rule, place_test)):
        if rule:
            x = lookup(g, rule)
            print(f"{name} rule edge>{rule['min_edge']} odds<={rule['max_odds']:.0f}: test {x['bets']} bets ROI {x['roi']:+.3f} "
                  f"[{x['roi_lo']:+.3f}, {x['roi_hi']:+.3f}]")

    bins = [0, .02, .05, .1, .15, .2, .3, .4, .6, 1]
    model = dict(
        source=str(args.data),
        periods=dict(warmup=[str(df.date.min().date()), dates[0]], train=[dates[0], dates[1]], blend=[dates[1], dates[2]],
                     test=[dates[2], str(df.date.max().date())]),
        races=dict(train=int(train.race_id.nunique()), blend=int(bl.race_id.nunique()), test=int(test.race_id.nunique())),
        features=FEATURES,
        prior={k: list(v) for k, v in PRIOR.items()},
        debut_log_odds=DEBUT_LOG_ODDS,
        fundamental=dict(mean=fm.mean_.tolist(), std=fm.std_.tolist(), coef=fm.coef_.tolist(), alpha=float(alpha), beta=float(beta)),
        market=dict(columns=MCOLS, mean=mm.mean_.tolist(), std=mm.std_.tolist(), coef=mm.coef_.tolist()),
        exotics=dict(g2=g2, g3=g3),
        rules=dict(win=win_rule, place=place_rule, qpl=QPL_RULE),
        lb_by_pos=[float(x) for x in train.assign(lb=train["lengths_behind"].clip(0, 30))
                   .groupby(train["result"].clip(upper=14))["lb"].mean().reindex(range(1, 15)).ffill().fillna(0)],
        kelly=dict(fraction=KELLY_FRACTION, max_race_exposure=MAX_RACE_EXPOSURE),
        r2=r2,
        backtest=dict(win=win_test, place=place_test),
        kelly_curve=dict(win=win_curve, place=place_curve),
        calibration=dict(
            public=calibration(test["public_p"].values, won, bins),
            market_anchored=calibration(test["model_p"].values, won, bins),
            place=place_cal,
        ),
    )
    args.out.write_text(json.dumps(model, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
