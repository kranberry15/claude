"""Point-in-time runner features for the HKJC dataset.

Every feature for a runner is computed only from races run *before* that
race (ordered by date, then race number), so the model never sees the
result it is predicting.  Features are chosen so they can also be read off
a current HKJC race card / horse page, which is what the tracker asks for.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

# Shrinkage priors: a rate with few observations is pulled toward the
# population rate.  The tracker applies the same formula.
PRIOR = {
    "win": (0.08, 5.0),      # (prior rate, pseudo-runs) for horse career win %
    "top3": (0.24, 5.0),     # horse career top-3 %
    "jockey": (0.08, 50.0),  # jockey win %
    "trainer": (0.08, 50.0), # trainer win %
}

FEATURES = [
    "rating",          # official HKJC rating
    "weight",          # weight carried (lb)
    "draw_frac",       # (draw-1)/(field-1): 0 = rail
    "draw_frac_hv",    # draw_frac at Happy Valley (tight track, draw matters more)
    "draw_frac_sprint",  # draw_frac at <= 1200m
    "age",
    "first_start",     # 1 if no previous run in the data
    "log_days_off",    # log(1 + days since last run), 0 on debut
    "dw_change",       # change in declared body weight since last run (lb)
    "last_pos",        # finishing position last start (capped at 14)
    "avg_pos3",        # mean finishing position, last 3 starts
    "avg_lb3",         # mean lengths beaten, last 3 starts (capped at 30)
    "log_starts",      # log(1 + career starts)
    "win_rate",        # shrunk career win %
    "top3_rate",       # shrunk career top-3 %
    "jockey_rate",     # shrunk jockey win %
    "trainer_rate",    # shrunk trainer win %
    "won_last",        # 1 if won last start
    "last_log_odds",   # ln(final win odds) last start - the market's past opinion
    "avg_log_odds3",   # mean ln(final win odds), last 3 starts
]

DEBUT_LOG_ODDS = float(np.log(30.0))


def shrink(hits, n, kind):
    p0, k = PRIOR[kind]
    return (hits + p0 * k) / (n + k)


def load_raw(data_dir: Path = DATA) -> pd.DataFrame:
    races = pd.read_csv(data_dir / "races.csv", parse_dates=["date"])
    runs = pd.read_csv(data_dir / "runs.csv")
    df = runs.merge(
        races[["race_id", "date", "venue", "race_no", "distance", "race_class", "surface", "going"]],
        on="race_id",
    )
    # Drop malformed races: not exactly one winner, or implausible book.
    g = df.groupby("race_id")
    book = g["win_odds"].transform(lambda o: (1.0 / o).sum())
    ok = (g["won"].transform("sum") == 1) & book.between(1.1, 1.4) & (g["race_id"].transform("size") >= 5)
    df = df[ok].copy()
    df["field"] = df.groupby("race_id")["race_id"].transform("size")
    return df.sort_values(["date", "race_no", "horse_no"]).reset_index(drop=True)


def _entity_rate(df: pd.DataFrame, col: str, kind: str) -> pd.Series:
    """Win rate of jockey/trainer from strictly earlier races.

    Aggregated per race first so that a trainer with two runners in the
    same race does not see either one's result.
    """
    per = (
        df.groupby([col, "date", "race_no"], sort=False)["won"]
        .agg(["sum", "size"])
        .reset_index()
        .sort_values(["date", "race_no"])
    )
    g = per.groupby(col)
    per["prior_w"] = g["sum"].cumsum() - per["sum"]
    per["prior_n"] = g["size"].cumsum() - per["size"]
    m = df[[col, "date", "race_no"]].merge(per, on=[col, "date", "race_no"], how="left")
    return pd.Series(shrink(m["prior_w"].values, m["prior_n"].values, kind), index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pos"] = df["result"].clip(upper=14)
    df["lb"] = df["lengths_behind"].clip(lower=0, upper=30)
    df["top3"] = (df["result"] <= 3).astype(float)

    h = df.groupby("horse_id", sort=False)
    prev_date = h["date"].shift(1)
    df["first_start"] = prev_date.isna().astype(float)
    days = (df["date"] - prev_date).dt.days
    df["log_days_off"] = np.log1p(days).fillna(0.0)
    df["dw_change"] = (df["declared_weight"] - h["declared_weight"].shift(1)).fillna(0.0)

    df["log_odds"] = np.log(df["win_odds"])
    df["won_last"] = (h["result"].shift(1) == 1).astype(float)
    df["last_log_odds"] = h["log_odds"].shift(1).fillna(DEBUT_LOG_ODDS)
    df["avg_log_odds3"] = h["log_odds"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(DEBUT_LOG_ODDS)
    df["last_pos"] = h["pos"].shift(1)
    df["avg_pos3"] = h["pos"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["avg_lb3"] = h["lb"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    # Debutants: impute a typical mid-field run (flagged by first_start).
    df["last_pos"] = df["last_pos"].fillna(7.0)
    df["avg_pos3"] = df["avg_pos3"].fillna(7.0)
    df["avg_lb3"] = df["avg_lb3"].fillna(6.0)

    starts = h.cumcount()
    wins = h["won"].cumsum() - df["won"]
    top3 = h["top3"].cumsum() - df["top3"]
    df["log_starts"] = np.log1p(starts)
    df["win_rate"] = shrink(wins, starts, "win")
    df["top3_rate"] = shrink(top3, starts, "top3")

    df["jockey_rate"] = _entity_rate(df, "jockey_id", "jockey")
    df["trainer_rate"] = _entity_rate(df, "trainer_id", "trainer")

    df["rating"] = df["horse_rating"].astype(float)
    df["weight"] = df["actual_weight"].astype(float)
    df["draw_frac"] = ((df["draw"] - 1) / (df["field"] - 1)).clip(0, 1)
    df["draw_frac_hv"] = df["draw_frac"] * (df["venue"] == "HV")
    df["draw_frac_sprint"] = df["draw_frac"] * (df["distance"] <= 1200)
    df["age"] = df["horse_age"].astype(float)

    # Public's estimate: normalised implied probability from final win odds.
    inv = 1.0 / df["win_odds"]
    df["public_p"] = inv / inv.groupby(df["race_id"]).transform("sum")
    df["log_public_p"] = np.log(df["public_p"])
    return df


def load_features(data_dir: Path = DATA) -> pd.DataFrame:
    return build_features(load_raw(data_dir))
