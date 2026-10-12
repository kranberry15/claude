"""Benter-style end-to-end handicapping + wagering for a HKD 200 race weekend.

    python -m benter.weekend_slip                       # mock history, demo slip
    python -m benter.weekend_slip --history runs.csv --odds odds.csv

The script is self-contained (numpy / pandas / scipy only) so it can be
copied out of the repo as one file.  It follows the pipeline in Bill Benter,
"Computer Based Horse Race Handicapping and Wagering Systems: A Report"
(1994):

    1. Data & variables   point-in-time handicapping factors per runner
    2. Fundamental model  conditional (multinomial) logit, one choice set per race
    3. Combined model     blend the fundamental probability with the tote's
                          implied probability, fit on races stage 2 never saw
    4. Edge               edge = p * odds - 1 for every Win, Place, Quinella
    5. Wagering           joint Kelly over the race's finishing orders,
                          scaled to fractional Kelly, rounded to HKD 10 units,
                          capped at the weekend bankroll

Input schema (one row per runner; rows of the upcoming card have
finish_position / finish_time empty and win_odds = current tote odds):

    race_id, date, venue, race_no, distance, horse_id, horse_no, horse_name,
    jockey_id, trainer_id, rating, weight_carried, finish_position, finish_time,
    win_odds                       (rating = official HKJC rating)

Optional odds table for the upcoming card (Place / Quinella pools):

    race_id, bet_type (WIN|PLACE|QUINELLA), selection ("7" or "3-7"), odds

All odds are HKJC decimal odds (dividend per HKD 1, stake included), e.g. a
win dividend of HKD 46 on a HKD 10 unit is odds 4.6.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    total_bankroll: float = 200.0       # HKD for the whole race weekend
    kelly_fraction: float = 0.25        # quarter-Kelly
    bet_unit: float = 10.0              # HKJC minimum / increment for W, P, Q
    max_race_fraction: float = 0.50     # never put more than this share of the bankroll on one race
    min_edge: float = 0.05              # ignore bets whose edge is below this
    min_win_prob: float = 0.03          # ignore runners the model gives < 3% (longshot noise)
    max_odds: dict = field(default_factory=lambda: {"WIN": 40.0, "PLACE": 15.0, "QUINELLA": 120.0})
    max_candidates: int = 12            # bets per race passed to the Kelly optimiser
    max_bets: int = 6                   # lines on the final slip
    # HKJC deductions (takeout): Win, Place, Quinella are 17.5%.
    takeout: dict = field(default_factory=lambda: {"WIN": 0.175, "PLACE": 0.175, "QUINELLA": 0.175})
    # Benter's (1994) correction to Harville: 2nd and 3rd place probabilities
    # use p**gamma, because pure Harville over-rates favourites for the minor
    # placings.  Set both to 1.0 for plain Harville.
    gamma_second: float = 0.81
    gamma_third: float = 0.65
    # Shrinkage priors for win rates: (pseudo-rides) toward the population rate.
    k_jockey: float = 50.0
    k_trainer: float = 50.0
    k_pair: float = 20.0
    l2: float = 1e-3                    # ridge on the standardised coefficients


FEATURES = [
    "speed_rating",       # mean adjusted speed figure, last 3 starts
    "first_start",        # 1 = no previous run (speed_rating imputed)
    "weight_disadv",      # (lb above race-average weight) x (distance km)
    "rating_rel",         # official rating minus race average (controls for class)
    "log_days",           # ln(days since last run)
    "layoff",             # max(0, ln(days / 60)): rust after a long break
    "quick_backup",       # 1 = ran within the last 13 days: incomplete recovery
    "jockey_logit",       # shrunk jockey win rate, logit vs population
    "trainer_logit",      # shrunk trainer win rate, logit vs population
    "jt_interaction",     # jockey x trainer pair: logit(pair rate) - logit(expected pair rate)
]

# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------


def make_mock_history(n_meetings: int = 260, races_per_meeting: int = 10, seed: int = 7) -> pd.DataFrame:
    """Simulate HKJC-like results so the whole pipeline can run end to end.

    Each runner has a latent "true utility" (ability + jockey + trainer +
    pair synergy - weight burden - freshness penalties).  Finishing order is
    drawn by adding iid Gumbel noise and sorting, which is exactly the random
    utility model the conditional logit assumes, so a well-specified model can
    recover the truth.  The public's odds are a noisy, slightly shrunk view of
    the same utilities, so real (but small) overlays exist.

    The last meeting is the upcoming race card: results are blanked out.
    """
    rng = np.random.default_rng(seed)
    n_horses, n_jockeys, n_trainers = 700, 40, 30
    ability = rng.normal(0, 0.9, n_horses)
    rating = np.clip(np.round(60 + 18 * ability + rng.normal(0, 6, n_horses)), 20, 125)
    jockey_skill = rng.normal(0, 0.25, n_jockeys)
    trainer_skill = rng.normal(0, 0.20, n_trainers)
    pair_synergy = rng.normal(0, 0.25, (n_jockeys, n_trainers)) * (rng.random((n_jockeys, n_trainers)) < 0.15)
    horse_trainer = rng.integers(0, n_trainers, n_horses)
    claim = np.zeros(n_jockeys)
    claim[:8] = [10, 10, 7, 7, 5, 5, 2, 2]
    distances = np.array([1000, 1200, 1400, 1650, 1800, 2000, 2200])
    sec_per_m = {1000: 0.0562, 1200: 0.0578, 1400: 0.0587, 1650: 0.0598, 1800: 0.0603, 2000: 0.0609, 2200: 0.0614}

    last_run = np.full(n_horses, np.datetime64("NaT", "D"))
    date = np.datetime64("2023-09-03", "D")
    rows = []
    for m in range(n_meetings):
        date = date + np.timedelta64(3 if m % 2 == 0 else 4, "D")  # Sun / Wed rhythm
        venue = "ST" if m % 2 == 0 else "HV"
        variant = rng.normal(0, 0.5)  # sec per 1000m the whole card ran slow (+) / fast (-)
        used = set()
        for race_no in range(1, races_per_meeting + 1):
            dist = int(rng.choice(distances if venue == "ST" else distances[[0, 1, 3, 4, 6]]))
            n = int(rng.integers(8, 15))
            pool = np.setdiff1d(np.arange(n_horses), list(used))
            # horses that ran recently are less likely to be declared again
            days = (date - last_run[pool]).astype("timedelta64[D]").astype(float)
            days[np.isnat(last_run[pool])] = np.nan
            w = np.where(np.isnan(days), 0.3, np.clip(days / 21.0, 0.05, 1.5))
            horses = rng.choice(pool, n, replace=False, p=w / w.sum())
            used.update(horses)
            # HKJC handicap: top weight 133 lb, 1 lb per rating point, floor 113,
            # less the apprentice claim of jockeys 0-7 (10 / 7 / 5 / 2 lb)
            jk = rng.integers(0, n_jockeys, n)
            tr = horse_trainer[horses]
            wt = np.clip(133 - (rating[horses].max() - rating[horses]), 113, 133) - claim[jk]
            dsl = (date - last_run[horses]).astype("timedelta64[D]").astype(float)
            dsl[np.isnat(last_run[horses])] = np.nan
            fresh = np.where(np.isnan(dsl), -0.25, 0.0)
            dsl0 = np.nan_to_num(dsl, nan=60.0)
            fresh -= np.where(dsl0 < 14, 0.20, 0.0)
            fresh -= 0.35 * np.log(np.maximum(dsl0, 60.0) / 60.0)
            util = (ability[horses] + jockey_skill[jk] + trainer_skill[tr] + pair_synergy[jk, tr]
                    - 0.012 * (wt - wt.mean()) * dist / 1000 + fresh)
            perf = util + rng.gumbel(0, 1, n)
            order = np.argsort(-perf)
            pos = np.empty(n, int)
            pos[order] = np.arange(1, n + 1)
            # better performance -> faster time; ~0.35 s per 1000m per util unit
            t = dist * sec_per_m[dist] + (variant - 0.35 * perf) * dist / 1000 + rng.normal(0, 0.05, n)
            # the public: shrunk + noisy utilities, then the 17.5% take
            pub_u = 0.85 * util + rng.normal(0, 0.40, n)
            pub_p = np.exp(pub_u - pub_u.max())
            pub_p /= pub_p.sum()
            odds = np.maximum(1.0, np.round(0.825 / pub_p, 1))
            nos = rng.permutation(n) + 1
            for i in range(n):
                rows.append(dict(
                    race_id=f"{str(date)}-{venue}-R{race_no:02d}", date=str(date), venue=venue, race_no=race_no,
                    distance=dist, horse_id=int(horses[i]), horse_no=int(nos[i]), horse_name=f"HORSE {horses[i]:03d}",
                    jockey_id=int(jk[i]), trainer_id=int(tr[i]), rating=float(rating[horses[i]]),
                    weight_carried=float(wt[i]),
                    finish_position=int(pos[i]), finish_time=round(float(t[i]), 2), win_odds=float(odds[i]),
                ))
            last_run[horses] = date
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    card = df["date"] == df["date"].max()
    df.loc[card, ["finish_position", "finish_time"]] = np.nan
    return df


def mock_exotic_odds(card: pd.DataFrame, cfg: Config, seed: int = 0) -> pd.DataFrame:
    """Place and Quinella tote odds for the mock card.

    A pool's dividend on a selection is roughly (1 - takeout) / (share of the
    pool on it).  We take the public's win probabilities from the win odds and
    push them through plain Harville to get the public's place / quinella
    shares, plus some pool noise (the minor pools are thinner than Win).  With
    real data, replace this with the HKJC odds pages.
    """
    rng = np.random.default_rng(seed)
    out = []
    for rid, r in card.groupby("race_id"):
        q = (1 / r["win_odds"].values)
        q /= q.sum()
        nos = r["horse_no"].values
        orders, probs = finishing_orders(q, 1.0, 1.0)
        places = 3 if len(q) >= 7 else 2
        for i, no in enumerate(nos):
            p_place = probs[(orders[:, :places] == i).any(1)].sum()
            o = (1 - cfg.takeout["PLACE"]) / p_place * rng.lognormal(0, 0.08)
            out.append((rid, "PLACE", str(no), max(1.0, round(o, 1))))
        for i, j in itertools.combinations(range(len(nos)), 2):
            p_q = probs[np.isin(orders[:, 0], [i, j]) & np.isin(orders[:, 1], [i, j])].sum()
            a, b = sorted((nos[i], nos[j]))
            o = (1 - cfg.takeout["QUINELLA"]) / p_q * rng.lognormal(0, 0.08)
            out.append((rid, "QUINELLA", f"{a}-{b}", max(1.0, round(o, 1))))
    return pd.DataFrame(out, columns=["race_id", "bet_type", "selection", "odds"])


def load_history(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    need = {"race_id", "date", "venue", "race_no", "distance", "horse_id", "horse_no", "horse_name", "jockey_id",
            "trainer_id", "rating", "weight_carried", "finish_position", "finish_time", "win_odds"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df


# ---------------------------------------------------------------------------
# 1b. VARIABLES  (Benter used ~130; these are the four families asked for)
# ---------------------------------------------------------------------------


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _prior_counts(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Rides and wins for `keys` strictly before each race *day*.

    Counting by day (not by race) means a jockey's afternoon winners can't leak
    into the rates used for his morning rides on the same card.
    """
    done = df["won"].notna()
    daily = (df.assign(ride=done.astype(float), w=df["won"].fillna(0))
             .groupby(keys + ["date"])[["ride", "w"]].sum().sort_index())
    prior = daily.groupby(level=list(range(len(keys)))).cumsum() - daily
    return df[keys + ["date"]].merge(prior.reset_index(), on=keys + ["date"], how="left")[["ride", "w"]]


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Point-in-time handicapping factors.  A runner's features use only races
    run before that day, so the model never sees the result it predicts."""
    df = df.sort_values(["date", "race_no", "horse_no"]).reset_index(drop=True)
    df["won"] = np.where(df["finish_position"].isna(), np.nan, (df["finish_position"] == 1).astype(float))
    km = df["distance"] / 1000.0

    # --- Speed rating, adjusted for distance and track variant -------------
    # par: typical time at this venue & distance.  variant: how much slower
    # (+) or faster (-) than par the whole meeting ran, per 1000m - the going,
    # wind, rail position.  Removing it makes a 69.8 on a slow day comparable
    # with a 69.2 on a fast one.  Figures are per 1000m so 1000m and 2200m
    # runs share one scale: 1 point = 0.1 s per 1000m.
    par = df.groupby(["venue", "distance"])["finish_time"].transform("median")
    dev = (df["finish_time"] - par) / km
    variant = dev.groupby(df["date"].astype(str) + df["venue"]).transform("median")
    # cap at +-40 points (4 s per 1000m): eased / tailed-off runs aren't form
    df["speed_fig"] = (100.0 - 10.0 * (dev - variant)).clip(60, 140)

    g = df.groupby("horse_id", sort=False)
    df["speed_rating"] = g["speed_fig"].transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
    df["first_start"] = (g.cumcount() == 0).astype(float)
    # debutants get the population mean; the first_start coefficient learns
    # how much worse (or better) unraced horses actually run.
    df["speed_rating"] = df["speed_rating"].fillna(df["speed_rating"].mean())

    # --- Weight-carried disadvantage ---------------------------------------
    # Only weight relative to rivals matters (a race-constant shift cancels in
    # the logit), and a pound costs more over further: lb above average x km.
    df["weight_disadv"] = (df["weight_carried"] - df.groupby("race_id")["weight_carried"].transform("mean")) * km
    # In a handicap the best horse gets top weight, so on its own weight looks
    # *good* for a horse.  Holding the official rating fixed isolates the
    # burden itself (apprentice claims, the 113 / 133 lb limits).
    df["rating_rel"] = df["rating"] - df.groupby("race_id")["rating"].transform("mean")

    # --- Days since last run -------------------------------------------------
    # Freshness is U-shaped: too soon = not recovered, too long = rusty.
    days = g["date"].diff().dt.days
    days = days.fillna(days.median())
    df["log_days"] = np.log(days.clip(lower=1))
    df["layoff"] = np.log(np.maximum(days / 60.0, 1.0))
    df["quick_backup"] = (days < 14).astype(float)

    # --- Jockey / trainer win rates and their interaction --------------------
    # Raw strike rates are noisy for small samples, so each is shrunk toward
    # the population rate p0 (empirical Bayes / beta prior with k pseudo-rides).
    # The pair term asks: does this jockey ride this trainer's horses better
    # than their separate rates imply?  Its prior is the independence
    # expectation j * t / p0, so it is 0 until the pair has real evidence.
    p0 = float(df["won"].mean())
    j = _prior_counts(df, ["jockey_id"])
    t = _prior_counts(df, ["trainer_id"])
    jt = _prior_counts(df, ["jockey_id", "trainer_id"])
    j_rate = (j["w"] + p0 * cfg.k_jockey) / (j["ride"] + cfg.k_jockey)
    t_rate = (t["w"] + p0 * cfg.k_trainer) / (t["ride"] + cfg.k_trainer)
    expected_pair = (j_rate * t_rate / p0).clip(1e-4, 0.9)
    pair_rate = (jt["w"] + expected_pair * cfg.k_pair) / (jt["ride"] + cfg.k_pair)
    df["jockey_logit"] = _logit(j_rate.values) - _logit(p0)
    df["trainer_logit"] = _logit(t_rate.values) - _logit(p0)
    df["jt_interaction"] = _logit(pair_rate.values) - _logit(expected_pair.values)

    # public implied probability: 1/odds renormalised so the race sums to 1
    inv = 1.0 / df["win_odds"]
    df["public_p"] = inv / inv.groupby(df["race_id"]).transform("sum")
    return df


# ---------------------------------------------------------------------------
# 2. PROBABILITY MODEL - conditional (multinomial) logit
# ---------------------------------------------------------------------------
#
# Random-utility derivation (McFadden 1974).  Give runner i in race r a
# utility  U_i = b'x_i + e_i , where x_i are its factors and e_i is iid
# Gumbel noise (everything the factors miss on the day).  The runner with the
# highest U wins.  The Gumbel max property gives the closed form
#
#        P(i wins race r) = exp(b'x_i) / sum_{j in r} exp(b'x_j)        (1)
#
# so within every race the probabilities are positive and sum to exactly 1,
# whatever b is.  One coefficient vector b is shared by all runners: a pound
# of weight costs the same whether you wear saddle-cloth 1 or 14.
#
# Why not statsmodels MNLogit?  MNLogit fits one coefficient vector per
# *labelled* alternative (alternative 1, 2, ... K) with chooser-level
# covariates - right for "which of 4 products did the shopper buy", wrong for
# racing, where the alternatives are different horses every race and the
# covariates belong to the alternative.  The racing model is McFadden's
# *conditional* logit (statsmodels has it as
# statsmodels.discrete.conditional_models.ConditionalLogit); it is fit here
# directly with scipy so the likelihood is explicit.
#
# Log-likelihood over races (w_r = the winner of race r):
#
#     LL(b) = sum_r [ b'x_{w_r} - log sum_{j in r} exp(b'x_j) ]
#     grad  = sum_r [ x_{w_r} - sum_j P_j x_j ]             (winner minus field mean)
#     -Hess = sum_r [ sum_j P_j x_j x_j' - m_r m_r' ],  m_r = sum_j P_j x_j
#
# LL is concave, so L-BFGS finds the global maximum; the inverse of -Hess
# gives the standard errors.  Fit quality is Benter's pseudo-R^2:
# 1 - LL(model)/LL(every runner equally likely).


class ConditionalLogit:
    def __init__(self, l2: float = 0.0, standardize: bool = True):
        self.l2, self.standardize = l2, standardize

    @staticmethod
    def _groups(race_ids):
        _, idx = np.unique(np.asarray(race_ids), return_inverse=True)
        return idx, int(idx.max()) + 1

    def _z(self, X):
        return (np.asarray(X, float) - self.mean_) / self.std_

    def fit(self, X, race_ids, won):
        X = np.asarray(X, float)
        self.mean_ = X.mean(0) if self.standardize else np.zeros(X.shape[1])
        self.std_ = X.std(0) + 1e-12 if self.standardize else np.ones(X.shape[1])
        Z, won = self._z(X), np.asarray(won, float)
        idx, n_r = self._groups(race_ids)

        def nll(b):
            p, logp = race_softmax(Z @ b, idx, n_r, with_log=True)
            ll = (won * logp).sum()
            grad = Z.T @ (won - p * np.bincount(idx, won, n_r)[idx])
            return -ll / n_r + self.l2 * b @ b, -grad / n_r + 2 * self.l2 * b

        res = minimize(nll, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=5000, gtol=1e-9, ftol=1e-13))
        self.coef_ = res.x
        # observed information -> standard errors (per standardised unit)
        p = race_softmax(Z @ self.coef_, idx, n_r)
        m = np.zeros((n_r, Z.shape[1]))
        np.add.at(m, idx, p[:, None] * Z)
        info = (Z * p[:, None]).T @ Z - m.T @ m
        info += 2 * self.l2 * n_r * np.eye(len(self.coef_))  # the ridge term's curvature keeps it invertible
        self.se_ = np.sqrt(np.diag(np.linalg.inv(info)))
        self.loglik_ = -res.fun * n_r + self.l2 * n_r * self.coef_ @ self.coef_
        return self

    def predict_proba(self, X, race_ids):
        idx, n_r = self._groups(race_ids)
        return race_softmax(self._z(X) @ self.coef_, idx, n_r)


def race_softmax(u, idx, n_r, with_log=False):
    """Equation (1), computed per race with the max subtracted for stability."""
    mx = np.full(n_r, -np.inf)
    np.maximum.at(mx, idx, u)
    e = np.exp(u - mx[idx])
    s = np.bincount(idx, e, n_r)[idx]
    if with_log:
        return e / s, u - mx[idx] - np.log(s)
    return e / s


def pseudo_r2(p, won, race_ids):
    won = np.asarray(won) == 1
    ll = np.log(np.clip(p[won], 1e-12, None)).sum()
    ll0 = -np.log(pd.Series(race_ids).value_counts().values).sum()
    return 1.0 - ll / ll0


@dataclass
class FittedModel:
    fundamental: ConditionalLogit
    alpha: float
    beta: float
    metrics: dict = field(default_factory=dict)  # held-out fit, filled by fit_model

    def win_probabilities(self, df: pd.DataFrame) -> np.ndarray:
        """Stage 2 (Benter's combined model):

            c_i = exp(alpha ln f_i + beta ln pi_i) / sum_j exp(alpha ln f_j + beta ln pi_j)

        f = fundamental probability, pi = public implied probability.  The
        public's odds carry information the fundamental model lacks (inside
        knowledge, trackwork, vet reports), so the best estimate leans on both.
        Fitting alpha and beta on races the fundamental model did not see stops
        the in-sample overconfidence of f from inflating alpha.
        """
        rid = df["race_id"].values
        f = self.fundamental.predict_proba(df[FEATURES].values, rid)
        idx, n_r = ConditionalLogit._groups(rid)
        c = race_softmax(self.alpha * np.log(f) + self.beta * np.log(df["public_p"].values), idx, n_r)
        sums = np.bincount(idx, c, n_r)
        assert np.allclose(sums, 1.0, atol=1e-12), "win probabilities must sum to 1 in every race"
        return c


def fit_model(feat: pd.DataFrame, cfg: Config, verbose: bool = True) -> FittedModel:
    """Chronological split of finished races: warm-up (features only, every
    horse looks like a debutant) | fundamental fit | combined fit | test."""
    hist = feat[feat["won"].notna()]
    hist = hist[hist.groupby("race_id")["won"].transform("sum") == 1]  # drop dead heats / voids
    dates = np.sort(hist["date"].unique())
    cut = lambda q: dates[int(q * (len(dates) - 1))]
    warm, fit_end, blend_end = cut(0.15), cut(0.65), cut(0.85)
    train = hist[(hist["date"] >= warm) & (hist["date"] < fit_end)]
    blend = hist[(hist["date"] >= fit_end) & (hist["date"] < blend_end)]
    test = hist[hist["date"] >= blend_end]

    fm = ConditionalLogit(l2=cfg.l2).fit(train[FEATURES].values, train["race_id"].values, train["won"].values)
    f_blend = fm.predict_proba(blend[FEATURES].values, blend["race_id"].values)
    comb = ConditionalLogit(standardize=False).fit(
        np.column_stack([np.log(f_blend), np.log(blend["public_p"].values)]),
        blend["race_id"].values, blend["won"].values)
    model = FittedModel(fm, *comb.coef_)
    rid, won = test["race_id"].values, test["won"].values
    f_test = fm.predict_proba(test[FEATURES].values, rid)
    model.metrics = dict(
        races=dict(train=train.race_id.nunique(), blend=blend.race_id.nunique(), test=test.race_id.nunique()),
        r2_public=pseudo_r2(test["public_p"].values, won, rid), r2_fundamental=pseudo_r2(f_test, won, rid),
        r2_combined=pseudo_r2(model.win_probabilities(test), won, rid))

    if verbose:
        m = model.metrics
        print(f"races: train {m['races']['train']}  blend {m['races']['blend']}  test {m['races']['test']}")
        print("\nFundamental conditional logit (coef per 1 sd of the factor):")
        print(f"  {'factor':16s} {'coef':>8s} {'s.e.':>7s} {'z':>7s}")
        for name, b, se in zip(FEATURES, fm.coef_, fm.se_):
            print(f"  {name:16s} {b:+8.3f} {se:7.3f} {b / se:+7.1f}")
        print(f"\nCombined model: alpha (fundamental) {model.alpha:.3f}   beta (public) {model.beta:.3f}")
        print("Held-out pseudo-R^2:  public {r2_public:.4f}   fundamental {r2_fundamental:.4f}   "
              "combined {r2_combined:.4f}".format(**m))
    return model


# ---------------------------------------------------------------------------
# 3. ODDS COMPARISON & EDGE
# ---------------------------------------------------------------------------


def finishing_orders(c: np.ndarray, g2: float, g3: float):
    """All ordered top-3 finishes (i, j, k) with their probabilities.

    Discounted Harville (Benter 1994):
        P(i, j, k) = c_i * c_j^g2 / sum_{m != i} c_m^g2 * c_k^g3 / sum_{m != i,j} c_m^g3
    The table sums to 1 by construction and gives Win, Place and Quinella
    probabilities that are mutually consistent - the Kelly step needs that,
    because those bets pay off on overlapping outcomes.
    """
    n = len(c)
    I, J, K = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    keep = (I != J) & (I != K) & (J != K)
    I, J, K = I[keep], J[keep], K[keep]
    c2, c3 = c ** g2, c ** g3
    p = c[I] * c2[J] / (c2.sum() - c2[I]) * c3[K] / (c3.sum() - c3[I] - c3[J])
    return np.column_stack([I, J, K]), p / p.sum()


def race_bets(race: pd.DataFrame, c: np.ndarray, odds_tbl: pd.DataFrame | None, cfg: Config):
    """Every Win / Place / Quinella line in a race with model probability,
    tote odds and edge.  Also returns the finishing-order probabilities and
    each line's payoff indicator over them (outcomes x lines) for Kelly."""
    nos = race["horse_no"].values
    names = race["horse_name"].values
    n = len(nos)
    orders, probs = finishing_orders(c, cfg.gamma_second, cfg.gamma_third)
    places = 3 if n >= 7 else 2  # HKJC: 3 places with 7+ starters, 2 with 4-6
    lookup = {}
    if odds_tbl is not None:
        r = odds_tbl[odds_tbl["race_id"] == race["race_id"].iloc[0]]
        lookup = {(b, str(s)): o for b, s, o in zip(r["bet_type"], r["selection"], r["odds"])}

    lines, payoff = [], []
    for i in range(n):
        lines.append(("WIN", str(nos[i]), names[i], float(race["win_odds"].iloc[i])))
        payoff.append(orders[:, 0] == i)
        if ("PLACE", str(nos[i])) in lookup and n >= 4:
            lines.append(("PLACE", str(nos[i]), names[i], lookup[("PLACE", str(nos[i]))]))
            payoff.append((orders[:, :places] == i).any(1))
    for i, j in itertools.combinations(range(n), 2):
        a, b = sorted((nos[i], nos[j]))
        key = ("QUINELLA", f"{a}-{b}")
        if key in lookup:
            ni, nj = (names[i], names[j]) if nos[i] < nos[j] else (names[j], names[i])
            lines.append(("QUINELLA", key[1], f"{ni} / {nj}", lookup[key]))
            payoff.append(np.isin(orders[:, 0], [i, j]) & np.isin(orders[:, 1], [i, j]))

    bets = pd.DataFrame(lines, columns=["bet_type", "selection", "name", "odds"])
    W = np.column_stack(payoff).astype(float)
    bets["model_p"] = probs @ W
    bets["edge"] = compute_edge(bets["model_p"].values, bets["odds"].values)
    bets.insert(0, "race_id", race["race_id"].iloc[0])
    return bets, probs, W


def compute_edge(true_p, public_odds):
    """Edge = (true probability x public odds) - 1: expected profit per HKD 1.

    The tote's own implied probability is 1/odds, inflated by the 17.5% take,
    so a runner is an overlay only when the model's probability beats 1/odds,
    not merely beats the public's (renormalised) opinion.
    """
    return np.asarray(true_p) * np.asarray(public_odds) - 1.0


# 4. BET OPTIMISATION - fractional Kelly
# ---------------------------------------------------------------------------
#
# Kelly maximises the expected log of wealth after the race.  For one bet at
# decimal odds o with win probability p the optimum fraction is
#
#        f* = (p o - 1) / (o - 1) = edge / (o - 1).
#
# Several bets in one race are not independent: Win on 5, Place on 5 and
# Quinella 5-8 all pay on "5 wins, 8 second".  So they are optimised jointly
# over the race's finishing orders s (from the discounted Harville table):
#
#     W_s(f) = 1 + sum_b f_b R_sb,     R_sb = o_b - 1 if line b pays in s else -1
#     G(f)   = sum_s P(s) log W_s(f)                     (concave in f)
#     dG/df_b = sum_s P(s) R_sb / W_s(f)
#
# maximised subject to f_b >= 0 and sum_b f_b <= max_race_fraction.
#
# Fractional Kelly: stake kappa * f* with kappa = 0.25.  Growth at kappa*f* is
# about kappa(2 - kappa) of full Kelly's (~44% for a quarter) while the
# variance of the bankroll falls by kappa^2 (to ~6%).  It is also the main
# protection against the model being over-confident, which is why Benter bet
# fractions of Kelly.
#
# Whole units.  The tote takes HKD 10 units, and with HKD 200 a unit is 5% of
# the bankroll, so rounding kappa*f* to the nearest unit is too crude
# (three lines worth HKD 4 each round to nothing, though one unit on the best
# of them may be worth having).  Since kappa*f* is exactly full Kelly on a
# bankroll of kappa*B (HKD 50 here), the stakes are picked as whole units
# maximising
#
#     G_units(s) = sum_s P(s) log(kappa*B + sum_b s_b R_sb)
#
# greedily: add the unit that raises G_units most, stop when no unit raises
# it.  A unit is only bought if it improves fractional-Kelly growth, so a
# line needs a full-Kelly fraction of roughly 2 units / (kappa*B) ~ 10% before
# it is worth one HKD 10 unit.  Most weekends will offer few such lines, and
# then the correct slip is short or empty.


def kelly_fractions(outcome_p: np.ndarray, payoff: np.ndarray, odds: np.ndarray, cap: float) -> np.ndarray:
    """Continuous joint full-Kelly fractions f* (reported for reference)."""
    R = payoff * odds[None, :] - 1.0

    def neg_growth(f):
        w = 1.0 + R @ f
        return -(outcome_p @ np.log(w)), -(R.T @ (outcome_p / w))

    edge = outcome_p @ R
    f0 = np.clip(edge / np.maximum(odds - 1.0, 1e-9), 0, None) * 0.1  # feasible, inside the domain
    res = minimize(neg_growth, f0, jac=True, method="SLSQP", bounds=[(0.0, cap)] * len(odds),
                   constraints=[{"type": "ineq", "fun": lambda f: cap - f.sum(), "jac": lambda f: -np.ones_like(f)}],
                   options=dict(maxiter=500, ftol=1e-12))
    return np.clip(res.x, 0, None)


def kelly_units(outcome_p: np.ndarray, payoff: np.ndarray, odds: np.ndarray,
                kelly_bank: float, unit: float, max_stake: float) -> np.ndarray:
    """Greedy whole-unit stakes maximising E[log(kelly_bank + R s)]."""
    R = payoff * odds[None, :] - 1.0
    stake = np.zeros(len(odds))
    wealth = np.full(len(outcome_p), float(kelly_bank))
    growth = outcome_p @ np.log(wealth)
    while stake.sum() + unit <= max_stake + 1e-9:
        trial = wealth[:, None] + unit * R              # wealth if one more unit goes on each line
        g = np.full(len(odds), -np.inf)
        ok = (trial > 0).all(0)
        g[ok] = outcome_p @ np.log(trial[:, ok])
        b = int(np.argmax(g))
        if g[b] <= growth + 1e-12:
            break
        stake[b] += unit
        wealth, growth = trial[:, b], g[b]
    return stake


def optimise_race(bets: pd.DataFrame, outcome_p: np.ndarray, payoff: np.ndarray, cfg: Config) -> pd.DataFrame:
    """Kelly over the race's overlays.  Only positive-edge lines (above
    min_edge, inside sane odds, no tiny probabilities) are candidates; the rest
    would get f = 0 anyway, bar small hedging positions."""
    ok = (bets["edge"] >= cfg.min_edge) & (bets["odds"] <= bets["bet_type"].map(cfg.max_odds))
    win_p = bets.loc[bets["bet_type"] == "WIN"].set_index("selection")["model_p"]
    sel_ok = bets["selection"].map(lambda s: min(win_p.get(h, 0) for h in s.split("-")) >= cfg.min_win_prob)
    cand = bets[ok & sel_ok].nlargest(cfg.max_candidates, "edge")
    if cand.empty:
        return cand.assign(full_kelly_hkd=[], frac_kelly_hkd=[], stake_hkd=[])
    W, o = payoff[:, cand.index], cand["odds"].values
    B = cfg.total_bankroll
    f = kelly_fractions(outcome_p, W, o, cfg.max_race_fraction)
    units = kelly_units(outcome_p, W, o, cfg.kelly_fraction * B, cfg.bet_unit, cfg.max_race_fraction * B)
    return cand.assign(full_kelly_hkd=f * B, frac_kelly_hkd=cfg.kelly_fraction * f * B, stake_hkd=units)


def build_betting_slip(card_bets: list[pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """Combine the races into one slip for the weekend.

    Races are sized independently against the same weekend bankroll (their
    outcomes are independent and each stake is small).  If the lines add up to
    more than the bankroll, or more than max_bets, the lowest-edge lines lose
    units first, so the slip never exceeds HKD total_bankroll.
    """
    cols = ["race_id", "bet_type", "selection", "name", "model_p", "odds", "edge", "full_kelly_hkd",
            "frac_kelly_hkd", "stake_hkd", "expected_profit_hkd"]
    s = pd.concat([b for b in card_bets if len(b)] or [pd.DataFrame(columns=cols)], ignore_index=True)
    s = s[s["stake_hkd"] >= cfg.bet_unit].sort_values("edge", ascending=False).head(cfg.max_bets)
    while s["stake_hkd"].sum() > cfg.total_bankroll + 1e-9:
        k = s.index[-1]
        s.loc[k, "stake_hkd"] -= cfg.bet_unit
        s = s[s["stake_hkd"] >= cfg.bet_unit]
    s = s.assign(expected_profit_hkd=s["stake_hkd"] * s["edge"])
    return s.sort_values(["race_id", "bet_type", "selection"])[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def prepare(history: pd.DataFrame, cfg: Config, verbose: bool = True):
    """Build features and fit the model once; returns (model, upcoming card)."""
    feat = build_features(history, cfg)
    model = fit_model(feat, cfg, verbose)
    card = feat[feat["won"].isna()].copy()
    if card.empty:
        raise ValueError("no upcoming runners (rows with empty finish_position) in the history")
    return model, card


def apply_live_odds(card: pd.DataFrame, odds_tbl: pd.DataFrame | None) -> pd.DataFrame:
    """Overwrite the card's win odds with the odds table's WIN rows (e.g. from
    benter.live_odds) and drop runners the tote no longer lists (scratched).
    Races with no WIN rows in the table keep the card's odds."""
    if odds_tbl is None:
        return card
    win = odds_tbl[odds_tbl["bet_type"] == "WIN"]
    if win.empty:
        return card
    live = {(r, str(s)): float(o) for r, s, o in zip(win["race_id"], win["selection"], win["odds"])}
    covered = set(win["race_id"])
    key = list(zip(card["race_id"], card["horse_no"].astype(str)))
    in_live = np.array([k in live for k in key])
    keep = ~card["race_id"].isin(covered).values | in_live
    card = card[keep].copy()
    card["win_odds"] = [live.get(k, o) for k, o in zip(zip(card["race_id"], card["horse_no"].astype(str)), card["win_odds"])]
    inv = 1.0 / card["win_odds"]
    card["public_p"] = inv / inv.groupby(card["race_id"]).transform("sum")
    return card


def slip_for_card(model: FittedModel, card: pd.DataFrame, odds_tbl: pd.DataFrame | None, cfg: Config):
    """Score the card at the given odds and build the slip.  Cheap enough to
    re-run on every odds refresh."""
    card = apply_live_odds(card, odds_tbl)
    card["model_p"] = model.win_probabilities(card)
    per_race = []
    for _, race in card.groupby("race_id", sort=True):
        race = race.sort_values("horse_no")
        bets, outcome_p, payoff = race_bets(race, race["model_p"].values, odds_tbl, cfg)
        per_race.append(optimise_race(bets, outcome_p, payoff, cfg))
    return card, build_betting_slip(per_race, cfg)


def run(history: pd.DataFrame, odds_tbl: pd.DataFrame | None, cfg: Config, verbose: bool = True):
    model, card = prepare(history, cfg, verbose)
    return slip_for_card(model, card, odds_tbl, cfg)


def print_slip(slip: pd.DataFrame, cfg: Config):
    print("\n" + "=" * 96)
    print(f" BETTING SLIP   bankroll HKD {cfg.total_bankroll:.0f}   {cfg.kelly_fraction:g}-Kelly   unit HKD {cfg.bet_unit:.0f}")
    print("=" * 96)
    if slip.empty:
        print(" NO BET - no line clears the edge threshold with a stake of at least one unit.")
        print(" Keep the bankroll; passing is a winning decision when there is no overlay.")
        return
    print(f" {'Race':20s} {'Type':8s} {'Sel':5s} {'Horse(s)':20s} {'Model P':>7s} {'Odds':>5s} {'Edge':>7s}"
          f" {'Full-K':>6s} {'Frac-K':>6s} {'STAKE':>6s} {'E[profit]':>9s}")
    for _, r in slip.iterrows():
        print(f" {r.race_id:20s} {r.bet_type:8s} {r.selection:5s} {r['name'][:20]:20s} {r.model_p:7.3f} {r.odds:5.1f} "
              f"{r.edge:+7.1%} {r.full_kelly_hkd:6.1f} {r.frac_kelly_hkd:6.1f} {r.stake_hkd:6.0f} {r.expected_profit_hkd:+9.2f}")
    print("-" * 96)
    print(f" TOTAL STAKE HKD {slip.stake_hkd.sum():.0f} of {cfg.total_bankroll:.0f}"
          f"   expected profit HKD {slip.expected_profit_hkd.sum():+.2f}")
    print(" Full-K / Frac-K: the line's share of the race's continuous joint Kelly portfolio.  Stakes are whole")
    print(" units optimised jointly per race, so one unit can carry the Kelly weight of several correlated lines.")
    print(" Odds move until the off: re-run with final tote odds and skip any line whose edge has gone.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--history", help="runs CSV (see module docstring); omit to use mock data")
    ap.add_argument("--odds", help="PLACE / QUINELLA odds CSV for the upcoming card")
    ap.add_argument("--bankroll", type=float, default=200.0)
    ap.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction")
    ap.add_argument("--min-edge", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7, help="mock data seed")
    ap.add_argument("--out", help="write the slip to this CSV")
    a = ap.parse_args(argv)
    cfg = Config(total_bankroll=a.bankroll, kelly_fraction=a.kelly, min_edge=a.min_edge)

    if a.history:
        history = load_history(a.history)
        odds_tbl = pd.read_csv(a.odds, dtype={"selection": str}) if a.odds else None
    else:
        history = make_mock_history(seed=a.seed)
        odds_tbl = mock_exotic_odds(history[history["finish_position"].isna()], cfg, a.seed)
        # The mock world draws finishing orders by Gumbel sorting, which is
        # exactly plain Harville, so Benter's real-data discount doesn't apply.
        cfg.gamma_second = cfg.gamma_third = 1.0
    _, slip = run(history, odds_tbl, cfg)
    print_slip(slip, cfg)
    if a.out:
        slip.to_csv(a.out, index=False)
        print(f" wrote {a.out}")


if __name__ == "__main__":
    main()
