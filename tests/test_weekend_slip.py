"""Checks for benter.weekend_slip.      python -m tests.test_weekend_slip"""
import numpy as np
import pandas as pd

from benter import weekend_slip as ws


def test_conditional_logit_matches_statsmodels():
    try:
        from statsmodels.discrete.conditional_models import ConditionalLogit as SMConditionalLogit
    except ImportError:
        print("  statsmodels not installed - skipped")
        return
    rng = np.random.default_rng(0)
    rows, b_true = [], np.array([0.8, -0.5, 0.3])
    for r in range(400):
        n = rng.integers(6, 14)
        X = rng.normal(size=(n, 3))
        u = X @ b_true + rng.gumbel(size=n)
        rows += [(r, *x, float(i == u.argmax())) for i, x in enumerate(X)]
    d = pd.DataFrame(rows, columns=["race", "a", "b", "c", "won"])
    ours = ws.ConditionalLogit(standardize=False).fit(d[["a", "b", "c"]].values, d["race"].values, d["won"].values)
    sm = SMConditionalLogit(d["won"].values, d[["a", "b", "c"]].values, groups=d["race"].values).fit(disp=0)
    assert np.allclose(ours.coef_, sm.params, atol=1e-4), (ours.coef_, sm.params)
    assert np.allclose(ours.se_, sm.bse, rtol=1e-3), (ours.se_, sm.bse)


def test_finishing_orders():
    c = np.array([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
    for g2, g3 in [(1.0, 1.0), (0.81, 0.65)]:
        orders, p = ws.finishing_orders(c, g2, g3)
        assert np.isclose(p.sum(), 1.0)
        win = np.bincount(orders[:, 0], p, len(c))
        assert np.allclose(win, c)  # discounting only touches 2nd / 3rd
        top3 = sum(np.bincount(orders[:, k], p, len(c)) for k in range(3))
        assert np.allclose(top3.sum(), 3.0)


def test_single_bet_kelly_is_edge_over_odds_minus_one():
    p, o = 0.30, 4.5
    payoff = np.array([[1.0], [0.0]])
    f = ws.kelly_fractions(np.array([p, 1 - p]), payoff, np.array([o]), cap=0.99)
    assert np.isclose(f[0], (p * o - 1) / (o - 1), atol=1e-5)


def test_kelly_units():
    outcome_p, payoff = np.array([0.3, 0.7]), np.array([[1.0], [0.0]])
    # f* = 0.35/3.5 = 10% -> optimal stake on a HKD 50 Kelly bank is 5: one unit of 10 is borderline-good
    assert ws.kelly_units(outcome_p, payoff, np.array([4.5]), 50, 10, 100)[0] == 10
    # f* = 0.05/3.5 = 1.4% -> HKD 0.7 optimum, a unit would cut growth
    assert ws.kelly_units(np.array([0.21, 0.79]), payoff, np.array([5.0]), 50, 10, 100)[0] == 0
    # negative edge -> never
    assert ws.kelly_units(np.array([0.1, 0.9]), payoff, np.array([5.0]), 50, 10, 100)[0] == 0


def test_end_to_end_slip_respects_budget():
    for seed in (1, 2, 3):
        cfg = ws.Config()
        cfg.gamma_second = cfg.gamma_third = 1.0
        h = ws.make_mock_history(n_meetings=120, seed=seed)
        odds = ws.mock_exotic_odds(h[h["finish_position"].isna()], cfg, seed)
        card, slip = ws.run(h, odds, cfg, verbose=False)
        sums = card.groupby("race_id")["model_p"].sum()
        assert np.allclose(sums, 1.0)
        assert slip["stake_hkd"].sum() <= cfg.total_bankroll
        assert (slip["stake_hkd"] % cfg.bet_unit == 0).all()
        assert (slip["edge"] >= cfg.min_edge).all()
        assert len(slip) <= cfg.max_bets


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
