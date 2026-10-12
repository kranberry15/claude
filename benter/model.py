"""Benter (1994) two-stage model.

Stage 1 - fundamental model: a multinomial (conditional) logit over the
runners in each race, P_i = exp(b.x_i) / sum_j exp(b.x_j).

Stage 2 - combined model: the fundamental probability f_i is blended with
the public's implied probability pi_i (from the tote odds) in a second
conditional logit, c_i ∝ exp(alpha*ln f_i + beta*ln pi_i).  Fitting alpha
and beta on races the fundamental model did not see is what captures the
information in the odds that the fundamental model lacks.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class ConditionalLogit:
    def __init__(self, l2: float = 1e-3, standardize: bool = True):
        self.l2 = l2
        self.standardize = standardize

    @staticmethod
    def _race_index(race_ids):
        _, idx = np.unique(race_ids, return_inverse=True)
        return idx, idx.max() + 1

    def _prep(self, X):
        return (X - self.mean_) / self.std_

    def fit(self, X, race_ids, won):
        X = np.asarray(X, float)
        if self.standardize:
            self.mean_, self.std_ = X.mean(0), X.std(0) + 1e-12
        else:
            self.mean_, self.std_ = np.zeros(X.shape[1]), np.ones(X.shape[1])
        Z = self._prep(X)
        idx, n_r = self._race_index(race_ids)
        won = np.asarray(won, float)
        n = len(won)

        def nll(b):
            u = Z @ b
            m = np.full(n_r, -np.inf)
            np.maximum.at(m, idx, u)
            e = np.exp(u - m[idx])
            s = np.bincount(idx, e, n_r)
            logp = u - m[idx] - np.log(s[idx])
            p = e / s[idx]
            ll = (won * logp).sum()
            # gradient: sum over races of x_winner - E_p[x]
            grad = Z.T @ (won - p * np.bincount(idx, won, n_r)[idx])
            # per-race scaling keeps L-BFGS tolerances meaningful
            return (-ll / n_r + self.l2 * b @ b), (-grad / n_r + 2 * self.l2 * b)

        res = minimize(nll, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B", options=dict(maxiter=2000, gtol=1e-8, ftol=1e-12))
        self.coef_ = res.x
        return self

    def utility(self, X):
        return self._prep(np.asarray(X, float)) @ self.coef_

    def predict_proba(self, X, race_ids):
        return softmax_by_race(self.utility(X), race_ids)


def softmax_by_race(u, race_ids):
    idx, n_r = ConditionalLogit._race_index(race_ids)
    m = np.full(n_r, -np.inf)
    np.maximum.at(m, idx, u)
    e = np.exp(u - m[idx])
    return e / np.bincount(idx, e, n_r)[idx]


def log_likelihood(p, won):
    return float(np.log(np.clip(p[np.asarray(won) == 1], 1e-12, None)).sum())


def pseudo_r2(p, won, race_ids):
    """Benter's R^2: 1 - LL(model) / LL(each runner equally likely)."""
    _, counts = np.unique(race_ids, return_counts=True)
    ll0 = -np.log(counts).sum()
    return 1.0 - log_likelihood(p, won) / ll0
