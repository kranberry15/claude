"""Place and quinella-place probabilities from win probabilities.

Harville (1973) assumes the race for 2nd is a fresh race among the rest,
P(j 2nd | i 1st) = p_j / (1 - p_i).  Benter (1994) and Lo & Bacon-Shone
showed this overstates favourites for the minor places, and suggested
discounting: P(j 2nd | i 1st) ∝ p_j ** g2, P(k 3rd | i, j) ∝ p_k ** g3 with
g < 1.  We fit g2, g3 by maximum likelihood on observed finishing orders.

HKJC rules used here:
  * PLACE pays on 1st-3rd with 7+ starters, 1st-2nd with 4-6 starters.
  * QUINELLA PLACE (QPL) pays when both horses finish in the first three
    (any order); there are three winning pairs per race.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def order_probs(p: np.ndarray, g2: float, g3: float) -> np.ndarray:
    """P[i, j, k] = P(i 1st, j 2nd, k 3rd) for one race (n <= ~16)."""
    p = np.asarray(p, float)
    n = len(p)
    a2, a3 = p ** g2, p ** g3
    eye = np.eye(n, dtype=bool)
    # second given first: a2[j] / (sum a2 - a2[i])
    s2 = a2.sum() - a2                                   # [i]
    P2 = np.where(eye, 0.0, a2[None, :] / s2[:, None])   # [i, j]
    # third given first two: a3[k] / (sum a3 - a3[i] - a3[j])
    s3 = a3.sum() - a3[:, None] - a3[None, :]            # [i, j]
    with np.errstate(divide="ignore", invalid="ignore"):
        P3 = a3[None, None, :] / s3[:, :, None]          # [i, j, k]
    bad = eye[:, :, None] | eye[:, None, :] | eye[None, :, :]
    P3 = np.where(bad, 0.0, P3)
    return p[:, None, None] * P2[:, :, None] * P3


def place_and_qpl(p: np.ndarray, g2: float, g3: float):
    """Return (place_prob[n], qpl_prob[n, n] symmetric) for one race."""
    n = len(p)
    if n < 3:
        raise ValueError("need at least 3 runners")
    T = order_probs(p, g2, g3)
    first = T.sum((1, 2))
    second = T.sum((0, 2))
    third = T.sum((0, 1))
    places = 3 if n >= 7 else 2
    place = first + second + (third if places == 3 else 0.0)
    # pair (a, b) both in top three: any of the 6 orderings with a third horse
    Q = T.sum(2) + T.sum(1) + T.sum(0)                  # (a,b) in positions (12), (13), (23)
    qpl = Q + Q.T
    np.fill_diagonal(qpl, 0.0)
    return place, qpl


def fit_discount(races, init=(0.8, 0.7)):
    """races: iterable of (win_probs, finish_positions).  Returns (g2, g3)."""
    data = []
    for p, pos in races:
        pos = np.asarray(pos)
        if len(p) < 4 or not {1, 2, 3} <= set(pos.tolist()):
            continue
        i, j, k = (int(np.where(pos == r)[0][0]) for r in (1, 2, 3))
        data.append((np.clip(np.asarray(p, float), 1e-6, 1), i, j, k))

    def nll(g):
        g2, g3 = g
        ll = 0.0
        for p, i, j, k in data:
            a2, a3 = p ** g2, p ** g3
            ll += np.log(a2[j]) - np.log(a2.sum() - a2[i])
            ll += np.log(a3[k]) - np.log(a3.sum() - a3[i] - a3[j])
        return -ll

    res = minimize(nll, np.array(init), method="L-BFGS-B", bounds=[(0.3, 1.5), (0.3, 1.5)])
    return tuple(float(x) for x in res.x)
