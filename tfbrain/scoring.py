from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import json, pickle
from pathlib import Path

def ridge_fit(X, y, lam: float = 1.0):
    # Standardized ridge with bias term (no L2 on bias)
    X = np.asarray(X, float); y = np.asarray(y, float)
    mu, sigma = X.mean(0), X.std(0)
    sigma[sigma == 0] = 1.0
    Z = (X - mu) / sigma
    Zb = np.c_[Z, np.ones(len(Z))]
    I = np.eye(Zb.shape[1]); I[-1, -1] = 0.0
    W = np.linalg.solve(Zb.T @ Zb + lam * I, Zb.T @ y)
    return {"mu": mu, "sigma": sigma, "W": W}

def ridge_apply(model: Dict[str, Any], X):
    X = np.asarray(X, float)
    Z = (X - model["mu"]) / model["sigma"]
    Zb = np.c_[Z, np.ones(len(Z))]
    return Zb @ model["W"]

def isotonic_fit(x, y):
    # PAV for nondecreasing fit
    x = np.asarray(x); y = np.asarray(y)
    order = np.argsort(x); x, y = x[order], y[order]
    y_hat = y.astype(float).copy()
    n = len(y_hat); lvl = y_hat.copy(); w = np.ones(n)
    i = 0
    while i < n - 1:
        if lvl[i] <= lvl[i + 1]:
            i += 1; continue
        j = i
        while j >= 0 and lvl[j] > lvl[j + 1]:
            wsum = w[j] + w[j + 1]
            lvl[j] = lvl[j + 1] = (w[j] * lvl[j] + w[j + 1] * lvl[j + 1]) / wsum
            w[j] = w[j + 1] = wsum
            j -= 1
        i += 1
    # expand into a step function
    return {"x": x, "lvl": lvl}

def isotonic_apply(model: Dict[str, Any], xq):
    x, lvl = model["x"], model["lvl"]
    idx = np.searchsorted(x, xq, side="right") - 1
    idx = np.clip(idx, 0, len(lvl) - 1)
    return lvl[idx]

 


def pick_tau_by_target(scores: np.ndarray,
                       y: np.ndarray,
                       months: np.ndarray,
                       target_trades_per_month: float,
                       min_precision: float,
                       min_trades: int = 10) -> float:
    """
    Choose a decision threshold tau on TRAIN that aims for a target average trades/month
    while meeting a minimum precision. If constraints cannot be met, fall back to a
    quantile that yields at least `min_trades`.
    """
    if len(scores) == 0:
        return float("+inf")
    # group by month to estimate trades per month as we vary tau
    uniq_months = np.unique(months)
    if target_trades_per_month <= 0 or len(uniq_months) == 0:
        # no target → default fallback: 80th percentile of scores or last value
        k = max(min_trades, int(0.2 * len(scores)))
        k = min(max(1, k), len(scores))
        tau = float(np.sort(scores)[-k])
        return tau

    # sort descending by score
    order = np.argsort(scores)[::-1]
    scores_sorted = scores[order]
    y_sorted = y[order]
    m_sorted = months[order]

    # scan thresholds at each unique score
    best_tau = float("-inf")
    best_gap = float("+inf")
    for i in range(len(scores_sorted)):
        tau = scores_sorted[i]
        sel = scores >= tau
        if not np.any(sel):
            continue
        # trades per month at this tau
        trades_per_month = []
        precision_numer = 0
        precision_denom = 0
        for m in uniq_months:
            mask_m = sel & (months == m)
            n_m = int(mask_m.sum())
            if n_m > 0:
                trades_per_month.append(n_m)
                precision_numer += int((y[mask_m] > 0).sum())
                precision_denom += n_m
        avg_tpm = float(np.mean(trades_per_month)) if trades_per_month else 0.0
        precision = (precision_numer / precision_denom) if precision_denom > 0 else 0.0

        if precision_denom >= min_trades and precision >= min_precision:
            gap = abs(avg_tpm - target_trades_per_month)
            if gap < best_gap:
                best_gap = gap
                best_tau = float(tau)

    if best_tau != float("-inf"):
        return best_tau

    # Fallback: pick the smallest tau that yields at least min_trades overall
    k = max(min_trades, int(0.2 * len(scores)))
    k = min(max(1, k), len(scores))
    return float(np.sort(scores)[-k])
