from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

@dataclass
class Weights:
    w: np.ndarray  # includes bias as last element
    tau: float     # decision threshold

def mv_distance(a: np.ndarray, b: np.ndarray):
    mu_a = a.mean(axis=1, keepdims=True); sd_a = a.std(axis=1, keepdims=True) + 1e-12
    mu_b = b.mean(axis=1, keepdims=True); sd_b = b.std(axis=1, keepdims=True) + 1e-12
    A = (a-mu_a)/sd_a; B = (b-mu_b)/sd_b
    return float(np.sqrt(((A-B)**2).sum()))

def hits_from_banks(window_seq: np.ndarray, banks_for_length: Dict[str, Any]):
    good_hits = 0; bad_hits = 0; disc_hits = 0
    for kind, bank in banks_for_length.items():
        if kind not in ["GOOD", "BAD", "DISCORD"]: 
            continue
        for sh in bank.shapelets:
            d = mv_distance(window_seq, sh.series)
            if d <= sh.epsilon:
                if kind == "GOOD": good_hits += 1
                elif kind == "BAD": bad_hits += 1
                else: disc_hits += 1
    return good_hits, bad_hits, disc_hits

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float=1.0) -> np.ndarray:
    # closed-form ridge: (X'X + lam I)^-1 X'y
    n, d = X.shape
    I = np.eye(d)
    A = X.T @ X + lam * I
    b = X.T @ y
    w = np.linalg.solve(A, b)
    return w

def pick_tau(scores: np.ndarray, R: np.ndarray, min_trades:int=5):
    # grid over score quantiles to maximize expected R with basic support
    qs = np.linspace(0.1, 0.9, 17)
    best_tau = np.percentile(scores, 50)
    best_er = -1e9
    for q in qs:
        tau = np.percentile(scores, q*100)
        mask = scores >= tau
        if mask.sum() < min_trades:
            continue
        er = R[mask].mean()
        if er > best_er:
            best_er = er; best_tau = tau
    return best_tau

def build_train_weights(train_df: pd.DataFrame, lam: float=1.0) -> Weights:
    # Features for weights: good_hits, -bad_hits, -disc_hits, accel, kappa_long, donch_pos, bias
    cols = ["good_hits","neg_bad_hits","neg_disc_hits","accel","kappa_long","donch_pos"]
    X = train_df[cols].fillna(0).values
    X = np.c_[X, np.ones(len(X))]  # bias
    y = train_df["realized_R"].values  # regress R directly
    w = ridge_fit(X, y, lam=lam)
    scores = X @ w
    tau = pick_tau(scores, y, min_trades=max(5, int(0.2*len(y))))
    return Weights(w=w, tau=tau)

def score_row(row: pd.Series, w: np.ndarray) -> float:
    x = np.array([
        row.get("good_hits",0.0),
        row.get("neg_bad_hits",0.0),
        row.get("neg_disc_hits",0.0),
        row.get("accel",0.0),
        row.get("kappa_long",0.0),
        row.get("donch_pos",0.0),
        1.0
    ], dtype=float)
    return float(x @ w)
