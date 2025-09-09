from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

@dataclass
class Shapelet:
    series: np.ndarray  # shape (C, T)
    epsilon: float
    meta: Dict[str, Any]

@dataclass
class Bank:
    kind: str  # "GOOD" | "BAD" | "DISCORD"
    shapelets: List[Shapelet]
    meta: Dict[str, Any]

def z_norm(x: np.ndarray):
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-12
    return (x - mu) / sd

def mv_distance(a: np.ndarray, b: np.ndarray):
    # multivariate euclidean (channels x time) on z-normalized sequences
    A = z_norm(a); B = z_norm(b)
    return float(np.sqrt(((A-B)**2).sum()))

def extract_event_windows(features: pd.DataFrame, events_df: pd.DataFrame, channels: List[str], pre: int, post: int) -> Tuple[List[np.ndarray], List[int]]:
    seqs = []; labels = []
    arrs = [features[ch].values for ch in channels]
    X = np.vstack([a for a in arrs])  # C x N
    for _, e in events_df.iterrows():
        t0 = int(e["bar_idx"])
        start = t0 - pre
        end = t0 + post
        if start < 0 or end >= X.shape[1]: 
            continue
        seg = X[:, start:end+1].copy()
        if np.any(~np.isfinite(seg)):
            continue
        seqs.append(seg)
        labels.append(int(e["label"]))
    return seqs, labels

def select_shapelets(seqs: List[np.ndarray], topk: int) -> List[np.ndarray]:
    """Greedy medoid selection with progress: build distance matrix then farthest-first coverage."""
    if len(seqs) == 0:
        return []
    n = len(seqs)
    D = np.zeros((n, n), dtype=float)
    for i in tqdm(range(n), desc="Mining: distance matrix", leave=False):
        for j in range(i + 1, n):
            d = mv_distance(seqs[i], seqs[j])
            D[i, j] = d; D[j, i] = d
    selected = []
    sums = D.sum(axis=1)
    idx = int(np.argmin(sums))
    selected.append(idx)
    with tqdm(total=min(topk, n)-1, desc="Mining: select shapelets", leave=False) as pb:
        while len(selected) < min(topk, n):
            min_to_sel = np.min(D[:, selected], axis=1)
            min_to_sel[selected] = -1
            next_idx = int(np.argmax(min_to_sel))
            if next_idx in selected:
                break
            selected.append(next_idx)
            pb.update(1)
    return [seqs[i] for i in selected]

def calibrate_epsilon(shapelet: np.ndarray, seqs_in_class: List[np.ndarray], q: float=0.15) -> float:
    dists = [mv_distance(shapelet, s) for s in seqs_in_class]
    if len(dists)==0:
        return float("inf")
    eps = float(np.quantile(dists, q))
    return eps

def build_banks(features: pd.DataFrame, events_df: pd.DataFrame, channels: List[str], lengths: List[Dict[str,int]], topk_per_class:int, eps_q: float, meta: Dict[str,Any]) -> Dict[str, Bank]:
    banks = {}
    for L in lengths:
        pre, post = int(L["pre"]), int(L["post"])
        seqs, labels = extract_event_windows(features, events_df, channels, pre, post)
        seqs_good = [s for s,l in zip(seqs, labels) if l==1]
        seqs_bad  = [s for s,l in zip(seqs, labels) if l==-1]
        # GOOD bank
        goods = select_shapelets(seqs_good, topk_per_class)
        good_shapelets = []
        for sh in goods:
            eps = calibrate_epsilon(sh, seqs_good, eps_q/100.0)
            good_shapelets.append(Shapelet(series=sh, epsilon=eps, meta={"pre":pre,"post":post,"channels":channels}))
        # BAD bank
        bads = select_shapelets(seqs_bad, topk_per_class)
        bad_shapelets = []
        for sh in bads:
            eps = calibrate_epsilon(sh, seqs_bad, eps_q/100.0)
            bad_shapelets.append(Shapelet(series=sh, epsilon=eps, meta={"pre":pre,"post":post,"channels":channels}))
        # DISCORD bank (optional simple: pick farthest from GOOD medoids among 0-labeled or leftover)
        disc_shapelets = []  # keep empty for simplicity; can be added similarly
        banks_key = f"L{pre}_{post}"
        banks[banks_key] = {
            "GOOD": Bank(kind="GOOD", shapelets=good_shapelets, meta=meta),
            "BAD":  Bank(kind="BAD",  shapelets=bad_shapelets,  meta=meta),
            "DISCORD": Bank(kind="DISCORD", shapelets=disc_shapelets, meta=meta),
        }
    return banks
