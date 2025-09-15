from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

@dataclass
class Shapelet:
    series: np.ndarray   # (C,T)
    epsilon: float
    meta: Dict[str, Any]

@dataclass
class Bank:
    kind: str
    shapelets: List[Shapelet]
    meta: Dict[str, Any]

def z_norm(x: np.ndarray):
    mu = x.mean(axis=1, keepdims=True); sd = x.std(axis=1, keepdims=True) + 1e-12
    return (x - mu)/sd

def mv_distance(a: np.ndarray, b: np.ndarray):
    A = z_norm(a); B = z_norm(b)
    return float(np.sqrt(((A-B)**2).sum()))

def extract_event_windows(features: pd.DataFrame, events: pd.DataFrame, channels: List[str], pre: int, post: int) -> Tuple[List[np.ndarray], List[int], List[int]]:
    seqs = []; labels = []; idxs = []
    arrs = [features[ch].values for ch in channels]
    X = np.vstack(arrs)  # C x N
    for _, e in events.iterrows():
        t0 = int(e["bar_idx"])
        start = t0 - pre; end = t0 + post
        if start < 0 or end >= X.shape[1]: continue
        seg = X[:, start:end+1].copy()
        if np.any(~np.isfinite(seg)): continue
        seqs.append(seg); labels.append(int(e["label"])); idxs.append(t0)
    return seqs, labels, idxs

def select_shapelets(seqs: List[np.ndarray], topk: int) -> List[np.ndarray]:
    if len(seqs)==0: return []
    n = len(seqs)
    D = np.zeros((n,n), dtype=float)
    for i in tqdm(range(n), desc="Mining: distance matrix", leave=False):
        for j in range(i+1, n):
            d = mv_distance(seqs[i], seqs[j])
            D[i,j]=d; D[j,i]=d
    selected=[]; sums = D.sum(axis=1)
    idx = int(np.argmin(sums)); selected.append(idx)
    with tqdm(total=min(topk,n)-1, desc="Mining: select shapelets", leave=False) as pb:
        while len(selected) < min(topk,n):
            min_to_sel = np.min(D[:, selected], axis=1)
            min_to_sel[selected] = -1
            next_idx = int(np.argmax(min_to_sel))
            if next_idx in selected: break
            selected.append(next_idx); pb.update(1)
    return [seqs[i] for i in selected]

def calibrate_epsilon(shapelet: np.ndarray, seqs_in_class: List[np.ndarray], q: float) -> float:
    dists = [mv_distance(shapelet, s) for s in seqs_in_class]
    if len(dists)==0: return float("inf")
    return float(np.quantile(dists, q))

def ppv(seqs_g: List[np.ndarray], seqs_b: List[np.ndarray], sh: np.ndarray, eps: float) -> float:
    hits_g = sum(1 for s in seqs_g if mv_distance(sh,s) <= eps)
    hits_b = sum(1 for s in seqs_b if mv_distance(sh,s) <= eps)
    denom = hits_g + hits_b
    return (hits_g/denom) if denom>0 else 0.0

def build_banks_per_regime(features: pd.DataFrame,
                           events: pd.DataFrame,
                           channels: List[str],
                           lengths: List[Dict[str,int]],
                           topk_per_class: int,
                           eps_percentile: float,
                           regime_ids,
                           ppv_prune: float,
                           meta: Dict[str,Any],
                           max_events_per_fold: Optional[int] = None):
    # Safety cap for event counts to avoid runaway compute
    events_n = len(events)
    max_events = int(max_events_per_fold) if max_events_per_fold is not None else 25000
    if events_n > max_events and events_n > 0:
        try:
            from math import floor
            f = max_events / float(events_n)
            grp = events.groupby(["label", "month"], dropna=False)
            counts = grp.size()
            target = (counts * f).apply(lambda x: int(floor(x)))
            # Ensure we don't drop all from any non-empty group
            target = target.where(counts == 0, target.clip(lower=1))
            # Adjust to exact budget by allocating leftovers by largest fractional part
            frac = (counts * f) - target
            need = max_events - int(target.sum())
            if need > 0:
                order = frac.sort_values(ascending=False)
                for idx in order.index[:need]:
                    target[idx] = int(target[idx]) + 1
            elif need < 0:
                # Too many selected due to min-1 constraint; remove from groups with smallest fractional part
                order = frac.sort_values(ascending=True)
                for idx in order.index:
                    if need == 0:
                        break
                    if target[idx] > 0:
                        target[idx] = int(target[idx]) - 1
                        need += 1

            parts = []
            for key, tcnt in target.items():
                if tcnt <= 0:
                    continue
                gdf = grp.get_group(key)
                if len(gdf) <= tcnt:
                    parts.append(gdf)
                else:
                    parts.append(gdf.sample(n=int(tcnt), random_state=42, replace=False))
            events = pd.concat(parts, ignore_index=True) if parts else events.head(0)
            tqdm.write(f"[mining] Capping events from {events_n} → {max_events} for compute safety (configurable).")
        except Exception:
            # Fallback: simple uniform sample if stratification failed
            events = events.sample(n=max_events, random_state=42, replace=False)
            tqdm.write(f"[mining] Capping events from {events_n} → {max_events} (uniform fallback).")

    banks = {}
    for reg in sorted(pd.unique(regime_ids)):
        ev_reg = events[events["regime_id"]==reg]
        reg_key = f"reg{int(reg)}"
        banks[reg_key] = {}
        for L in lengths:
            pre, post = int(L["pre"]), int(L["post"])
            seqs, labels, idxs = extract_event_windows(features, ev_reg, channels, pre, post)
            seqs_good = [s for s,l in zip(seqs, labels) if l==1]
            seqs_bad  = [s for s,l in zip(seqs, labels) if l!=1]
            goods = select_shapelets(seqs_good, topk_per_class)
            bads  = select_shapelets(seqs_bad,  topk_per_class)
            good_shapelets=[]; bad_shapelets=[]
            for sh in goods:
                eps = calibrate_epsilon(sh, seqs_good, eps_percentile/100.0)
                if ppv(seqs_good, seqs_bad, sh, eps) < ppv_prune: 
                    continue
                good_shapelets.append(Shapelet(series=sh, epsilon=eps, meta={"pre":pre,"post":post,"channels":channels,"regime":int(reg)}))
            for sh in bads:
                eps = calibrate_epsilon(sh, seqs_bad, eps_percentile/100.0)
                bad_shapelets.append(Shapelet(series=sh, epsilon=eps, meta={"pre":pre,"post":post,"channels":channels,"regime":int(reg)}))
            banks[reg_key][f"L{pre}_{post}"] = {
                "GOOD": Bank(kind="GOOD", shapelets=good_shapelets, meta=meta),
                "BAD":  Bank(kind="BAD",  shapelets=bad_shapelets,  meta=meta)
            }
    return banks
