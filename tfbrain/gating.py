import numpy as np
import pandas as pd
from typing import Dict, Any
from tqdm import tqdm
from .mining import mv_distance
from .utils import drawdown_R

def _seq(features, channels, start, end):
    arrs = [features[ch].values for ch in channels]
    X = np.vstack(arrs)
    if start < 0 or end >= X.shape[1]: return None
    seg = X[:, start:end+1]
    if np.any(~np.isfinite(seg)): return None
    return seg

def compute_min_dists(features: pd.DataFrame, banks: Dict[str,Any], bar_idx: int, regime_id: int, channels, eps_mult: float):
    good_min = np.inf; bad_min = np.inf; good_hits=0; bad_hits=0
    for reg_key, lengths in banks.items():
        if reg_key != f"reg{int(regime_id)}": 
            continue
        for _, bundle in lengths.items():
            for kind, bank in bundle.items():
                for sh in bank.shapelets:
                    pre = int(sh.meta["pre"]); post = int(sh.meta["post"])
                    seg = _seq(features, channels, bar_idx - pre, bar_idx + post)
                    if seg is None: continue
                    d = mv_distance(seg, sh.series)
                    if kind=="GOOD":
                        good_min = min(good_min, d)
                        if d <= sh.epsilon * eps_mult: good_hits += 1
                    else:
                        bad_min = min(bad_min, d)
                        if d <= sh.epsilon * eps_mult: bad_hits += 1
        # we've processed the matching regime; no need to scan others
        break
    return good_min, bad_min, good_hits, bad_hits

def train_gates_on_fold(features: pd.DataFrame,
                        train_df: pd.DataFrame,
                        banks: Dict[str, Any],
                        regimes: pd.Series,
                        channels: list,
                        cfg: Dict[str, Any]) -> Dict[str, Any]:
    gs = cfg["gating_search"]
    eps_list = gs["eps_multiplier"]
    K_list   = gs["K_hits"]
    L_list   = gs.get("lookback_L", [0])
    # NOTE: lookback_L is currently a compatibility placeholder (no rolling K-of-N aggregation yet).
    # Gating uses single-bar motif hits. Keep config key for forward-compatibility.
    M_list   = gs["bad_margin"]
    Bbuf     = gs["breakout_buffer_atr"]
    lam      = gs.get("ridge_lambda", 1.0)

    # Recompute GOOD/BAD mins and hits for a specific eps multiplier
    def recompute_min_hits(df_in: pd.DataFrame, epsm: float) -> pd.DataFrame:
        df = df_in.copy()
        # Vectorized apply over rows; returns 4 columns per row
        vals = df.apply(
            lambda r: compute_min_dists(
                features=features,
                banks=banks,
                bar_idx=int(r["bar_idx"]),
                regime_id=int(r["regime_id"]),
                channels=channels,
                eps_mult=float(epsm)
            ),
            axis=1,
            result_type="expand"
        )
        vals.columns = ["good_min", "bad_min", "good_hits", "bad_hits"]
        for c in vals.columns:
            df[c] = vals[c].astype(float)
        df["LLR_sum"] = df["bad_min"] - df["good_min"]
        return df

    def ridge_fit(X,y,lam=1.0):
        X = np.asarray(X); y=np.asarray(y)
        X = np.c_[X, np.ones(len(X))]
        A = X.T@X + lam*np.eye(X.shape[1])
        b = X.T@y
        w = np.linalg.solve(A,b)
        return w

    best=None
    for epsm in eps_list:
        # Recompute per-eps multiplier so the sweep actually changes the gating
        df_base = recompute_min_hits(train_df, epsm)
        for K in K_list:
            # NOTE: lookback_L kept for config compatibility; not used in v2 gating.
            # Future: aggregate motif hits over last L bars if needed.
            for L in L_list:  # (kept for compatibility)
                for dm in M_list:
                    for bb in Bbuf:
                        df = df_base.copy()
                        df["fire_rule"] = (
                            (df["good_hits"] >= K) &
                            (df["bad_hits"] == 0) &
                            ((df["bad_min"] - df["good_min"]) >= dm) &
                            (df["breakout_ok_"+str(bb)].astype(bool))
                        )
                        min_tr_pm = int(gs.get("min_trades_per_month", 3))
                        # Estimate train-month count from the training table itself
                        train_months_count = int(df[df.get("is_train", True)]["month"].nunique()) if "month" in df.columns else 1
                        min_rows = max(3, min_tr_pm * max(1, train_months_count))
                        if df["fire_rule"].sum() < min_rows:
                            continue
                        X = df.loc[df["fire_rule"], ["LLR_sum","accel","kappa_long","donch_pos","ctx_sim"]].values
                        y = df.loc[df["fire_rule"], "realized_R"].values
                        w = ridge_fit(X, y, lam=lam)
                        Xall = np.c_[df[["LLR_sum","accel","kappa_long","donch_pos","ctx_sim"]].values, np.ones(len(df))]
                        df["S"] = (Xall @ w)
                        taus = np.quantile(df.loc[df["fire_rule"], "S"], np.linspace(0.1,0.9,17))
                        best_tau=None; best_er=-1e9; best_mask=None
                        for tau in taus:
                            mask = df["fire_rule"] & (df["S"] >= tau)
                            n = int(mask.sum())
                            min_tr_pm = int(gs.get("min_trades_per_month", 3))
                            train_months_count = int(df[df.get("is_train", True)]["month"].nunique()) if "month" in df.columns else 1
                            min_rows = max(3, min_tr_pm * max(1, train_months_count))
                            if n < min_rows:
                                continue
                            R = df.loc[mask, "realized_R"].values.tolist()
                            if len(R)==0: continue
                            er = float(np.mean(R))
                            dd = drawdown_R(R)
                            if dd > gs["max_dd_R"]: 
                                continue
                            if er > best_er:
                                best_er = er; best_tau = float(tau); best_mask = mask
                        if best_tau is None:
                            continue
                        candidate = {
                            "eps_mult": float(epsm), "K": int(K), "L": int(L), "bad_margin": float(dm), "breakout_buffer_atr": float(bb),
                            "tau": best_tau, "er": best_er, "w": w.tolist(), "coverage": int(best_mask.sum())
                        }
                        if (best is None) or (candidate["er"] > best["er"]):
                            best = candidate
    if best is None:
        raise RuntimeError(
            "Gating search found no valid configuration (too strict thresholds or insufficient fire_rule rows). "
            "Loosen eps_multiplier/K_hits/bad_margin or widen candidate compression thresholds."
        )
    return best

def prepare_training_table(features: pd.DataFrame,
                           candidates: pd.DataFrame,
                           events: pd.DataFrame,
                           regimes: pd.DataFrame,
                           banks: Dict[str,Any],
                           fp_z,
                           regime_ctx: Dict[str,Any],
                           channels: list,
                           cfg: Dict[str,Any]) -> pd.DataFrame:
    from .utils import cosine_sim, rbf_sim
    sim_type = regime_ctx.get("sim_type", "cosine")
    gamma = float(regime_ctx.get("gamma", 1.0))
    centers_z = np.array(regime_ctx["centers_z"], dtype=float)

    ev = events.merge(
        candidates[["bar_idx","side","donch_high","donch_low","atr","close"]],
        on="bar_idx",
        how="left",
        suffixes=("", "_cand")
    )
    # unify side column (events.side vs candidates.side)
    if "side" not in ev.columns:
        if "side_cand" in ev.columns:
            ev["side"] = ev["side_cand"]
        elif "side_x" in ev.columns:
            ev["side"] = ev["side_x"]
        elif "side_y" in ev.columns:
            ev["side"] = ev["side_y"]
    # optional tidy-up (don’t fail if they’re absent)
    for col in ("side_cand","side_x","side_y"):
        if col in ev.columns:
            try: ev.drop(columns=[col], inplace=True)
            except Exception: pass
    ev["regime_id"] = regimes["regime_id"].reindex(ev["bar_idx"].values).values
    rows=[]
    epsm = cfg["gating_search"]["eps_multiplier"][0]
    for _, r in tqdm(ev.iterrows(), total=len(ev), desc="Build training table", leave=False):
        i0 = int(r["bar_idx"]); side = int(r["side"]); reg_id = int(r["regime_id"]) if not np.isnan(r["regime_id"]) else 0
        good_min, bad_min, gh, bh = compute_min_dists(features, banks, i0, reg_id, channels, epsm)
        llr = (bad_min - good_min)
        flags = {}
        for bb in cfg["gating_search"]["breakout_buffer_atr"]:
            if side==+1:
                flags["breakout_ok_"+str(bb)] = float(r["close"]) >= float(r["donch_high"]) + bb*float(r["atr"])
            else:
                flags["breakout_ok_"+str(bb)] = float(r["close"]) <= float(r["donch_low"])  - bb*float(r["atr"])

        # ctx_sim using z-normalized fingerprint row and its regime center
        if 0 <= reg_id < len(centers_z):
            zrow = np.array(fp_z[i0], dtype=float) if hasattr(fp_z, "__getitem__") else fp_z[i0]
            center = centers_z[reg_id]
            ctx = cosine_sim(zrow, center) if sim_type == "cosine" else rbf_sim(zrow, center, gamma=gamma)
        else:
            ctx = 0.0

        rows.append({
            "bar_idx": i0, "side": side, "regime_id": reg_id,
            "good_min": good_min, "bad_min": bad_min,
            "good_hits": gh, "bad_hits": bh,
            "LLR_sum": llr,
            "accel": float(features["accel"].iloc[i0]) if i0 < len(features) else 0.0,
            "kappa_long": float(features["kappa_long"].iloc[i0]) if i0 < len(features) else 0.0,
            "donch_pos": float(features["donch_pos"].iloc[i0]) if i0 < len(features) else 0.0,
            "ctx_sim": float(ctx),
            **flags,
            "realized_R": float(r["realized_R"]),
            "label": int(r["label"])
        })
    return pd.DataFrame(rows)
