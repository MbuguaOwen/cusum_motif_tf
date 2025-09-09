import numpy as np
import pandas as pd
from tqdm import tqdm


def first_touch_labels(candidates: pd.DataFrame,
                       ticks: pd.DataFrame,
                       sl_mult: float,
                       tp_mult: float,
                       timeout_bars: int,
                       bar_seconds: int = 60,
                       slippage_bps: int = 3):
    """
    Label each candidate (with 'side') using tick-level first-touch.
    LONG: tp = entry + tp_mult*ATR; sl = entry - sl_mult*ATR.
    SHORT: tp = entry - tp_mult*ATR; sl = entry + sl_mult*ATR.
    Timeout yields label 0 and R=0.
    Returns DataFrame with: ts, bar_idx, side, entry_price, atr, tp_price, sl_price,
    exit_ts, exit_price, label, R, touched, latency_bars.
    """
    events = []
    if len(candidates) == 0:
        return pd.DataFrame(columns=[
            "ts","bar_idx","side","entry_price","atr","tp_price","sl_price","exit_ts","exit_price","label","R","touched","latency_bars"
        ])
    ticks = ticks.sort_values("timestamp").reset_index(drop=True)
    t_idx = 0
    t_times = ticks["timestamp"].values
    t_prices = ticks["price"].values.astype(float)
    for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Tick label"):
        t0 = int(row["timestamp"] if "timestamp" in row else row.get("ts"))
        side = str(row.get("side", "long"))
        entry = float(row.get("entry_price", row.get("close")))
        atr0 = float(row.get("atr", 0.0))
        if not np.isfinite(atr0) or atr0 <= 0:
            continue
        if side == "long":
            tp = entry + tp_mult * atr0
            sl = entry - sl_mult * atr0
        else:
            tp = entry - tp_mult * atr0
            sl = entry + sl_mult * atr0
        # find starting tick index
        while t_idx < len(t_times) and t_times[t_idx] < t0:
            t_idx += 1
        label = 0; exit_price = entry; exit_ts = t0; touched = False
        limit_ts = t0 + timeout_bars * bar_seconds * 1000
        j = t_idx
        while j < len(t_times) and t_times[j] <= limit_ts:
            p = t_prices[j]
            if side == "long":
                if p >= tp:
                    label = +1; exit_price = p; exit_ts = int(t_times[j]); touched = True; break
                if p <= sl:
                    label = -1; exit_price = p; exit_ts = int(t_times[j]); touched = True; break
            else:
                if p <= tp:  # tp is lower than entry for shorts
                    label = +1; exit_price = p; exit_ts = int(t_times[j]); touched = True; break
                if p >= sl:
                    label = -1; exit_price = p; exit_ts = int(t_times[j]); touched = True; break
            j += 1
        # apply slippage (bps)
        slip = entry * (slippage_bps / 10000.0)
        if label == +1:
            # winning: adjust favorable exit a bit against us
            exit_price = exit_price - slip if side == "long" else exit_price + slip
        elif label == -1:
            # losing: adjust unfavorable exit a bit more against us
            exit_price = exit_price + slip if side == "long" else exit_price - slip
        # R multiple
        if label == 0:
            R = 0.0
        else:
            R_raw = (exit_price - entry) / (sl_mult * atr0)
            R = float(R_raw if side == "long" else -R_raw)
        latency_bars = int(max(0, round((exit_ts - t0) / (bar_seconds * 1000))))
        events.append({
            "ts": t0,
            "bar_idx": int(row.get("bar_idx", -1)),
            "side": side,
            "entry_price": entry,
            "atr": atr0,
            "tp_price": tp,
            "sl_price": sl,
            "exit_ts": exit_ts,
            "exit_price": float(exit_price),
            "label": int(label),
            "R": float(R),
            "touched": bool(touched),
            "latency_bars": latency_bars,
        })
    return pd.DataFrame(events)
