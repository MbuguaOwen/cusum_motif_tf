from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from .utils import to_ms, read_csv_or_parquet

def _find_month_file(base: Path, symbol: str, month: str):
    # recursive patterns, prefer parquet if both exist
    cands = []
    pats = [
        f"{symbol}_{month}.parquet", f"{symbol}_{month}.csv",
        f"{symbol}-{month}.parquet", f"{symbol}-{month}.csv",
        f"{symbol}-1m-{month}.parquet", f"{symbol}-1m-{month}.csv",
        f"{symbol}-ticks-{month}.parquet", f"{symbol}-ticks-{month}.csv",
        f"**/{symbol}_{month}.parquet", f"**/{symbol}_{month}.csv",
        f"**/{symbol}-{month}.parquet", f"**/{symbol}-{month}.csv",
        f"**/{symbol}-1m-{month}.parquet", f"**/{symbol}-1m-{month}.csv",
        f"**/{symbol}-ticks-{month}.parquet", f"**/{symbol}-ticks-{month}.csv",
    ]
    for p in pats: cands.extend(base.glob(p))
    cands = sorted(set(cands), key=lambda p: (p.suffix != ".parquet", str(p)))
    return cands[0] if cands else None

def _read_bars_df(fp: Path) -> pd.DataFrame:
    if fp.suffix.lower() == ".csv":
        df = pd.read_csv(fp, header=0)
        # Detect headerless by checking if first column name looks numeric
        first_col_name = str(df.columns[0])
        if first_col_name.isdigit() or first_col_name.startswith("173") or df.shape[1] >= 6 and "timestamp" not in df.columns and "open_time" not in df.columns:
            df = pd.read_csv(fp, header=None)
            # Binance kline 12 cols
            cols12 = ["open_time","open","high","low","close","volume",
                      "close_time","quote_volume","n_trades",
                      "taker_buy_base","taker_buy_quote","ignore"]
            df = df.iloc[:, :min(df.shape[1], 12)]
            df.columns = cols12[:df.shape[1]]
    else:
        df = pd.read_parquet(fp)

    # Normalize columns
    lc = {c.lower(): c for c in df.columns}
    if "timestamp" in lc:
        ts = df[lc["timestamp"]]
    elif "open_time" in lc:
        ts = df[lc["open_time"]]
    elif df.index.name is not None:
        df = df.reset_index()
        ts = df[df.columns[0]]
    else:
        raise ValueError("Bars file missing a timestamp/open_time column")

    # Build normalized frame
    out = pd.DataFrame()
    out["timestamp"] = ts.apply(to_ms)

    def pick(*names):
        for n in names:
            if n in lc: return df[lc[n]]
        return None

    for want, alts in {
        "open":  ["open","o","Open"],
        "high":  ["high","h","High"],
        "low":   ["low","l","Low"],
        "close": ["close","c","Close"],
        "volume":["volume","v","Volume","quote_volume","Volume USDT","Volume USD"]
    }.items():
        s = pick(*alts)
        if s is None: raise ValueError(f"Bars file missing required column: {want}")
        out[want] = pd.to_numeric(s, errors="coerce")

    out = out.dropna(subset=["timestamp","open","high","low","close","volume"])
    return out[["timestamp","open","high","low","close","volume"]]

def load_bars_1m(inputs_dir: Path, symbol: str, months: List[str]) -> pd.DataFrame:
    base = inputs_dir / "bars_1m"
    rows = []
    for m in months:
        fp = _find_month_file(base, symbol, m)
        if fp is None:
            raise FileNotFoundError(
                f"Bars missing for {symbol} {m}. "
                f"Looked for {symbol}_{{YYYY-MM}}.* or {symbol}-1m-{{YYYY-MM}}.* anywhere under {base}."
            )
        df = _read_bars_df(fp)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True).sort_values("timestamp")
    return out

def _normalize_tick_columns(df: pd.DataFrame) -> pd.DataFrame:
    lc = {c.lower(): c for c in df.columns}
    cols = {}
    # timestamp
    for k in ["timestamp","ts","time"]: 
        if k in lc: cols["timestamp"] = lc[k]; break
    if "timestamp" not in cols: raise ValueError("Ticks file missing timestamp/ts/time")
    # price
    for k in ["price","p","last_price"]:
        if k in lc: cols["price"] = lc[k]; break
    if "price" not in cols: raise ValueError("Ticks file missing price")
    # qty
    for k in ["qty","quantity","size","amount","q"]:
        if k in lc: cols["qty"] = lc[k]; break
    if "qty" not in cols: cols["qty"] = None
    # maker flag
    for k in ["is_buyer_maker","isbuyer_maker","buyer_maker"]:
        if k in lc: cols["is_buyer_maker"] = lc[k]; break

    out = pd.DataFrame()
    t = df[cols["timestamp"]]
    # try ms first, fallback to parse
    try:
        t = pd.to_numeric(t, errors="coerce")
        out["timestamp"] = t.apply(to_ms)
    except Exception:
        out["timestamp"] = pd.to_datetime(t, utc=True).view("int64") // 10**6
    out["price"] = pd.to_numeric(df[cols["price"]], errors="coerce")
    out["qty"] = pd.to_numeric(df[cols["qty"]], errors="coerce") if cols["qty"] else 0.0
    out["is_buyer_maker"] = df[cols["is_buyer_maker"]].astype(int) if "is_buyer_maker" in cols else 0
    return out[["timestamp","price","qty","is_buyer_maker"]].dropna(subset=["timestamp","price"])

def load_ticks(inputs_dir: Path, symbol: str, months: List[str]) -> pd.DataFrame:
    base = inputs_dir / "ticks"
    rows = []
    for m in months:
        fp = _find_month_file(base, symbol, m)
        if fp is None:
            raise FileNotFoundError(
                f"Ticks missing for {symbol} {m}. "
                f"Looked for {symbol}_{{YYYY-MM}}.* or {symbol}-ticks-{{YYYY-MM}}.* under {base}."
            )
        # ticks always csv in your sample
        df = pd.read_csv(fp)
        df = _normalize_tick_columns(df)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True).sort_values("timestamp")
    return out
