from pathlib import Path
from typing import List, Optional
import pandas as pd
from .utils import to_ms, read_csv_or_parquet

def _find_month_file(base: Path, symbol: str, month: str, kind: str | None = None):
    cands = []
    # Common patterns (bars & ticks)
    pats = [
        f"{symbol}_{month}.parquet", f"{symbol}_{month}.csv",
        f"{symbol}-{month}.parquet", f"{symbol}-{month}.csv",
        f"{symbol}-1m-{month}.parquet", f"{symbol}-1m-{month}.csv",
        f"**/{symbol}_{month}.parquet", f"**/{symbol}_{month}.csv",
        f"**/{symbol}-{month}.parquet", f"**/{symbol}-{month}.csv",
        f"**/{symbol}-1m-{month}.parquet", f"**/{symbol}-1m-{month}.csv",
    ]
    # Extra patterns for ticks only (your current naming)
    if (kind or "").lower() == "ticks":
        pats += [
            f"{symbol}-ticks-{month}.parquet",
            f"{symbol}-ticks-{month}.csv",
            f"**/{symbol}-ticks-{month}.parquet",
            f"**/{symbol}-ticks-{month}.csv",
            # common nested layout: inputs/ticks/<SYMBOL>/<SYMBOL>-ticks-YYYY-MM.csv
            f"{symbol}/{symbol}-ticks-{month}.parquet",
            f"{symbol}/{symbol}-ticks-{month}.csv",
            f"**/{symbol}/{symbol}-ticks-{month}.parquet",
            f"**/{symbol}/{symbol}-ticks-{month}.csv",
        ]
    for p in pats:
        cands.extend(base.glob(p))
    cands = sorted(set(cands), key=lambda p: (p.suffix != ".parquet", str(p)))
    return cands[0] if cands else None

def _read_bars_df(fp: Path) -> pd.DataFrame:
    if fp.suffix.lower() == ".csv":
        df = pd.read_csv(fp, header=0)
        first_col_name = str(df.columns[0])
        if first_col_name.isdigit() or df.shape[1] >= 6 and "timestamp" not in df.columns and "open_time" not in df.columns:
            df = pd.read_csv(fp, header=None)
            cols12 = ["open_time","open","high","low","close","volume",
                      "close_time","quote_volume","n_trades",
                      "taker_buy_base","taker_buy_quote","ignore"]
            df = df.iloc[:, :min(df.shape[1], 12)]
            df.columns = cols12[:df.shape[1]]
    else:
        df = pd.read_parquet(fp)
    lc = {c.lower(): c for c in df.columns}
    if "timestamp" in lc: ts = df[lc["timestamp"]]
    elif "open_time" in lc: ts = df[lc["open_time"]]
    elif df.index.name is not None:
        df = df.reset_index(); ts = df[df.columns[0]]
    else:
        raise ValueError("Bars file missing a timestamp/open_time column")
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
    out = out.dropna(subset=["timestamp","open","high","low","close","volume"]
                     )
    return out[["timestamp","open","high","low","close","volume"]]

def _read_ticks_filtered_csv(fp: Path,
                             min_ts: Optional[int],
                             max_ts: Optional[int],
                             chunksize: int = 2_000_000) -> pd.DataFrame:
    """
    Stream a tick CSV and return only two columns: timestamp (int64), price (float32),
    filtered to [min_ts, max_ts] if provided.
    """
    # Discover column names (header-only read)
    header = pd.read_csv(fp, nrows=0)
    lc = {c.lower(): c for c in header.columns}
    tcol = lc.get("timestamp") or lc.get("ts") or lc.get("time")
    pcol = lc.get("price") or lc.get("p") or lc.get("last_price")
    if tcol is None or pcol is None:
        raise ValueError(f"{fp} missing required tick columns (timestamp/price)")

    out_chunks = []
    for chunk in pd.read_csv(fp, usecols=[tcol, pcol], chunksize=chunksize):
        # Normalize column names and dtypes early
        chunk = chunk.rename(columns={tcol: "timestamp", pcol: "price"})
        chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
        chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
        # Drop bad rows before filtering
        chunk = chunk.dropna(subset=["timestamp", "price"])
        # Restrict to time window if given (keeps memory small)
        if min_ts is not None:
            chunk = chunk[chunk["timestamp"] >= min_ts]
        if max_ts is not None:
            chunk = chunk[chunk["timestamp"] <= max_ts]
        if not chunk.empty:
            # compact dtypes
            chunk["timestamp"] = chunk["timestamp"].astype("int64")
            chunk["price"] = chunk["price"].astype("float32")
            out_chunks.append(chunk[["timestamp", "price"]])

    if not out_chunks:
        return pd.DataFrame(columns=["timestamp", "price"])
    df = pd.concat(out_chunks, ignore_index=True)
    return df.sort_values("timestamp")

def load_bars_1m(inputs_dir: Path, symbol: str, months: List[str]) -> pd.DataFrame:
    base = inputs_dir / "bars_1m"
    rows = []
    for m in months:
        fp = _find_month_file(base, symbol, m)
        if fp is None:
            raise FileNotFoundError(f"Bars missing for {symbol} {m} under {base}")
        df = _read_bars_df(fp)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True).sort_values("timestamp")
    return out

def load_ticks(inputs_dir: Path,
               symbol: str,
               months: List[str],
               min_ts: Optional[int] = None,
               max_ts: Optional[int] = None,
               chunksize: int = 2_000_000) -> pd.DataFrame:
    """
    Load ticks for the given months, streamed (CSV) and filtered by [min_ts, max_ts].
    Returns only ['timestamp','price'] to minimize memory for labeling.
    """
    base = inputs_dir / "ticks"
    rows = []
    for m in months:
        fp = _find_month_file(base, symbol, m, kind="ticks")
        if fp is None:
            examples = [
                f"{symbol}-{m}.csv",
                f"{symbol}_{m}.csv",
                f"{symbol}-1m-{m}.csv",
                f"{symbol}-ticks-{m}.csv",
                f"{symbol}/{symbol}-ticks-{m}.csv",
            ]
            raise FileNotFoundError(
                f"Ticks missing for {symbol} {m} under {base}\n"
                f"Tried patterns like: {', '.join(examples)}"
            )

        if fp.suffix.lower() == ".parquet":
            # If parquet exists, select needed cols then filter
            df = pd.read_parquet(fp)
            lc = {c.lower(): c for c in df.columns}
            tcol = lc.get("timestamp") or lc.get("ts") or lc.get("time")
            pcol = lc.get("price") or lc.get("p") or lc.get("last_price")
            if tcol is None or pcol is None:
                raise ValueError(f"{fp} missing required tick columns (timestamp/price)")
            df = df.rename(columns={tcol: "timestamp", pcol: "price"})[["timestamp", "price"]]
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df = df.dropna(subset=["timestamp", "price"])
            if min_ts is not None:
                df = df[df["timestamp"] >= min_ts]
            if max_ts is not None:
                df = df[df["timestamp"] <= max_ts]
            df["timestamp"] = df["timestamp"].astype("int64")
            df["price"] = df["price"].astype("float32")
        else:
            # Stream CSV
            df = _read_ticks_filtered_csv(fp, min_ts=min_ts, max_ts=max_ts, chunksize=chunksize)

        if not df.empty:
            rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "price"])
    out = pd.concat(rows, ignore_index=True).sort_values("timestamp")
    return out[["timestamp", "price"]]
