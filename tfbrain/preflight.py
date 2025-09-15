from __future__ import annotations
from pathlib import Path
import re
import sys
import json
import pandas as pd

RE_BAR = [
    r"{sym}[-_]1m[-_]{ym}\.(?:csv|parquet)",
    r"{sym}[-_]{ym}[-_]1m\.(?:csv|parquet)",
    r"{sym}[-_]{ym}\.(?:csv|parquet)",
]
RE_TICK = [
    r"{sym}[-_]ticks[-_]{ym}\.(?:csv|parquet)",
    r"{sym}[-_]{ym}[-_]ticks\.(?:csv|parquet)",
    r"{sym}[-_]{ym}\.(?:csv|parquet)",
]

def _glob_like(dirpath: Path, patterns: list[str]) -> list[Path]:
    if not dirpath.exists():
        return []
    found = []
    for p in dirpath.rglob("*"):
        name = p.name.lower()
        for pat in patterns:
            if re.fullmatch(pat, name):
                found.append(p)
                break
    return found

def _verify_month_files(root: Path, sub: str, sym: str, months: list[str], patterns: list[str]) -> tuple[list[str], dict]:
    missing = []
    hits = {}
    for ym in months:
        pats = [pat.format(sym=sym.lower(), ym=ym.lower()) for pat in patterns]
        paths = _glob_like(root / sub / sym, pats)
        if not paths:
            # also search nested symbol folders or directly under sub
            paths = _glob_like(root / sub, [pat.format(sym=sym.lower(), ym=ym.lower()) for pat in patterns])
        if not paths:
            missing.append(ym)
        else:
            # pick the first, but keep full list for debugging
            hits[ym] = [str(p) for p in sorted(paths)]
    return missing, hits

def require_columns(fp: Path, required: list[str], nrows: int = 5) -> None:
    try:
        if fp.suffix.lower() == ".parquet":
            df = pd.read_parquet(fp)
        else:
            df = pd.read_csv(fp, nrows=nrows)
            # Heuristic: allow headerless Binance-style 12-col bars CSV
            first_col_name = str(df.columns[0])
            if ("timestamp" not in df.columns and "open_time" not in df.columns) and (first_col_name.isdigit() or df.shape[1] >= 6):
                return  # accept; data loader handles headerless mapping
    except Exception as e:
        raise RuntimeError(f"Failed to read {fp}: {e}")
    cols = {c.lower() for c in df.columns}
    req = [c.lower() for c in required]
    missing = [c for c in req if c not in cols]
    if missing:
        raise RuntimeError(f"{fp} missing columns {missing}. Present: {sorted(cols)}")

def preflight(cfg: dict) -> None:
    # --- schema presence ---
    paths = cfg.get("paths", {})
    inputs = Path(paths.get("inputs_dir", "inputs")).resolve()
    outputs = Path(paths.get("outputs_dir", "outputs")).resolve()
    if not inputs.exists():
        raise RuntimeError(f"inputs_dir does not exist: {inputs}")

    symbols = cfg.get("symbols") or []
    if not symbols:
        raise RuntimeError("Config must declare non-empty 'symbols: [..]'")

    months = cfg.get("data", {}).get("months") or []
    if not months:
        raise RuntimeError("Config must declare non-empty 'data.months: [YYYY-MM, ...]'")

    # --- feature channels existence (names only; actual computation checked later) ---
    channels = cfg.get("features", {}).get("channels") or []
    if not channels:
        raise RuntimeError("features.channels must list at least one feature channel")

    # --- gating grid sanity ---
    gs = cfg.get("gating_search", {})
    if not gs:
        raise RuntimeError("gating_search section is required")
    if not gs.get("eps_multiplier"):
        raise RuntimeError("gating_search.eps_multiplier must be a non-empty list")
    if not gs.get("K_hits"):
        raise RuntimeError("gating_search.K_hits must be a non-empty list")
    if not gs.get("bad_margin"):
        raise RuntimeError("gating_search.bad_margin must be a non-empty list")
    if not gs.get("breakout_buffer_atr"):
        raise RuntimeError("gating_search.breakout_buffer_atr must be a non-empty list")
    if gs.get("ridge_lambda", None) is None:
        raise RuntimeError("gating_search.ridge_lambda must be set")

    # --- data files availability ---
    bars_root = inputs / "bars_1m"
    ticks_root = inputs / "ticks"
    bar_missing = {}
    tick_missing = {}
    examples = {}
    for sym in symbols:
        bm, bhits = _verify_month_files(bars_root, "", sym, months, RE_BAR)
        tm, thits = _verify_month_files(ticks_root, "", sym, months, RE_TICK)
        if bm: bar_missing[sym] = bm
        if tm: tick_missing[sym] = tm
        examples[sym] = {"bars": bhits.get(months[0], [])[:1], "ticks": thits.get(months[0], [])[:1]}

    if bar_missing or tick_missing:
        msg = ["\n❌ Missing files detected:"]
        if bar_missing:
            msg.append("bars_1m:")
            for s, ms in bar_missing.items():
                msg.append(f"  - {s}: {ms}")
        if tick_missing:
            msg.append("ticks:")
            for s, ms in tick_missing.items():
                msg.append(f"  - {s}: {ms}")
        msg.append("\nExamples of recognized patterns per first month:")
        for s, ex in examples.items():
            msg.append(f"  - {s}: bars={ex['bars']} ticks={ex['ticks']}")
        raise RuntimeError("\n".join(msg))

    # --- column checks (sample) ---
    # Take first available bar/tick per first symbol+month and verify key columns
    sym0 = symbols[0]
    ym0 = months[0]
    bar_paths = _glob_like(bars_root / sym0, [p.format(sym=sym0.lower(), ym=ym0.lower()) for p in RE_BAR])
    tick_paths = _glob_like(ticks_root / sym0, [p.format(sym=sym0.lower(), ym=ym0.lower()) for p in RE_TICK])
    if bar_paths:
        require_columns(Path(bar_paths[0]), ["timestamp","open","high","low","close","volume"])
    if tick_paths:
        # allow price/qty/is_buyer_maker minimal
        require_columns(Path(tick_paths[0]), ["timestamp","price"])

    # --- percentiles sanity ---
    cand = cfg.get("candidates", {})
    comp = cand.get("compression", {})
    for k in ["bbw_p_max", "donch_w_p_max", "atr_p_max"]:
        if k in comp:
            v = comp[k]
            if not (0 <= v <= 1):
                raise RuntimeError(f"candidates.compression.{k} must be in [0,1], got {v}")

    print("Preflight OK: schema, files, and basic grids look good.")
