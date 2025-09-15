from pathlib import Path
import pandas as pd, pickle, json

sym = "BTCUSDT"
fold = 0
root = Path(".")/ "outputs" / sym / f"fold_{fold}"
art  = root / "artifacts"

def exists(p): 
    print(("✓" if p.exists() else "✗"), p); 
    return p.exists()

print("== Files ==")
have = {
  "cands": exists(root/"candidates.parquet"),
  "events": exists(root/"events.parquet"),
  "regimes": exists(root/"regimes.csv"),
  "banks": exists(art/"banks.pkl"),
  "gating": exists(art/"gating.json"),
  "train_table": exists(root/"train_table.csv") or exists(root/"train_table_debug.csv"),
}

def safe_read_parquet(p):
    try: return pd.read_parquet(p)
    except Exception as e: print("ERR reading", p, e); return pd.DataFrame()

if have["cands"]:
    cands = safe_read_parquet(root/"candidates.parquet")
    print("\n== Candidates ==", len(cands))
    if len(cands):
        print(cands[["timestamp","bar_idx","side"]].head())

if have["events"]:
    events = safe_read_parquet(root/"events.parquet")
    print("\n== Events (labels) ==", len(events))
    if len(events):
        print(events["label"].value_counts(dropna=False).rename("count").to_frame())
        print("mean realized_R:", float(events["realized_R"].mean()))
else:
    events = pd.DataFrame()

if have["banks"]:
    banks = pickle.load(open(art/"banks.pkl","rb"))
    # count shapelets per regime/kind
    total_g = total_b = 0
    for reg_key, lens in banks.items():
        g=b=0
        for L, bundle in lens.items():
            g += len(bundle["GOOD"].shapelets)
            b += len(bundle["BAD"].shapelets)
        total_g += g; total_b += b
        print(f"[banks] {reg_key}: GOOD={g} BAD={b}")
    print(f"[banks] TOTAL GOOD={total_g} BAD={total_b}")

# Train-table quick view (if present)
tt = None
for name in ["train_table_debug.csv","train_table.csv"]:
    p = root/name
    if p.exists():
        tt = pd.read_csv(p)
        print("\n== Train table ==", len(tt))
        cols = [c for c in ["good_hits","bad_hits","good_min","bad_min","LLR_sum","ctx_sim"] if c in tt.columns]
        print(tt[cols].describe())
        # fireable supply under the most permissive idea (K=1, Δ≥0, BAD==0, any breakout flag present)
        bo_cols = [c for c in tt.columns if c.startswith("breakout_ok_")]
        if bo_cols:
            fire = (tt["good_hits"]>=1) & (tt["bad_hits"]==0) & (tt[bo_cols].any(axis=1))
            print("fireable_rows (K=1, Δ=0, BAD==0, any breakout):", int(fire.sum()))
        else:
            print("No breakout_ok_* columns in train table.")
        break

# Summary diagnosis
print("\n== Diagnosis ==")
if have["cands"] and len(cands)==0:
    print("• Starved at CANDIDATES. Loosen candidate compression/trigger (bbw_p_max, donch_w_p_max, atr_p_max, accel_min, body_frac_min), reduce spacing, disable require_cusum.")
if have["events"] and len(events)==0:
    print("• Starved at TICK LABELS. Make labels easier: lower tp/sl multipliers or increase time_limit_bars in labeling config.")
if have["banks"] and 'banks' in locals() and sum(len(banks[k][L]["GOOD"].shapelets) for k in banks for L in banks[k])==0:
    print("• Starved at BANKS (no GOOD shapelets). Lower ppv_prune, raise eps_percentile/topk_per_class, ensure events have +1 labels.")
if tt is not None:
    if "good_hits" in tt and int((tt["good_hits"]>=1).sum())==0:
        print("• Starved at MATCH TIME (ε too tight). Increase eps_multiplier in gating_search and/or eps_percentile in mining.")
    if "bad_hits" in tt and int((tt["bad_hits"]>0).sum())>0 and "fireable_rows" in locals() and fire.sum()==0:
        print("• BAD filter suppresses everything. Temporarily allow <=1 BAD hit in gate.")
print("Done.")

