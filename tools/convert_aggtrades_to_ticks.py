# tools/convert_aggtrades_to_ticks.py
from pathlib import Path
import re, json, sys
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x  # fallback if tqdm missing

SRC = Path("inputs/ticks")
REPORT = SRC / "_agg2ticks_report.json"

def convert_file(f: Path):
    rec = {"src": f.name, "dst": None, "in_rows": 0, "out_rows": 0,
           "ts_unit": None, "status": "ok", "msg": ""}
    m = re.match(r"([A-Z0-9]+)-aggTrades-(\d{4})-(\d{2})\.csv$", f.name)
    if not m:
        rec["status"] = "skip"; rec["msg"] = "name_not_matched"
        return rec
    sym, yr, mo = m.groups()
    try:
        df = pd.read_csv(
            f, header=None,
            names=["agg_id","price","qty","first_id","last_id","time","is_buyer_maker","is_best_match"]
        )
        rec["in_rows"] = len(df)

        # microseconds → milliseconds if needed
        if (df["time"].astype("int64") > 10**13).any():
            df["time"] = (df["time"].astype("int64") // 1000)
            rec["ts_unit"] = "us→ms"
        else:
            rec["ts_unit"] = "ms"

        df["is_buyer_maker"] = df["is_buyer_maker"].map(
            {True:1, False:0, "True":1, "False":0, 1:1, 0:0}
        ).fillna(0).astype(int)

        out = df.rename(columns={"time":"timestamp"})[
            ["timestamp","price","qty","is_buyer_maker"]
        ]
        rec["out_rows"] = len(out)

        dst = f.with_name(f"{sym}-ticks-{yr}-{mo}.csv")
        out.to_csv(dst, index=False)
        rec["dst"] = dst.name
        return rec
    except Exception as e:
        rec["status"] = "error"; rec["msg"] = str(e)
        return rec

def main():
    files = sorted(SRC.glob("*-aggTrades-*.csv"))
    if not files:
        print("No *-aggTrades-YYYY-MM.csv files found under inputs/ticks.")
        sys.exit(0)

    rows = []
    for f in tqdm(files, desc="Converting aggTrades → ticks"):
        rows.append(convert_file(f))

    # write report
    with REPORT.open("w", encoding="utf-8") as w:
        json.dump(rows, w, indent=2)

    # console summary
    ok = sum(r["status"] == "ok" for r in rows)
    skip = sum(r["status"] == "skip" for r in rows)
    err = sum(r["status"] == "error" for r in rows)
    print(f"\nDone ✅  Converted: {ok} | Skipped: {skip} | Errors: {err}")
    print(f"Report → {REPORT}")

if __name__ == "__main__":
    main()
