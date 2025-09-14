import argparse
from pathlib import Path
from .config import load_config
from .walkforward import run_walkforward

def main():
    ap = argparse.ArgumentParser(description="CUSUM-Hybrid Motif TF v2 (Regime + LLR gating)")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--mode", required=True, choices=["walkforward"], help="Pipeline mode")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(__file__).resolve().parents[1]
    if args.mode == "walkforward":
        stats = run_walkforward(cfg, root)
        print(stats)

if __name__ == "__main__":
    main()
