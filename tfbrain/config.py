from typing import Dict, Any
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise RuntimeError("Config must be a YAML mapping at the top level")

    REQUIRED_TOP = {"paths","symbols","data","features","cusum","candidates","label","mining","gating_search","regime","risk","walkforward"}
    unknown = set(cfg.keys()) - REQUIRED_TOP
    # We allow extras, but block likely typos
    danger = [k for k in unknown if k.lower() in {"feature", "gatings", "gate", "walkfroward", "minnings"}]
    if danger:
        raise RuntimeError(f"Suspicious config keys {danger}. Did you mean one of {sorted(REQUIRED_TOP)}?")

    # Minimal defaults to keep current behavior
    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("inputs_dir", "inputs")
    cfg["paths"].setdefault("outputs_dir", "outputs")

    # Keep other sections as-is; specific validations happen in preflight
    return cfg
