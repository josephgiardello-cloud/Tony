from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

DEFAULTS: Dict[str, Any] = {
    "defaults": {"log_level": "WARNING", "output_format": "brief", "csv_path": None},
    "scoring": {"sp_weight": 1.0, "echo_weight": 1.0, "attribution_weight": 1.0, "drp_weight": 1.0, "vo_weight": 0.5},
    "validation": {"require_ein": True, "allow_duplicate_years": False},
}

def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return DEFAULTS.copy()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(p.read_text()) or {}
    elif p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
    else:
        raise ValueError("Config must be YAML or JSON.")
    merged = DEFAULTS.copy()
    for k, v in data.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k].update(v)
        else:
            merged[k] = v
    return merged
