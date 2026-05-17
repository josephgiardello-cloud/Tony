from __future__ import annotations
from typing import Dict, Any

def build_ledger(payload: Dict[str, Any], scores: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "meta": {
            "ein": payload.get("ein"),
            "years": [r["year"] for r in payload["records"]],
            "config": {"scoring": cfg["scoring"]},
        },
        "scores": scores["records"],
        "overall": scores["overall"],
        "notes": ["TONY ledger: bounded proxies for stylization/ego decay."],
    }
