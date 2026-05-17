from __future__ import annotations
from typing import Dict, Any

def stylization_penalty(record: Dict[str, Any]) -> float: return float(record.get("sp_sim", 0.0))
def echo_contamination(record: Dict[str, Any]) -> float: return float(record.get("echo_sim", 0.0))
def attribution_composite(record: Dict[str, Any]) -> float: return float(record.get("attr_sim", 0.0))
def delayed_reporting_penalty(record: Dict[str, Any]) -> float: return float(record.get("drp_sim", 0.0))
def volatility_overlay(record: Dict[str, Any]) -> float: return float(record.get("vo_sim", 0.0))

def score_record(record: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, float]:
    sp = stylization_penalty(record) * weights["sp_weight"]
    ec = echo_contamination(record) * weights["echo_weight"]
    ac = attribution_composite(record) * weights["attribution_weight"]
    drp = delayed_reporting_penalty(record) * weights["drp_weight"]
    vo = volatility_overlay(record) * weights["vo_weight"]
    total = sp + ec + ac + drp + vo
    return {"sp": sp, "e": ec, "attr": ac, "drp": drp, "vo": vo, "total": total}

def score_payload(payload: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    scored = [{"year": r["year"], **score_record(r, weights)} for r in payload["records"]]
    overall = sum(x["total"] for x in scored) / max(len(scored), 1)
    return {"records": scored, "overall": overall}
