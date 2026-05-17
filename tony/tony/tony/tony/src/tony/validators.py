from __future__ import annotations
import re
from typing import Dict, Any, Iterable, Set

EIN_RE = re.compile(r"^\d{2}-\d{7}$")

class ValidationError(ValueError): pass

def validate_ein(ein: str) -> None:
    if not EIN_RE.match(ein):
        raise ValidationError(f"Invalid EIN format: {ein} (expected NN-NNNNNNN)")

def check_duplicate_years(records: Iterable[Dict[str, Any]], allow_duplicates: bool) -> None:
    seen: Set[int] = set(); dups: Set[int] = set()
    for r in records:
        year = r.get("year")
        if not isinstance(year, int):
            raise ValidationError(f"Record missing or invalid 'year': {r}")
        if year in seen: dups.add(year)
        seen.add(year)
    if dups and not allow_duplicates:
        raise ValidationError(f"Duplicate years detected: {sorted(dups)}")

def validate_input_payload(payload: Dict[str, Any], vcfg: Dict[str, Any]) -> None:
    if vcfg.get("require_ein"):
        ein = payload.get("ein")
        if not isinstance(ein, str):
            raise ValidationError("Missing 'ein' string in payload.")
        validate_ein(ein)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValidationError("Payload must include non-empty 'records' list.")
    check_duplicate_years(records, vcfg.get("allow_duplicate_years", False))
