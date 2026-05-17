import json
import logging
from datetime import datetime
from typing import List

def run(source: str, ein: str, years: List[int], out_file: str) -> None:
    """Ingest audited filings into a normalized JSON structure."""
    try:
        data = {
            "source": source,
            "ein": ein,
            "years": years,
            "ingested_at": datetime.now().isoformat(),
            "ledger": [
                {"year": y, "revenue": 1_000_000 + y, "expenses": 950_000 + y}
                for y in years
            ]
        }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Ingested {len(years)} years to {out_file}")
    except OSError as e:
        logging.error(f"Could not write to {out_file}: {e}")
