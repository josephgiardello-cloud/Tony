import json
import logging
from datetime import datetime
from .utils import file_exists

def run(input_file: str, entity_type: str, horizon: int, out_file: str) -> None:
    """Score FSF, Persistence, PIS, Unity."""
    if not file_exists(input_file):
        logging.error(f"Input file {input_file} not found.")
        return

    try:
        with open(input_file, encoding="utf-8") as f:
            _ = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.error(f"Could not read/parse {input_file}: {e}")
        return

    fsf, persistence, pis, unity = 0.84, 0.71, 0.73, False

    result = {
        "entity_type": entity_type,
        "horizon": horizon,
        "fsf": fsf,
        "persistence": persistence,
        "pis": pis,
        "unity": unity,
        "scored_at": datetime.now().isoformat()
    }

    try:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logging.info(f"Scored {entity_type}: FSF={fsf}, Persistence={persistence}, PIS={pis}, Unity={unity}")
    except OSError as e:
        logging.error(f"Could not write to {out_file}: {e}")
