import json
import logging
from .utils import file_exists

def run(input_file: str, fmt: str) -> None:
    """Generate a report from scored data."""
    if not file_exists(input_file):
        logging.error(f"Input file {input_file} not found.")
        return

    try:
        with open(input_file, encoding="utf-8") as f:
            result = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.error(f"Could not read/parse {input_file}: {e}")
        return

    if fmt == "brief":
        print("\n=== TONY BRIEF REPORT ===")
        print(f"Entity: {result['entity_type']}")
        print(f"Horizon: {result['horizon']} years")
        print(f"FSF: {result['fsf']}")
        print(f"Persistence: {result['persistence']}")
        print(f"PIS: {result['pis']}")
        print(f"Unity: {result['unity']}")
        print("=========================\n")
    else:
        print(json.dumps(result, indent=2))
