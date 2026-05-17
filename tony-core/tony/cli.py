import argparse
import logging
from . import ingest, score, report
from .utils import parse_years
import os
from datetime import datetime

# Dual logging: console + file
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"tony_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)

def main() -> None:
    parser = argparse.ArgumentParser(description="TONY Diagnostic Tool")
    subparsers = parser.add_subparsers(dest="command")

    p_ingest = subparsers.add_parser("ingest")
    p_ingest.add_argument("--source", required=True)
    p_ingest.add_argument("--ein", required=True)
    p_ingest.add_argument("--years", required=True, type=parse_years)
    p_ingest.add_argument("--out", required=True)

    p_score = subparsers.add_parser("score")
    p_score.add_argument("--input", required=True)
    p_score.add_argument("--entity-type", required=True)
    p_score.add_argument("--horizon", type=int, default=10)
    p_score.add_argument("--out", required=True)

    p_report = subparsers.add_parser("report")
    p_report.add_argument("--input", required=True)
    p_report.add_argument("--format", required=True)

    args = parser.parse_args()

    if args.command == "ingest":
        ingest.run(args.source, args.ein, args.years, args.out)
    elif args.command == "score":
        score.run(args.input, args.entity_type, args.horizon, args.out)
    elif args.command == "report":
        report.run(args.input, args.format)
    else:
        parser.print_help()
