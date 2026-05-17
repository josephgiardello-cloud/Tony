import argparse
import json
import logging
from . import ingest, report, score
from .config import DEFAULT_CONFIG
from .dashboard import main as dashboard_main
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
    p_ingest.add_argument("--source", required=True, help="Path to CSV/Excel/PDF or the literal 'propublica'")
    p_ingest.add_argument("--ein")
    p_ingest.add_argument("--years", default="", type=parse_years)
    p_ingest.add_argument("--config")
    p_ingest.add_argument("--out", required=True)

    p_score = subparsers.add_parser("score")
    p_score.add_argument("--input", required=True)
    p_score.add_argument("--entity-type", required=True)
    p_score.add_argument("--horizon", type=int, default=10)
    p_score.add_argument("--config")
    p_score.add_argument("--out", required=True)

    p_report = subparsers.add_parser("report")
    p_report.add_argument("--input", required=True)
    p_report.add_argument("--format", required=True, choices=["md", "html", "json"])
    p_report.add_argument("--out")

    p_dashboard = subparsers.add_parser("dashboard")
    p_dashboard.add_argument("--input", required=True)
    p_dashboard.add_argument("--host", default="127.0.0.1")
    p_dashboard.add_argument("--port", type=int, default=8000)

    subparsers.add_parser("print-config")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest.run(args.source, args.ein, args.years, args.out, args.config)
    elif args.command == "score":
        score.run(args.input, args.entity_type, args.horizon, args.out, args.config)
    elif args.command == "report":
        report.run(args.input, args.format, args.out)
    elif args.command == "dashboard":
        dashboard_main(args.input, args.host, args.port)
    elif args.command == "print-config":
        print(json.dumps(DEFAULT_CONFIG, indent=2))
    else:
        parser.print_help()



if __name__ == "__main__":
    main()
