import json
import subprocess
import sys
from pathlib import Path


def test_cli_ingest_and_score_round_trip(sample_csv: Path, tmp_path: Path) -> None:
    normalized = tmp_path / "normalized.json"
    scored = tmp_path / "scored.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "tony.cli",
            "ingest",
            "--source",
            str(sample_csv),
            "--out",
            str(normalized),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tony.cli",
            "score",
            "--input",
            str(normalized),
            "--entity-type",
            "nonprofit",
            "--out",
            str(scored),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(scored.read_text(encoding="utf-8"))
    assert payload["summary"]["descriptor"] in {
        "Low Risk (Excellent)",
        "Moderate Risk (Acceptable)",
        "High Risk (Insufficient)",
    }