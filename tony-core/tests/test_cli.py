import json
import subprocess
import sys
from pathlib import Path

import tony.cli as cli


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


def test_cli_ingest_uses_env_ein(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_ingest_run(source, ein, years, out, config):
        captured["source"] = source
        captured["ein"] = ein
        captured["years"] = years
        captured["out"] = out
        captured["config"] = config
        return {"records": []}

    monkeypatch.setenv("PROPUBLICA_EIN", "530196605")
    monkeypatch.delenv("TONY_EIN", raising=False)
    monkeypatch.setattr(cli.ingest, "run", _fake_ingest_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tony",
            "ingest",
            "--source",
            "propublica",
            "--out",
            str(tmp_path / "normalized.json"),
        ],
    )

    cli.main()
    assert captured["ein"] == "530196605"


def test_cli_score_defaults_entity_type(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_score_run(input_file, entity_type, horizon, out_file, config_path=None):
        captured["input"] = input_file
        captured["entity_type"] = entity_type
        captured["horizon"] = horizon
        captured["out"] = out_file
        captured["config_path"] = config_path
        return {"summary": {"descriptor": "Unknown"}}

    monkeypatch.setattr(cli.score, "run", _fake_score_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tony",
            "score",
            "--input",
            str(tmp_path / "input.json"),
            "--out",
            str(tmp_path / "scored.json"),
        ],
    )

    cli.main()
    assert captured["entity_type"] == "nonprofit"