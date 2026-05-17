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


def test_cli_calibrate_dispatches(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_calibration_run(input_file, out_file, bins=10):
        captured["input_file"] = input_file
        captured["out_file"] = out_file
        captured["bins"] = bins
        return {"metrics": {"rows": 1}}

    monkeypatch.setattr(cli.calibration, "run", _fake_calibration_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tony",
            "calibrate",
            "--input",
            str(tmp_path / "bench.csv"),
            "--bins",
            "7",
            "--out",
            str(tmp_path / "calibrated.json"),
        ],
    )

    cli.main()
    assert captured["bins"] == 7


def test_cli_evaluate_dispatches(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_evaluation_run(input_file, out_file, bins=10, method="holdout", test_size=0.3, folds=5, random_state=42, stratified=True):
        captured["input_file"] = input_file
        captured["out_file"] = out_file
        captured["bins"] = bins
        captured["method"] = method
        captured["test_size"] = test_size
        captured["folds"] = folds
        captured["random_state"] = random_state
        captured["stratified"] = stratified
        return {"summary": {"rows": 1}}

    monkeypatch.setattr(cli.evaluation, "run", _fake_evaluation_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tony",
            "evaluate",
            "--input",
            str(tmp_path / "bench.csv"),
            "--bins",
            "9",
            "--method",
            "kfold",
            "--folds",
            "4",
            "--random-state",
            "7",
            "--no-stratified",
            "--out",
            str(tmp_path / "evaluation.json"),
        ],
    )

    cli.main()
    assert captured["bins"] == 9
    assert captured["method"] == "kfold"
    assert captured["folds"] == 4
    assert captured["random_state"] == 7
    assert captured["stratified"] is False


def test_cli_compliance_dispatches(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_compliance_run(input_file, out_file):
        captured["input_file"] = input_file
        captured["out_file"] = out_file
        return {"overall_score": 50.0}

    monkeypatch.setattr(cli.compliance, "run", _fake_compliance_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tony",
            "compliance-audit",
            "--input",
            str(tmp_path / "profile.json"),
            "--out",
            str(tmp_path / "report.json"),
        ],
    )

    cli.main()
    assert captured["input_file"]