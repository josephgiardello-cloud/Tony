import json
from pathlib import Path

import pytest

from tony.score import run, score_risk_adjustable


def test_score_risk_adjustable_returns_low_risk_descriptor() -> None:
    result = score_risk_adjustable(
        {
            "unrestricted_net_assets": 120000,
            "total_expenses": 240000,
            "expenses": 240000,
            "revenue": 300000,
            "assets": 400000,
            "liabilities": 120000,
            "program_expenses": 180000,
        },
        "hospital",
        12,
        6.0,
        3.0,
    )

    assert result["ContinuityDescriptor"] == "Low Risk (Excellent)"
    assert result["ModelRiskProbability"] < 0.4


def test_score_run_writes_history(normalized_payload: Path, tmp_path: Path) -> None:
    out_file = tmp_path / "scored.json"
    result = run(str(normalized_payload), "nonprofit", 12, str(out_file))
    stored = json.loads(out_file.read_text(encoding="utf-8"))

    assert stored["summary"]["descriptor"] == result["summary"]["descriptor"]
    assert len(stored["history"]) == 2
    assert stored["history"][1]["risk_probability"] < 0.5


def test_score_risk_adjustable_sparse_payload_returns_unknown_descriptor() -> None:
    result = score_risk_adjustable(
        {
            "year": 2025,
            "revenue": 0,
            "expenses": 0,
        },
        "nonprofit",
        12,
        6.0,
        3.0,
    )
    assert result["ContinuityDescriptor"] == "Unknown"
    assert 0.0 <= result["ModelRiskProbability"] <= 1.0


def test_score_run_raises_on_empty_records(tmp_path: Path) -> None:
    payload = {
        "metadata": {"source": "test"},
        "records": [],
    }
    input_file = tmp_path / "empty.json"
    out_file = tmp_path / "never_written.json"
    input_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="No records available"):
        run(str(input_file), "nonprofit", 12, str(out_file))
