import json
from pathlib import Path

import pandas as pd
import pytest

from tony.calibration import calibrate_probabilities, run


def test_calibrate_probabilities_returns_metrics() -> None:
    frame = pd.DataFrame(
        {
            "risk_probability": [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
            "outcome": [0, 0, 0, 1, 1, 1],
        }
    )

    result = calibrate_probabilities(frame, bins=3)
    assert result["metrics"]["rows"] == 6
    assert result["metrics"]["brier_after"] <= result["metrics"]["brier_before"]
    assert len(result["curve"]) > 0


def test_calibrate_probabilities_requires_columns() -> None:
    frame = pd.DataFrame({"score": [0.1, 0.9], "label": [0, 1]})

    with pytest.raises(ValueError, match="missing required columns"):
        calibrate_probabilities(frame)


def test_calibration_run_writes_json(tmp_path: Path) -> None:
    source = tmp_path / "calibration.csv"
    out = tmp_path / "result.json"
    source.write_text("risk_probability,outcome\n0.1,0\n0.9,1\n", encoding="utf-8")

    payload = run(str(source), str(out), bins=2)
    stored = json.loads(out.read_text(encoding="utf-8"))

    assert out.exists()
    assert stored["metrics"]["rows"] == payload["metrics"]["rows"]
