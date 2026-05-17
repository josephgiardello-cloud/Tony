import json
from pathlib import Path

import pandas as pd

from tony.evaluation import evaluate_probabilities, run


def test_evaluate_probabilities_returns_field_comparison() -> None:
    frame = pd.DataFrame(
        {
            "risk_probability": [0.95, 0.82, 0.71, 0.35, 0.21, 0.12],
            "outcome": [1, 1, 1, 0, 0, 0],
        }
    )

    result = evaluate_probabilities(frame, bins=4, method="holdout", test_size=0.33, random_state=7)

    assert result["summary"]["rows"] == 6
    assert "raw" in result["summary"]
    assert "calibrated" in result["summary"]
    assert "no_skill" in result["summary"]
    assert result["validation"]["method"] == "holdout"
    assert "out_of_sample" in result["validation"]
    assert result["field_comparison"]["dataset_scope"] == "external field benchmark"


def test_evaluate_probabilities_supports_kfold() -> None:
    frame = pd.DataFrame(
        {
            "risk_probability": [0.95, 0.88, 0.81, 0.72, 0.62, 0.38, 0.31, 0.21, 0.14, 0.08],
            "outcome": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        }
    )

    result = evaluate_probabilities(frame, bins=4, method="kfold", folds=5, random_state=3)

    assert result["validation"]["method"] == "kfold"
    assert result["validation"]["folds"] == 5
    assert len(result["validation"]["fold_details"]) == 5
    assert "out_of_sample" in result["validation"]


def test_evaluation_run_writes_json(tmp_path: Path) -> None:
    source = tmp_path / "evaluation.csv"
    source.write_text("risk_probability,outcome\n0.9,1\n0.8,1\n0.2,0\n0.1,0\n", encoding="utf-8")

    out = tmp_path / "evaluation.json"
    result = run(str(source), str(out), bins=3, method="holdout", test_size=0.5, random_state=11)

    stored = json.loads(out.read_text(encoding="utf-8"))
    assert stored["summary"]["rows"] == result["summary"]["rows"]
    assert stored["validation"]["method"] == "holdout"
