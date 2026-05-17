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
    assert "NormalizedFeatures" in result
    assert "feature_contributions" in result
    assert len(result["top_drivers"]) == 3


def test_score_run_writes_history(normalized_payload: Path, tmp_path: Path) -> None:
    out_file = tmp_path / "scored.json"
    result = run(str(normalized_payload), "nonprofit", 12, str(out_file))
    stored = json.loads(out_file.read_text(encoding="utf-8"))

    assert stored["summary"]["descriptor"] == result["summary"]["descriptor"]
    assert len(stored["history"]) == 2
    assert stored["history"][1]["risk_probability"] < 0.5
    assert "normalized_features" in stored["summary"]
    assert "feature_contributions" in stored["summary"]


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


def test_score_continuity_is_capped_for_tiny_expenses() -> None:
    result = score_risk_adjustable(
        {
            "year": 2025,
            "revenue": 500000,
            "expenses": 1,
            "assets": 600000,
            "liabilities": 150000,
            "unrestricted_net_assets": 900000,
            "program_expenses": 300000,
        },
        "nonprofit",
        12,
        6.0,
        3.0,
    )
    assert result["ContinuityRiskScore"] <= 120


def test_score_summary_includes_compensation_burden(tmp_path: Path) -> None:
    payload = {
        "metadata": {"source": "test"},
        "records": [
            {
                "year": 2023,
                "revenue": 1000000,
                "expenses": 900000,
                "assets": 1200000,
                "liabilities": 400000,
                "unrestricted_net_assets": 700000,
                "program_expenses": 600000,
                "executive_compensation": 120000,
                "staff_salaries": 260000,
                "admin_salaries": 80000,
            }
        ],
    }
    in_file = tmp_path / "comp_input.json"
    out_file = tmp_path / "comp_scored.json"
    in_file.write_text(json.dumps(payload), encoding="utf-8")

    result = run(str(in_file), "nonprofit", 12, str(out_file))
    summary = result["summary"]

    assert summary["executive_compensation"] == 120000.0
    assert summary["staff_salaries"] == 260000.0
    assert summary["admin_salaries"] == 80000.0
    assert summary["total_salaries"] == 460000.0
    assert summary["executive_salary_ratio"] == pytest.approx(0.133333, abs=1e-6)
    assert summary["salaries_to_expense_ratio"] == pytest.approx(0.511111, abs=1e-6)


def test_score_does_not_infer_total_salaries_from_exec_only(tmp_path: Path) -> None:
    payload = {
        "metadata": {"source": "test"},
        "records": [
            {
                "year": 2023,
                "revenue": 1000000,
                "expenses": 900000,
                "assets": 1200000,
                "liabilities": 400000,
                "unrestricted_net_assets": 700000,
                "program_expenses": 600000,
                "executive_compensation": 120000,
                "staff_salaries": None,
                "admin_salaries": None,
                "total_salaries": None,
            }
        ],
    }
    in_file = tmp_path / "comp_exec_only_input.json"
    out_file = tmp_path / "comp_exec_only_scored.json"
    in_file.write_text(json.dumps(payload), encoding="utf-8")

    result = run(str(in_file), "nonprofit", 12, str(out_file))
    summary = result["summary"]

    assert summary["executive_compensation"] == 120000.0
    assert summary["total_salaries"] is None
    assert summary["salaries_to_expense_ratio"] is None


def test_score_summary_includes_standard_grant_metrics_with_accurate_yoy(tmp_path: Path) -> None:
    payload = {
        "metadata": {"source": "test"},
        "records": [
            {
                "year": 2022,
                "revenue": 1000000,
                "expenses": 900000,
                "assets": 1200000,
                "liabilities": 400000,
                "unrestricted_net_assets": 600000,
                "program_expenses": 650000,
            },
            {
                "year": 2023,
                "revenue": 1100000,
                "expenses": 950000,
                "assets": 1300000,
                "liabilities": 420000,
                "unrestricted_net_assets": 660000,
                "program_expenses": 700000,
            },
        ],
    }
    in_file = tmp_path / "grant_metrics_input.json"
    out_file = tmp_path / "grant_metrics_scored.json"
    in_file.write_text(json.dumps(payload), encoding="utf-8")

    result = run(str(in_file), "nonprofit", 12, str(out_file))
    standard = result["summary"]["standard_grant_metrics"]

    assert standard["revenue_growth_yoy"] == pytest.approx(0.1, abs=1e-6)
    assert standard["expense_growth_yoy"] == pytest.approx((950000 - 900000) / 900000, abs=1e-4)
    assert standard["net_assets_growth_yoy"] == pytest.approx(0.1, abs=1e-6)
    assert standard["program_expense_ratio"] == pytest.approx(700000 / 950000, abs=1e-4)
    assert standard["liabilities_to_assets"] == pytest.approx(420000 / 1300000, abs=1e-4)


def test_score_grant_recommendation_flags_elevated_risk(tmp_path: Path) -> None:
    payload = {
        "metadata": {"source": "test"},
        "records": [
            {
                "year": 2023,
                "revenue": 400000,
                "expenses": 520000,
                "assets": 250000,
                "liabilities": 320000,
                "unrestricted_net_assets": 20000,
                "program_expenses": 280000,
            }
        ],
    }
    in_file = tmp_path / "grant_reco_input.json"
    out_file = tmp_path / "grant_reco_scored.json"
    in_file.write_text(json.dumps(payload), encoding="utf-8")

    result = run(str(in_file), "nonprofit", 12, str(out_file))
    recommendation = result["summary"]["grant_recommendation"]

    assert recommendation["label"] in {"Conditional", "Elevated Risk"}
    assert isinstance(recommendation["reasons"], list)


def test_score_summary_exposes_primary_final_risk_fields(tmp_path: Path) -> None:
    payload = {
        "metadata": {"source": "test"},
        "records": [
            {
                "year": 2023,
                "revenue": 1000000,
                "expenses": 900000,
                "assets": 1200000,
                "liabilities": 400000,
                "unrestricted_net_assets": 700000,
                "program_expenses": 600000,
            }
        ],
    }
    in_file = tmp_path / "final_primary_input.json"
    out_file = tmp_path / "final_primary_scored.json"
    in_file.write_text(json.dumps(payload), encoding="utf-8")

    result = run(str(in_file), "nonprofit", 12, str(out_file))
    summary = result["summary"]

    assert "final_risk_probability" in summary
    assert "final_risk_index" in summary
    assert "key_drivers" in summary
    assert 0.0 <= summary["final_risk_probability"] <= 1.0
    assert 0.0 <= summary["final_risk_index"] <= 100.0


def test_score_result_includes_layered_details_and_preset(tmp_path: Path) -> None:
    payload = {
        "metadata": {"source": "test"},
        "records": [
            {
                "year": 2022,
                "revenue": 1000000,
                "expenses": 940000,
                "assets": 1150000,
                "liabilities": 390000,
                "unrestricted_net_assets": 600000,
                "program_expenses": 680000,
            },
            {
                "year": 2023,
                "revenue": 1030000,
                "expenses": 960000,
                "assets": 1180000,
                "liabilities": 400000,
                "unrestricted_net_assets": 620000,
                "program_expenses": 690000,
            },
        ],
    }
    in_file = tmp_path / "layered_details_input.json"
    out_file = tmp_path / "layered_details_scored.json"
    in_file.write_text(json.dumps(payload), encoding="utf-8")

    result = run(str(in_file), "nonprofit", 12, str(out_file))
    details = result["details"]

    assert "layer_1_raw_features" in details
    assert "layer_2_base_model" in details
    assert "layer_3_adjustments" in details
    assert "layer_4_final" in details
    assert details["layer_3_adjustments"]["preset"] == "balanced"
