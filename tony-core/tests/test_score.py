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
    assert "shap_linear_logit_values" in stored["summary"]
    assert "shap_base_logit" in stored["summary"]


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


def test_score_includes_altman_z_baseline_and_zone(tmp_path: Path) -> None:
    payload = {
        "metadata": {"source": "test"},
        "records": [
            {
                "year": 2023,
                "revenue": 1200000,
                "expenses": 1000000,
                "assets": 2000000,
                "liabilities": 800000,
                "unrestricted_net_assets": 1200000,
                "program_expenses": 760000,
                "current_assets": 900000,
                "current_liabilities": 400000,
                "retained_earnings": 1100000,
                "ebit": 200000,
                "book_value_equity": 1200000,
            }
        ],
    }
    in_file = tmp_path / "altman_input.json"
    out_file = tmp_path / "altman_scored.json"
    in_file.write_text(json.dumps(payload), encoding="utf-8")

    result = run(str(in_file), "nonprofit", 12, str(out_file))
    standard = result["summary"]["standard_grant_metrics"]
    altman = result["summary"]["altman"]

    assert standard["altman_z_score"] is not None
    assert standard["altman_zone"] in {"safe", "grey", "distress"}
    assert altman["altman_z_score"] == standard["altman_z_score"]
    assert result["details"]["layer_4_final"]["altman_zone"] == standard["altman_zone"]


def test_score_supports_zscore_peer_benchmark_mode(tmp_path: Path) -> None:
    payload = {
        "metadata": {"source": "test"},
        "records": [
            {
                "year": 2023,
                "revenue": 1200000,
                "expenses": 1000000,
                "assets": 2000000,
                "liabilities": 800000,
                "unrestricted_net_assets": 1200000,
                "program_expenses": 760000,
            }
        ],
    }
    cfg = {
        "weights": {
            "continuity_months": 0.35,
            "operating_margin": 0.2,
            "program_expense_ratio": 0.2,
            "liabilities_to_assets": 0.15,
            "revenue_volatility": 0.1,
        },
        "thresholds": {
            "continuity_low": 6.0,
            "continuity_moderate": 3.0,
            "risk_probability_high": 0.66,
            "risk_probability_moderate": 0.4,
        },
        "model": {
            "random_state": 42,
            "max_iter": 1000,
            "class_weight": "balanced",
            "cache_models": False,
            "reference_profiles": [
                {
                    "continuity_months": 2.0,
                    "operating_margin": -0.1,
                    "program_expense_ratio": 0.5,
                    "liabilities_to_assets": 1.0,
                    "revenue_volatility": 0.5,
                    "risk_label": 1,
                },
                {
                    "continuity_months": 9.0,
                    "operating_margin": 0.08,
                    "program_expense_ratio": 0.8,
                    "liabilities_to_assets": 0.35,
                    "revenue_volatility": 0.1,
                    "risk_label": 0,
                },
            ],
        },
        "scoring_preset": "balanced",
        "scoring_presets": {
            "balanced": {
                "benchmark_gap": 0.25,
                "confidence_penalty": 0.2,
                "donor_penalty": 0.15,
                "cashflow_penalty": 0.15,
                "compliance_penalty": 0.1,
                "irs_penalty": 0.15,
                "altman_penalty": 0.2,
                "trend_relief": 0.2,
                "charity_relief": 0.15,
            }
        },
        "peer_benchmark": {
            "mode": "zscore_only",
            "zscore_clip": 3.0,
            "percentile_weight": 0.7,
            "zscore_weight": 0.3,
        },
        "time_weighting": {"enabled": False},
        "uncertainty": {"enabled": False},
        "final_index": {"ml_weight": 0.7, "health_weight": 0.3},
        "normalization": {},
        "labeling": {
            "risk_points_cutoff": 3,
            "severe_margin": -0.1,
            "weak_program_ratio": 0.55,
            "high_revenue_volatility": 0.35,
        },
        "entity_profiles": {},
        "entity_thresholds": {},
        "altman_z": {"safe_threshold": 2.6, "distress_threshold": 1.1},
    }

    in_file = tmp_path / "peer_mode_input.json"
    out_file = tmp_path / "peer_mode_scored.json"
    cfg_file = tmp_path / "peer_mode_config.json"
    in_file.write_text(json.dumps(payload), encoding="utf-8")
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

    result = run(str(in_file), "nonprofit", 12, str(out_file), config_path=str(cfg_file))
    summary = result["summary"]

    assert summary["peer_benchmark_mode"] == "zscore_only"
    assert summary["peer_zscore_score"] is not None
    assert summary["peer_benchmark_score"] == summary["peer_zscore_score"]
    assert "peer_group_filters" in summary


def test_score_includes_broader_organizational_and_financial_depth_metrics(tmp_path: Path) -> None:
    payload = {
        "metadata": {
            "board_independence": 0.8,
            "board_turnover_rate": 0.15,
            "succession_plan_score": 0.7,
            "conflict_of_interest_policy_score": 0.9,
            "board_diversity_index": 0.65,
            "outcome_achievement_rate": 0.75,
            "impact_evaluation_score": 0.7,
            "beneficiary_reach_growth": 0.6,
            "cybersecurity_maturity": 0.5,
            "business_continuity_plan_score": 0.6,
            "key_person_risk": 0.4,
            "staff_turnover_rate": 0.2,
            "volunteer_engagement_score": 0.7,
            "training_investment_score": 0.55,
            "dei_index": 0.6,
            "open_litigation_count": 1,
            "watchdog_flags": 0,
            "adverse_media_score": 0.1,
            "whistleblower_cases": 0,
        },
        "records": [
            {
                "year": 2023,
                "revenue": 1400000,
                "expenses": 1200000,
                "assets": 2600000,
                "liabilities": 900000,
                "unrestricted_net_assets": 1300000,
                "program_expenses": 860000,
                "current_assets": 700000,
                "current_liabilities": 300000,
                "fundraising_expense": 90000,
                "contributions_revenue": 450000,
                "investment_income": 70000,
                "endowment_assets": 1000000,
                "endowment_draw": 45000,
            }
        ],
    }

    in_file = tmp_path / "org_health_input.json"
    out_file = tmp_path / "org_health_scored.json"
    in_file.write_text(json.dumps(payload), encoding="utf-8")

    result = run(str(in_file), "nonprofit", 12, str(out_file))
    summary = result["summary"]

    assert summary["current_ratio"] == pytest.approx(700000 / 300000, abs=1e-4)
    assert summary["working_capital_ratio"] == pytest.approx((700000 - 300000) / 2600000, abs=1e-4)
    assert summary["fundraising_cost_to_raise_dollar"] == pytest.approx(0.2, abs=1e-6)
    assert summary["endowment_draw_rate"] == pytest.approx(0.045, abs=1e-6)
    assert summary["governance_score"] is not None
    assert summary["program_impact_score"] is not None
    assert summary["organizational_health_score"] is not None
    assert summary["legal_reputation_risk_score"] is not None
    assert isinstance(summary["scenario_stress_tests"], list)
    assert len(summary["scenario_stress_tests"]) >= 2
    assert summary["scenario_worst_case_probability"] >= summary["final_risk_probability"]
    assert isinstance(summary["plain_language_explanation"], list)
    assert len(summary["plain_language_explanation"]) >= 1
