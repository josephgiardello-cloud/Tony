import io
import json
from pathlib import Path

from tony.dashboard import create_app
from tony.score import run as score_run


def _scored_fixture(normalized_payload: Path, tmp_path: Path) -> Path:
    out_file = tmp_path / "scored_fixture.json"
    score_run(str(normalized_payload), "nonprofit", 12, str(out_file))
    return out_file


def test_dashboard_get_renders_page(normalized_payload: Path, tmp_path: Path) -> None:
    scored_path = _scored_fixture(normalized_payload, tmp_path)
    app = create_app(str(scored_path))

    with app.test_client() as client:
        response = client.get("/")

    assert response.status_code == 200
    assert b"TONY Dashboard" in response.data


def test_dashboard_post_upload_updates_output(
    normalized_payload: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    scored_path = _scored_fixture(normalized_payload, tmp_path)
    app = create_app(str(scored_path))

    def fake_ingest_run(source, ein, years, out_file, config_path=None):  # noqa: ANN001, ANN201
        payload = {"metadata": {"source": source}, "records": [{"year": 2023, "revenue": 1, "expenses": 1}]}
        Path(out_file).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fake_score_run(input_file, entity_type, horizon, out_file, config_path=None):  # noqa: ANN001, ANN201
        payload = {
            "metadata": {"source": "upload"},
            "summary": {
                "descriptor": "Moderate Risk (Acceptable)",
                "continuity_months": 4.2,
                "risk_probability": 0.41,
            },
            "history": [
                {
                    "year": 2023,
                    "continuity_months": 4.2,
                    "operating_margin": 0.1,
                    "program_expense_ratio": 0.7,
                    "liabilities_to_assets": 0.2,
                    "revenue_volatility": 0.0,
                    "risk_probability": 0.41,
                }
            ],
        }
        Path(out_file).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("tony.dashboard.ingest.run", fake_ingest_run)
    monkeypatch.setattr("tony.dashboard.score.run", fake_score_run)

    with app.test_client() as client:
        response = client.post(
            "/",
            data={
                "action": "upload",
                "entity_type": "nonprofit",
                "horizon": "12",
                "data_file": (io.BytesIO(b"year,revenue,expenses\n2023,1,1\n"), "input.csv"),
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    assert b"Dashboard updated with fresh scoring output." in response.data
    assert b"Moderate Risk (Acceptable)" in response.data


def test_dashboard_post_propublica_requires_ein(normalized_payload: Path, tmp_path: Path) -> None:
    scored_path = _scored_fixture(normalized_payload, tmp_path)
    app = create_app(str(scored_path))

    with app.test_client() as client:
        response = client.post(
            "/",
            data={
                "action": "propublica",
                "ein": "",
                "years": "2022,2023",
                "entity_type": "nonprofit",
                "horizon": "12",
            },
        )

    assert response.status_code == 200
    assert b"EIN is required for ProPublica lookup." in response.data


def test_dashboard_post_calibration_updates_metrics(
    normalized_payload: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    scored_path = _scored_fixture(normalized_payload, tmp_path)
    app = create_app(str(scored_path))

    def fake_calibration_run(input_file, out_file, bins=10):  # noqa: ANN001, ANN201
        payload = {
            "metrics": {
                "rows": 4,
                "brier_before": 0.2123,
                "brier_after": 0.1876,
                "auc_before": 0.71,
                "auc_after": 0.73,
            },
            "curve": [
                {
                    "mean_pred": 0.2,
                    "observed_rate": 0.25,
                    "calibrated_mean": 0.24,
                    "n": 2,
                },
                {
                    "mean_pred": 0.7,
                    "observed_rate": 0.75,
                    "calibrated_mean": 0.74,
                    "n": 2,
                },
            ],
            "samples": [],
        }
        Path(out_file).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("tony.dashboard.calibration.run", fake_calibration_run)

    with app.test_client() as client:
        response = client.post(
            "/",
            data={
                "action": "calibrate",
                "calibration_file": (io.BytesIO(b"risk_probability,outcome\n0.2,0\n0.8,1\n"), "bench.csv"),
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    assert b"Calibration completed with external benchmark data." in response.data
    assert b"Calibration Results" in response.data
    assert b"0.1876" in response.data


def test_dashboard_post_compliance_updates_results(
    normalized_payload: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    scored_path = _scored_fixture(normalized_payload, tmp_path)
    app = create_app(str(scored_path))

    def fake_compliance_run(input_file, out_file):  # noqa: ANN001, ANN201
        payload = {
            "overall_score": 63.6,
            "controls_total": 11,
            "controls_met": 7,
            "severity_missing": {"high": 2, "medium": 2, "low": 0},
            "domain_summary": {
                "governance": {"met": 1, "missing": 1},
                "fiduciary": {"met": 0, "missing": 2},
            },
            "priority_gaps": [
                {
                    "id": "FID-001",
                    "domain": "fiduciary",
                    "severity": "high",
                    "requirement": "Restricted funds tracking and spend controls",
                }
            ],
        }
        Path(out_file).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("tony.dashboard.compliance.run", fake_compliance_run)

    with app.test_client() as client:
        response = client.post(
            "/",
            data={
                "action": "compliance",
                "compliance_file": (io.BytesIO(b"{}"), "profile.json"),
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    assert b"Compliance gap assessment completed." in response.data
    assert b"Compliance Gap Results" in response.data
    assert b"FID-001" in response.data


def test_dashboard_basic_mode_shows_grant_standard_metrics(normalized_payload: Path, tmp_path: Path) -> None:
    scored_path = _scored_fixture(normalized_payload, tmp_path)
    app = create_app(str(scored_path))

    with app.test_client() as client:
        response = client.get("/?mode=basic")

    assert response.status_code == 200
    assert b"Basic Mode" in response.data
    assert b"Grant Recommendation" in response.data
    assert b"Operating Margin" in response.data
    assert b"Program Expense Ratio" in response.data
    assert b"Leverage (Liabilities/Assets)" in response.data
    assert b"Altman Z''" in response.data
    assert b"Revenue Growth (YoY)" in response.data
    assert b"Feature History (Standard)" in response.data
    assert b"Model Diagnostics" not in response.data


def test_dashboard_full_mode_shows_full_diagnostics(normalized_payload: Path, tmp_path: Path) -> None:
    scored_path = _scored_fixture(normalized_payload, tmp_path)
    app = create_app(str(scored_path))

    with app.test_client() as client:
        response = client.get("/?mode=full")

    assert response.status_code == 200
    assert b"Full Mode" in response.data
    assert b"Model Diagnostics" in response.data
    assert b"Compensation Burden" in response.data
