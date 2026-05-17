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
