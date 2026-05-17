import csv
import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from tony.ingest import run as ingest_run
from tony.score import score_risk_adjustable

# Real data file at workspace root
ME_GRANTS_CSV = Path(__file__).parent.parent.parent / "ME_grants.csv"


def test_ingest_csv_normalizes_records(sample_csv: Path, tmp_path: Path) -> None:
    out_file = tmp_path / "normalized.json"
    payload = ingest_run(str(sample_csv), None, [], str(out_file))

    assert out_file.exists()
    assert payload["metadata"]["record_count"] == 2
    assert payload["records"][1]["unrestricted_net_assets"] == 690000.0


def test_ingest_csv_writes_valid_json(sample_csv: Path, tmp_path: Path) -> None:
    """Ingested JSON must be parseable and contain the right schema keys."""
    out_file = tmp_path / "out.json"
    ingest_run(str(sample_csv), None, [], str(out_file))

    stored = json.loads(out_file.read_text(encoding="utf-8"))
    assert "metadata" in stored
    assert "records" in stored
    for rec in stored["records"]:
        for key in ("year", "revenue", "expenses", "assets", "liabilities"):
            assert key in rec, f"Missing key '{key}' in record {rec}"


def test_ingest_then_score_produces_descriptor(normalized_payload: Path, tmp_path: Path) -> None:
    """Full ingest → score pipeline returns a valid ContinuityDescriptor."""
    import json as _json
    data = _json.loads(normalized_payload.read_text(encoding="utf-8"))
    result = score_risk_adjustable(
        data, entity_type="nonprofit", horizon=12, continuity_low=6.0, continuity_moderate=3.0
    )

    valid_descriptors = {
        "Low Risk (Excellent)",
        "Moderate Risk (Watch)",
        "High Risk (Critical)",
        "Unknown",
    }
    assert result["ContinuityDescriptor"] in valid_descriptors
    assert 0.0 <= result["ModelRiskProbability"] <= 1.0
    assert isinstance(result["WeightedHealthScore"], float)


def test_ingest_extracts_compensation_and_expense_ratios(tmp_path: Path) -> None:
    source = tmp_path / "form990_like.csv"
    source.write_text(
        "year,revenue,expenses,assets,liabilities,unrestricted_net_assets,executive_compensation,staff_salaries,admin_salaries\n"
        "2023,1500000,1200000,1800000,700000,1100000,180000,420000,90000\n",
        encoding="utf-8",
    )

    out_file = tmp_path / "normalized_comp.json"
    payload = ingest_run(str(source), None, [], str(out_file))
    row = payload["records"][0]

    assert row["executive_compensation"] == 180000.0
    assert row["staff_salaries"] == 420000.0
    assert row["admin_salaries"] == 90000.0
    assert row["total_salaries"] == 690000.0
    assert row["executive_salary_ratio"] == pytest.approx(0.15, abs=1e-6)
    assert row["staff_salary_ratio"] == pytest.approx(0.35, abs=1e-6)
    assert row["admin_salary_ratio"] == pytest.approx(0.075, abs=1e-6)
    assert row["salaries_to_expense_ratio"] == pytest.approx(0.575, abs=1e-6)


def test_ingest_does_not_infer_total_salaries_from_exec_only(tmp_path: Path) -> None:
    source = tmp_path / "form990_exec_only.csv"
    source.write_text(
        "year,revenue,expenses,assets,liabilities,unrestricted_net_assets,executive_compensation\n"
        "2023,1000000,900000,1400000,500000,900000,120000\n",
        encoding="utf-8",
    )

    out_file = tmp_path / "normalized_exec_only.json"
    payload = ingest_run(str(source), None, [], str(out_file))
    row = payload["records"][0]

    assert row["executive_compensation"] == 120000.0
    assert row["total_salaries"] is None
    assert row["salaries_to_expense_ratio"] is None


@pytest.mark.skipif(
    not ME_GRANTS_CSV.exists() or ME_GRANTS_CSV.stat().st_size == 0,
    reason="ME_grants.csv not found or empty",
)
def test_me_grants_csv_loads_all_rows() -> None:
    """ME_grants.csv is parseable with all expected columns."""
    df = pd.read_csv(ME_GRANTS_CSV)
    required = {"title", "agency", "deadline", "amount"}
    assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"
    assert len(df) > 0, "ME_grants.csv has no data rows"


@pytest.mark.skipif(
    not ME_GRANTS_CSV.exists() or ME_GRANTS_CSV.stat().st_size == 0,
    reason="ME_grants.csv not found or empty",
)
def test_me_grants_amounts_are_positive() -> None:
    """Every grant amount in ME_grants.csv is a positive number."""
    df = pd.read_csv(ME_GRANTS_CSV)
    assert (df["amount"] > 0).all(), "Some grant amounts are not positive"


@pytest.mark.skipif(
    not ME_GRANTS_CSV.exists() or ME_GRANTS_CSV.stat().st_size == 0,
    reason="ME_grants.csv not found or empty",
)
def test_me_grants_deadlines_are_strings() -> None:
    """All deadline values are non-empty strings."""
    df = pd.read_csv(ME_GRANTS_CSV)
    assert df["deadline"].notna().all(), "Some deadline values are missing"
    assert (df["deadline"].str.len() > 0).all(), "Some deadline values are empty strings"


@pytest.mark.skipif(
    not ME_GRANTS_CSV.exists() or ME_GRANTS_CSV.stat().st_size == 0,
    reason="ME_grants.csv not found or empty",
)
def test_me_grants_roundtrip_to_tmp_csv(tmp_path: Path) -> None:
    """ME_grants.csv can be copied, re-read, and produces identical data."""
    import shutil
    dest = tmp_path / "me_grants_copy.csv"
    shutil.copy(ME_GRANTS_CSV, dest)

    original = pd.read_csv(ME_GRANTS_CSV)
    copy = pd.read_csv(dest)

    pd.testing.assert_frame_equal(original, copy)


def test_propublica_ingest_requires_ein(tmp_path: Path) -> None:
    out_file = tmp_path / "missing_ein.json"
    with pytest.raises(ValueError, match="EIN is required"):
        ingest_run("propublica", None, [], str(out_file))


def test_propublica_ingest_surfaces_http_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out_file = tmp_path / "http_error.json"

    class _FakeResponse:
        def raise_for_status(self) -> None:
            raise requests.HTTPError("503 Service Unavailable")

        def json(self) -> dict:
            return {}

    def _fake_get(*args, **kwargs):  # noqa: ANN002, ANN003
        return _FakeResponse()

    monkeypatch.setattr("tony.ingest.requests.get", _fake_get)

    with pytest.raises(requests.HTTPError, match="503"):
        ingest_run("propublica", "530196605", [], str(out_file))


def test_propublica_ingest_enriches_external_signals(monkeypatch: pytest.MonkeyPatch, propublica_payload: dict[str, object], tmp_path: Path) -> None:
    out_file = tmp_path / "propublica_enriched.json"

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return propublica_payload

    def _fake_get(*args, **kwargs):  # noqa: ANN002, ANN003
        return _FakeResponse()

    monkeypatch.setattr("tony.ingest.requests.get", _fake_get)

    payload = ingest_run("propublica", "530196605", [], str(out_file))
    metadata = payload["metadata"]
    assert "irs_teos_status" in metadata
    assert "irs_teos_status_risk" in metadata
    assert "charity_navigator_score" in metadata


def test_propublica_ingest_does_not_use_program_revenue_as_program_expenses(
    monkeypatch: pytest.MonkeyPatch,
    propublica_payload: dict[str, object],
    tmp_path: Path,
) -> None:
    payload = dict(propublica_payload)
    payload["filings_with_data"] = [
        {
            "tax_prd_yr": 2022,
            "totrevenue": 1000000,
            "totfuncexpns": 800000,
            "programservicerevenue": 700000,
        }
    ]

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    def _fake_get(*args, **kwargs):  # noqa: ANN002, ANN003
        return _FakeResponse()

    monkeypatch.setattr("tony.ingest.requests.get", _fake_get)

    out_file = tmp_path / "propublica_program_expenses.json"
    normalized = ingest_run("propublica", "530196605", [], str(out_file))

    assert normalized["records"][0]["program_expenses"] is None


def test_file_ingest_enriches_external_signals_when_ein_provided(sample_csv: Path, tmp_path: Path) -> None:
    out_file = tmp_path / "normalized_external.json"
    payload = ingest_run(str(sample_csv), "530196605", [], str(out_file))
    metadata = payload["metadata"]

    assert metadata["irs_teos_status_risk"] == 0.0
    assert metadata["charity_navigator_score"] == pytest.approx(0.99, abs=1e-6)

