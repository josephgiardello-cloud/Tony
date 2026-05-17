"""
Real integration tests for GrantApplication and GrantManager.

These tests exercise actual business logic — submit, review, export, status
updates — using a temp directory for file storage so no production files are
touched.  No HTTP or email mocking: notification side-effects are silenced by
overriding notify_dashboard and send_email_notification directly.
"""
import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make tony-env importable even when running from another cwd
sys.path.insert(0, str(Path(__file__).parent))

# Patch heavy side-effect imports before importing the module under test
import unittest.mock as mock

# Stub out modules that are not installed in the test environment
_ML_MOCK = mock.MagicMock()
_ML_MOCK.predict_risk.return_value = 0.35  # must be a serializable float

for _mod in ("accounting", "donor_tracking", "event_management", "volunteer",
             "communications", "yagmail", "babel.support"):
    sys.modules.setdefault(_mod, mock.MagicMock())
sys.modules["ml_risk_model"] = _ML_MOCK

from grant_management import GrantApplication, GrantManager  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ME_GRANTS_CSV = Path(__file__).parent.parent / "ME_grants.csv"

REAL_GRANTS: list[dict] = []
if ME_GRANTS_CSV.exists():
    with ME_GRANTS_CSV.open(newline="", encoding="utf-8") as _f:
        REAL_GRANTS = list(csv.DictReader(_f))


@pytest.fixture()
def mgr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GrantManager:
    """GrantManager backed by a temp JSON file, with noisy side effects silenced."""
    storage = str(tmp_path / "apps.json")
    monkeypatch.chdir(tmp_path)          # audit/alert logs land in tmp
    manager = GrantManager(storage_path=storage)
    # Silence dashboard HTTP calls and email dispatches
    manager.notify_dashboard = lambda msg: None
    manager.send_email_notification = lambda subject, body, to_email: None
    return manager


# ---------------------------------------------------------------------------
# GrantApplication unit tests
# ---------------------------------------------------------------------------

def test_grant_application_initial_state() -> None:
    app = GrantApplication("Acme Nonprofit", "Community Garden", 10_000, "Build a garden")
    assert app.status == "submitted"
    assert app.reviewer is None
    assert app.review_notes == ""
    assert app.risk_score is None
    assert app.decision_date is None
    assert app.documents == []
    assert app.id  # UUID assigned


def test_grant_application_with_documents() -> None:
    app = GrantApplication("Org", "Title", 5_000, "Desc", documents=["budget.pdf"])
    assert app.documents == ["budget.pdf"]


# ---------------------------------------------------------------------------
# GrantManager lifecycle tests
# ---------------------------------------------------------------------------

def test_submit_creates_application_and_persists(mgr: GrantManager, tmp_path: Path) -> None:
    app_id = mgr.submit_application("Community Health ME", "Opioid Recovery", 80_000, "Recovery support")

    assert app_id in mgr.applications
    stored = json.loads((tmp_path / "apps.json").read_text(encoding="utf-8"))
    assert app_id in stored
    assert stored[app_id]["project_title"] == "Opioid Recovery"
    assert stored[app_id]["status"] == "submitted"


def test_submit_sets_risk_score(mgr: GrantManager) -> None:
    app_id = mgr.submit_application("Test Org", "Test Project", 25_000, "Description")
    app = mgr.applications[app_id]
    # ml_risk_model is stubbed → risk score should be "unavailable" (fallback path)
    assert app.risk_score is not None
    assert "MLRiskScore" in app.risk_score


def test_assign_reviewer_changes_status(mgr: GrantManager) -> None:
    app_id = mgr.submit_application("Volunteer ME", "Mentorship", 20_000, "Youth mentorship")

    result = mgr.assign_reviewer(app_id, "Jane Reviewer")

    assert result is True
    app = mgr.applications[app_id]
    assert app.reviewer == "Jane Reviewer"
    assert app.status == "in_review"


def test_assign_reviewer_returns_false_for_missing_id(mgr: GrantManager) -> None:
    assert mgr.assign_reviewer("nonexistent-id", "Anyone") is False


def test_review_approves_application(mgr: GrantManager) -> None:
    app_id = mgr.submit_application("Housing ME", "Affordable Housing", 200_000, "Build units")
    mgr.assign_reviewer(app_id, "Alice")

    result = mgr.review_application(app_id, "Alice", "Looks good", "approved")

    assert result is True
    app = mgr.applications[app_id]
    assert app.status == "approved"
    assert app.review_notes == "Looks good"
    assert app.decision_date is not None


def test_review_rejects_invalid_decision(mgr: GrantManager) -> None:
    app_id = mgr.submit_application("Org", "Project", 1_000, "Desc")
    result = mgr.review_application(app_id, "Bob", "notes", "maybe")
    assert result is False
    assert mgr.applications[app_id].status == "submitted"  # unchanged


def test_update_status_valid(mgr: GrantManager) -> None:
    app_id = mgr.submit_application("Org", "Project", 5_000, "Desc")
    assert mgr.update_status(app_id, "in_review") is True
    assert mgr.applications[app_id].status == "in_review"


def test_update_status_invalid(mgr: GrantManager) -> None:
    app_id = mgr.submit_application("Org", "Project", 5_000, "Desc")
    assert mgr.update_status(app_id, "flying") is False


def test_add_review_notes(mgr: GrantManager) -> None:
    app_id = mgr.submit_application("Library ME", "Reading Program", 18_000, "Literacy")
    assert mgr.add_review_notes(app_id, "Budget seems inflated") is True
    assert mgr.applications[app_id].review_notes == "Budget seems inflated"


def test_upload_supporting_document(mgr: GrantManager) -> None:
    app_id = mgr.submit_application("Coastal ME", "Habitat Restore", 75_000, "Wetlands")
    mgr.upload_supporting_document(app_id, "habitat_plan.pdf")
    assert "habitat_plan.pdf" in mgr.applications[app_id].documents


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

def test_reload_applications_from_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Applications saved by one GrantManager are loaded by a fresh one."""
    storage = str(tmp_path / "apps.json")
    monkeypatch.chdir(tmp_path)

    mgr1 = GrantManager(storage_path=storage)
    mgr1.notify_dashboard = lambda msg: None
    mgr1.send_email_notification = lambda s, b, t: None
    app_id = mgr1.submit_application("Rural ME", "Broadband", 150_000, "Internet access")
    mgr1.assign_reviewer(app_id, "Carol")

    mgr2 = GrantManager(storage_path=storage)
    assert app_id in mgr2.applications
    assert mgr2.applications[app_id].reviewer == "Carol"
    assert mgr2.applications[app_id].status == "in_review"


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------

def test_export_csv_contains_all_applications(mgr: GrantManager, tmp_path: Path) -> None:
    mgr.submit_application("Org A", "Project Alpha", 10_000, "Alpha")
    mgr.submit_application("Org B", "Project Beta", 20_000, "Beta")

    out = str(tmp_path / "export.csv")
    mgr.export_applications(format="csv", out_path=out)

    df = pd.read_csv(out)
    assert len(df) == 2
    titles = set(df["Title"].tolist())
    assert titles == {"Project Alpha", "Project Beta"}


def test_export_reflects_status_updates(mgr: GrantManager, tmp_path: Path) -> None:
    app_id = mgr.submit_application("State Lib ME", "STEM Books", 18_000, "Literacy")
    mgr.assign_reviewer(app_id, "Dan")
    mgr.review_application(app_id, "Dan", "Great proposal", "approved")

    out = str(tmp_path / "export.csv")
    mgr.export_applications(format="csv", out_path=out)

    df = pd.read_csv(out)
    assert df.loc[df["Title"] == "STEM Books", "Status"].iloc[0] == "approved"


# ---------------------------------------------------------------------------
# Real ME_grants.csv integration test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_GRANTS, reason="ME_grants.csv is empty or missing")
def test_bulk_submit_from_me_grants_csv(mgr: GrantManager, tmp_path: Path) -> None:
    """Submit every row in ME_grants.csv and verify all are persisted and exportable."""
    submitted_ids: list[str] = []
    for row in REAL_GRANTS:
        try:
            amount = float(row["amount"])
        except (ValueError, KeyError):
            amount = 0.0
        app_id = mgr.submit_application(
            applicant_name=row.get("agency", "Unknown Agency"),
            project_title=row.get("title", "Untitled"),
            amount_requested=amount,
            description=f"Deadline: {row.get('deadline', 'N/A')}",
        )
        submitted_ids.append(app_id)

    # All rows stored
    assert len(mgr.applications) == len(REAL_GRANTS)

    # Export to CSV and validate row count
    out = str(tmp_path / "me_grants_export.csv")
    mgr.export_applications(format="csv", out_path=out)
    df = pd.read_csv(out)
    assert len(df) == len(REAL_GRANTS)

    # Spot-check: all amounts are positive
    assert (df["Amount"] > 0).all()


@pytest.mark.skipif(not REAL_GRANTS, reason="ME_grants.csv is empty or missing")
def test_full_lifecycle_on_real_grants(mgr: GrantManager) -> None:
    """Run the full submit → review → approve lifecycle on the first 3 ME grants."""
    for row in REAL_GRANTS[:3]:
        app_id = mgr.submit_application(
            applicant_name=row.get("agency", "Agency"),
            project_title=row.get("title", "Title"),
            amount_requested=float(row.get("amount", 0)),
            description=f"Deadline: {row.get('deadline', 'N/A')}",
        )
        mgr.assign_reviewer(app_id, "State Reviewer")
        result = mgr.review_application(app_id, "State Reviewer", "Reviewed", "approved")
        assert result is True
        assert mgr.applications[app_id].status == "approved"
