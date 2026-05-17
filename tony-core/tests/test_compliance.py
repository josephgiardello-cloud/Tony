import json
from pathlib import Path

from tony.compliance import evaluate, run


def test_evaluate_compliance_identifies_priority_gaps() -> None:
    profile = {
        "governance": {"conflict_policy": True},
        "filings": {"calendar": True},
        "fiduciary": {},
        "third_party": {},
        "policy": {},
        "incident": {},
        "access": {"rbac": True},
    }
    result = evaluate(profile)

    assert result["controls_total"] >= 10
    assert result["controls_met"] < result["controls_total"]
    assert result["severity_missing"]["high"] > 0
    assert len(result["priority_gaps"]) > 0


def test_compliance_run_writes_report(tmp_path: Path) -> None:
    source = tmp_path / "profile.json"
    out = tmp_path / "compliance_report.json"
    source.write_text(json.dumps({"governance": {"conflict_policy": True}}), encoding="utf-8")

    payload = run(str(source), str(out))
    stored = json.loads(out.read_text(encoding="utf-8"))

    assert out.exists()
    assert stored["overall_score"] == payload["overall_score"]
