import json
import sys
import subprocess
sys.path.append(r"c:/TONY2/tony-env")
from tony import score_risk_adjustable

def test_score_risk_adjustable():
    data = {
        "unrestricted_net_assets": 120000,
        "total_expenses": 240000
    }
    result = score_risk_adjustable(data, "hospital", 12, 6.0, 3.0)
    assert "ContinuityDescriptor" in result
    assert result["ContinuityDescriptor"] == "Low Risk (Excellent)"

def test_cli_score(tmp_path):
    in_file = tmp_path / "in.json"
    out_file = tmp_path / "out.json"
    with open(in_file, "w") as f:
        json.dump({"unrestricted_net_assets": 120000, "total_expenses": 240000}, f)
    cmd = [sys.executable, "c:/TONY2/tony-env/tony.py", "--input", str(in_file), "--entity-type", "hospital", "--horizon", "12", "--out", str(out_file)]
    result = subprocess.run(cmd, capture_output=True)
    assert out_file.exists()
    with open(out_file) as f:
        scored = json.load(f)
    assert "ContinuityDescriptor" in scored
