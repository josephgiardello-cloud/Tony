import json
import sys
import subprocess
sys.path.append(r"c:/TONY2/tony-env")

def test_cli_ingest(tmp_path):
    out_file = tmp_path / "out.json"
    python_exe = r"c:/TONY2/tony-env/Scripts/python.exe"
    cmd = [python_exe, "c:/TONY2/tony-env/tony.py", "--command", "fetch", "--input", "990", "--entity-type", "hospital", "--horizon", "12", "--out", str(out_file)]
    result = subprocess.run(cmd, capture_output=True)
    if not out_file.exists():
        print("STDOUT:\n", result.stdout.decode())
        print("STDERR:\n", result.stderr.decode())
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert "ContinuityDescriptor" in data
