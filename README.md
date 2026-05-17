# TONY

TONY is a nonprofit and grant analysis toolkit centered on the maintained package in [tony-core](tony-core).

## Current project structure

- [tony-core](tony-core): Active package, CLI, dashboard, tests, and release artifacts.
- [ME_grants.csv](ME_grants.csv): Real sample grant dataset used by integration tests.

Legacy prototype trees under `tony/` and local virtual environments are intentionally excluded from the active code path.

## Quick start

```bash
cd tony-core
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest tests -q
```

## Run the CLI

```bash
cd tony-core
tony ingest --source ../ME_grants.csv --out normalized.json
tony score --input normalized.json --out scored.json
tony report --input scored.json --format html --out report.html
tony dashboard --input scored.json
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
