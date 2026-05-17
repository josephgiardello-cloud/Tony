# TONY

TONY is a single Python package for nonprofit and grant workflows: ingest filing data, normalize it into a common ledger, score financial risk with pandas and scikit-learn, generate reports, and launch a dashboard with built-in visualizations.

## Capabilities

- Ingest CSV and Excel sources with pandas.
- Pull nonprofit filings from the ProPublica Nonprofit Explorer API by EIN.
- Parse filing PDFs through `camelot-py` or `tabula-py` when those optional dependencies are installed.
- Derive financial health features and fit a logistic regression model for risk probability.
- Render Markdown, HTML, or JSON reports.
- Serve a Flask + Plotly dashboard for interactive review.
- Override scoring weights, thresholds, and source settings with JSON config files.

## Package layout

```text
tony-core/
  README.md
  requirements.txt
  setup.py
  grant_dashboard.py
  tests/
  tony/
    __init__.py
    __main__.py
    cli.py
    config.py
    dashboard.py
    default_config.json
    ingest.py
    report.py
    score.py
    templates/
```

## Installation

```powershell
Set-Location C:\TONY2\tony-core
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional PDF table parsing:

```powershell
python -m pip install -e .[pdf]
```

## Examples

Normalize a local spreadsheet:

```powershell
tony ingest --source ..\ME_grants.csv --out normalized.json
```

Fetch filings from ProPublica:

```powershell
tony ingest --source propublica --ein 530196605 --years 2021,2022,2023 --out propublica.json
```

Score the normalized ledger:

```powershell
tony score --input normalized.json --entity-type nonprofit --horizon 12 --out scored.json
```

Generate an HTML report:

```powershell
tony report --input scored.json --format html --out report.html
```

Launch the dashboard:

```powershell
tony dashboard --input scored.json --host 127.0.0.1 --port 8000
```

Print the bundled config:

```powershell
tony print-config
```

## Configuration

The default configuration is in `tony/default_config.json`. You can override weights and thresholds with a JSON file like this:

```json
{
  "weights": {
    "continuity_months": 0.4,
    "operating_margin": 0.25,
    "program_expense_ratio": 0.15
  },
  "thresholds": {
    "continuity_low": 8.0,
    "continuity_moderate": 4.0,
    "risk_probability_moderate": 0.35,
    "risk_probability_high": 0.6
  }
}
```

Use it during ingestion or scoring:

```powershell
tony score --input normalized.json --entity-type nonprofit --config custom-config.json --out scored.json
```

## Data model

TONY normalizes source data into a common record schema:

- `year`
- `revenue`
- `expenses`
- `assets`
- `liabilities`
- `unrestricted_net_assets`
- `program_expenses`

Column aliases for common source-specific names such as `tax_prd_yr`, `totrevenue`, and `totfuncexpns` are configured in `tony/default_config.json`.

## Scoring model

The scorer derives these features with pandas:

- `continuity_months`
- `operating_margin`
- `program_expense_ratio`
- `liabilities_to_assets`
- `revenue_volatility`

It then trains a scikit-learn logistic regression model on the observed filing history plus bundled reference profiles. The output includes:

- A numeric risk probability
- A weighted health score
- A continuity descriptor
- Per-year feature history for reports and dashboards

## Use cases

- Finance teams benchmarking nonprofit resilience over multiple filing years.
- Grantmakers triaging applicant spreadsheets before committee review.
- Analysts pulling live filing data from ProPublica into repeatable scoring runs.
- Program staff who need browser-based charts instead of notebook-only analysis.

## Testing

```powershell
python -m pytest tests -q
```

The suite uses pytest fixtures, `tmp_path`, and monkeypatching. There are no hard-coded local paths or live network calls in the tests.