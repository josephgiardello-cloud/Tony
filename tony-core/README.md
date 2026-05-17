# TONY

TONY is a single Python package for nonprofit and grant workflows: ingest filing data, normalize it into a common ledger, score financial risk with pandas and scikit-learn, generate reports, and launch a dashboard with built-in visualizations.

## Capabilities

- Ingest CSV and Excel sources with pandas.
- Pull nonprofit filings from the ProPublica Nonprofit Explorer API by EIN.
- Parse filing PDFs through `camelot-py` or `tabula-py` when those optional dependencies are installed.
- Derive financial health features and fit a logistic regression model for risk probability.
- Render Markdown, HTML, or JSON reports.
- Serve a Flask + Plotly dashboard for interactive review.
- Calibrate model probabilities against external benchmark outcomes.
- Run governance/fiduciary/compliance gap audits from structured control profiles.
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

```bash
cd tony-core
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional PDF table parsing:

```bash
python -m pip install -e .[pdf]
```

## Examples

Normalize a local spreadsheet:

```bash
tony ingest --source ../ME_grants.csv --out normalized.json
```

Fetch filings from ProPublica:

```bash
tony ingest --source propublica --ein 530196605 --years 2021,2022,2023 --out propublica.json
```

If `--ein` is omitted for `source=propublica`, the CLI uses `PROPUBLICA_EIN` or `TONY_EIN` when set.

Score the normalized ledger:

```bash
tony score --input normalized.json --entity-type nonprofit --horizon 12 --out scored.json
```

`--entity-type` defaults to `nonprofit` when omitted.

Generate an HTML report:

```bash
tony report --input scored.json --format html --out report.html
```

Launch the dashboard:

```bash
tony dashboard --input scored.json --host 127.0.0.1 --port 8000
```

Calibrate with external outcomes (CSV requires `risk_probability` and `outcome` columns):

```bash
tony calibrate --input benchmark_labels.csv --bins 10 --out calibration.json
```

Run a compliance gap audit (JSON profile input):

```bash
tony compliance-audit --input compliance_profile.json --out compliance_report.json
```

Print the bundled config:

```bash
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

```bash
tony score --input normalized.json --entity-type nonprofit --config custom-config.json --out scored.json
```

You can also set `TONY_CONFIG=/path/to/config.json` and skip `--config`.

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

## External baseline artifacts

The repository includes baseline files sourced from public external references:

- `baselines/external_calibration_baseline.csv`
: Seed dataset built from IRS bulk datasets (Pub 78 active records and Automatic Revocation list), plus a Charity Navigator public rating fact for contrast.
- `baselines/external_compliance_controls.json`
: Control baseline mapped to public sources (IRS filing requirements, IRS TEOS bulk datasets, OFAC sanctions resources, Charity Navigator accountability indicators).
- `baselines/compliance_profile_template.json`
: Ready-to-upload compliance profile template for dashboard/CLI audits.

## Scoring model

The scorer derives these features with pandas:

- `continuity_months`
- `operating_margin`
- `program_expense_ratio`
- `liabilities_to_assets`
- `revenue_volatility`

It then trains a scikit-learn logistic regression model on observed filing history plus bundled reference profiles (or an external reference profile file, if configured). The output includes:

- A numeric risk probability
- A weighted health score
- A continuity descriptor
- Normalized feature values used for weighted scoring
- Feature contribution diagnostics and top risk drivers
- Per-year feature history for reports and dashboards

### Model robustness controls

- Entity-specific thresholds via `entity_thresholds` (for example `hospital`, `foundation`, `nonprofit_small`).
- Configurable labeling rules under `labeling`.
- Optional external reference profile source via `model.reference_profiles_file` (CSV or JSON).
- Optional pre-trained model loading via `model.pretrained_model_path`.
- In-process model caching via `model.cache_models` to avoid retraining on identical data.

### Weighted score normalization

Weighted score inputs are normalized using bounded sigmoid transforms driven by `normalization` settings in `tony/default_config.json`.
Each feature has explicit `center`, `scale`, and winsorization-style `low`/`high` bounds to avoid unstable outputs from extreme values.

## Use cases

- Finance teams benchmarking nonprofit resilience over multiple filing years.
- Grantmakers triaging applicant spreadsheets before committee review.
- Analysts pulling live filing data from ProPublica into repeatable scoring runs.
- Program staff who need browser-based charts instead of notebook-only analysis.

## Testing

```bash
python -m pytest tests -q
```

Test coverage includes:

- Ingest normalization from CSV/Excel fixtures.
- Error handling for missing EIN and upstream ProPublica HTTP errors.
- Scoring for normal inputs, sparse inputs, and empty-record rejection.
- Probability calibration quality against external labels.
- CLI behavior (round-trip ingest/score plus env-driven defaults).
- Dashboard interactivity for upload, ProPublica fetch, and calibration upload.
- Dashboard compliance gap review (governance, filings, fiduciary, third-party, policy, incident, access).
- Real-data validation of `../ME_grants.csv` when present.

The suite uses pytest fixtures, `tmp_path`, and monkeypatching. No test requires a live network call.