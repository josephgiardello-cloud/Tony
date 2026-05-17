# Scoring Methodology v1.0

## Purpose
This document defines the TONY risk-scoring methodology for nonprofit grant risk assessment with transparent assumptions and reproducible evaluation.

## High-Level Pipeline
1. Ingest and normalize financial and external fields.
2. Build core features:
   - continuity_months
   - operating_margin
   - program_expense_ratio
   - liabilities_to_assets
   - revenue_volatility
3. Generate base risk probability using logistic regression.
4. Apply optional calibration curve to base probability.
5. Apply adjustment layer using configurable penalties/reliefs.
6. Blend final risk probability with weighted health score to form final risk index.

## Feature Definitions
- continuity_months: unrestricted_net_assets / (expenses / 12), capped to [-24, 120] for model stability.
- operating_margin: (revenue - expenses) / revenue.
- program_expense_ratio: program_expenses / expenses.
- liabilities_to_assets: liabilities / assets.
- revenue_volatility: rolling std of log revenue change (fallback to absolute pct change).

## Base Model
- Model: sklearn Pipeline(SimpleImputer(median), StandardScaler, LogisticRegression).
- Label generation for internal training uses risk-point heuristic thresholds.
- Optional reference profiles can be merged into training.

## Adjustment Layer
Configurable scoring presets (conservative, balanced, lenient) weight:
- Penalties: benchmark_gap, confidence_penalty, donor_penalty, cashflow_penalty, compliance_penalty, irs_penalty, altman_penalty.
- Relief: trend_relief, charity_relief.

Adjusted probability is clipped to [0, 1].

## Altman Z'' Integration
Computed when sufficient fields are available:
- X1: Working Capital / Assets
- X2: Retained Earnings / Assets (fallback to book equity proxy)
- X3: EBIT / Assets (fallback EBIT = revenue - expenses)
- X4: Book Equity / Liabilities
- Z'': 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4

Default zones:
- Safe: Z'' > 2.6
- Grey: 1.1 <= Z'' <= 2.6
- Distress: Z'' < 1.1

## Peer Benchmarking
Peer score supports:
- percentile_only
- zscore_only
- blended

Dynamic peer grouping can filter reference profiles by configured keys:
- size_band (derived from assets)
- ntee_code
- state

If filtered peer group has fewer than min_group_rows, full reference pool fallback is used.

## Explainability
Outputs include:
- Feature contributions in model logit space.
- SHAP-compatible linear terms for logistic model:
  - shap_base_logit
  - shap_linear_logit_values
  - shap_total_logit
  - shap_probability_from_logit

## Evaluation Protocol
Use the evaluate command with out-of-sample validation.

Supported methods:
- holdout: train/test split (default 70/30), optional stratification.
- kfold: k-fold or stratified k-fold cross validation.

Reported metrics:
- Brier score
- ROC AUC
- Log loss
- Expected calibration error (ECE)
- Accuracy at threshold 0.5
- No-skill baseline comparison
- In-sample and out-of-sample metrics

Important rule: calibration for out-of-sample reporting is fit only on the training partition/folds and applied to the corresponding test partition.

## Data and Assumption Notes
- Small benchmark datasets can inflate apparent performance.
- Field-comparison claims should use larger, representative datasets with real outcomes.
- Outcome label definitions should be documented and stable (for example distress event vs survived horizon).

## Versioning
- Version: v1.0
- Updated: 2026-05-17
