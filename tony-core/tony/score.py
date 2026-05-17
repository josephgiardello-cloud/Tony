import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import load_config
from .utils import read_json, write_json

FEATURE_COLUMNS = [
    "continuity_months",
    "operating_margin",
    "program_expense_ratio",
    "liabilities_to_assets",
    "revenue_volatility",
]


def _descriptor(continuity_months: float | None, risk_probability: float, thresholds: dict[str, float]) -> str:
    if continuity_months is None:
        return "Unknown"
    if continuity_months >= thresholds["continuity_low"] and risk_probability < thresholds["risk_probability_moderate"]:
        return "Low Risk (Excellent)"
    if continuity_months >= thresholds["continuity_moderate"] and risk_probability < thresholds["risk_probability_high"]:
        return "Moderate Risk (Acceptable)"
    return "High Risk (Insufficient)"


def _feature_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        raise ValueError("No records available for scoring.")

    frame = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    if frame.empty:
        raise ValueError("No records available for scoring.")

    for column in [
        "revenue",
        "expenses",
        "program_expenses",
        "assets",
        "liabilities",
        "unrestricted_net_assets",
    ]:
        if column not in frame.columns:
            frame[column] = np.nan

    frame["continuity_months"] = np.where(
        frame["expenses"].fillna(0) > 0,
        frame["unrestricted_net_assets"].fillna(0) / (frame["expenses"].replace(0, np.nan) / 12),
        np.nan,
    )
    frame["operating_margin"] = np.where(
        frame["revenue"].fillna(0) > 0,
        (frame["revenue"].fillna(0) - frame["expenses"].fillna(0)) / frame["revenue"].replace(0, np.nan),
        0.0,
    )
    frame["program_expense_ratio"] = np.where(
        frame["expenses"].fillna(0) > 0,
        frame["program_expenses"].fillna(frame["expenses"] * 0.75) / frame["expenses"].replace(0, np.nan),
        np.nan,
    )
    frame["liabilities_to_assets"] = np.where(
        frame["assets"].fillna(0) > 0,
        frame["liabilities"].fillna(0) / frame["assets"].replace(0, np.nan),
        0.0,
    )
    frame["revenue_volatility"] = (
        frame["revenue"].fillna(0).pct_change().replace([np.inf, -np.inf], np.nan).abs().fillna(0)
    )
    return frame


def _label_records(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    return (
        (frame["continuity_months"].fillna(0) < thresholds["continuity_moderate"])
        | (frame["operating_margin"].fillna(0) < 0)
        | (frame["liabilities_to_assets"].fillna(0) > 0.9)
    ).astype(int)


def _train_model(feature_frame: pd.DataFrame, config: dict[str, Any]) -> Pipeline:
    training = feature_frame[FEATURE_COLUMNS].copy()
    training["risk_label"] = _label_records(feature_frame, config["thresholds"])
    reference = pd.DataFrame(config["model"]["reference_profiles"])
    training = pd.concat([training, reference], ignore_index=True, sort=False)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=config["model"]["random_state"],
                ),
            ),
        ]
    )
    model.fit(training[FEATURE_COLUMNS], training["risk_label"])
    return model


def _weighted_health_score(latest_row: pd.Series, weights: dict[str, float]) -> float:
    normalized = {
        "continuity_months": min(float(latest_row.get("continuity_months") or 0.0) / 12.0, 1.0),
        "operating_margin": min(max((float(latest_row.get("operating_margin") or 0.0) + 0.2) / 0.4, 0.0), 1.0),
        "program_expense_ratio": min(max(float(latest_row.get("program_expense_ratio") or 0.0), 0.0), 1.0),
        "liabilities_to_assets": 1.0 - min(max(float(latest_row.get("liabilities_to_assets") or 0.0), 0.0), 1.0),
        "revenue_volatility": 1.0 - min(max(float(latest_row.get("revenue_volatility") or 0.0), 0.0), 1.0),
    }
    score = sum(normalized[column] * weight for column, weight in weights.items())
    return round(score, 4)


def score_risk_adjustable(
    data: dict[str, Any],
    entity_type: str,
    horizon: int,
    continuity_low: float,
    continuity_moderate: float,
) -> dict[str, Any]:
    config = load_config()
    config["thresholds"]["continuity_low"] = continuity_low
    config["thresholds"]["continuity_moderate"] = continuity_moderate
    record = {
        "year": datetime.now().year,
        "expenses": data.get("expenses", data.get("total_expenses")),
        **data,
    }
    feature_frame = _feature_frame([record])
    model = _train_model(feature_frame, config)
    latest = feature_frame.iloc[-1]
    risk_probability = float(model.predict_proba(feature_frame[FEATURE_COLUMNS])[-1][1])
    continuity_months = round(float(latest["continuity_months"]), 2) if pd.notna(latest["continuity_months"]) else None
    return {
        "entity_type": entity_type,
        "horizon": horizon,
        "ContinuityRiskScore": continuity_months,
        "OperatingMargin": round(float(latest["operating_margin"]), 4),
        "ProgramExpenseRatio": round(float(latest["program_expense_ratio"]), 4),
        "LiabilitiesToAssets": round(float(latest["liabilities_to_assets"]), 4),
        "RevenueVolatility": round(float(latest["revenue_volatility"]), 4),
        "ModelRiskProbability": round(risk_probability, 4),
        "WeightedHealthScore": _weighted_health_score(latest, config["weights"]),
        "ContinuityDescriptor": _descriptor(continuity_months, risk_probability, config["thresholds"]),
        "scored_at": datetime.now().isoformat(),
    }


def run(
    input_file: str,
    entity_type: str,
    horizon: int,
    out_file: str,
    config_path: str | None = None,
) -> dict[str, Any]:
    payload = read_json(input_file)
    feature_frame = _feature_frame(payload.get("records", []))
    config = load_config(config_path)
    model = _train_model(feature_frame, config)
    feature_frame["risk_probability"] = model.predict_proba(feature_frame[FEATURE_COLUMNS])[:, 1]

    latest = feature_frame.iloc[-1]
    continuity_months = round(float(latest["continuity_months"]), 2) if pd.notna(latest["continuity_months"]) else None
    result = {
        "entity_type": entity_type,
        "horizon": horizon,
        "metadata": payload.get("metadata", {}),
        "summary": {
            "continuity_months": continuity_months,
            "operating_margin": round(float(latest["operating_margin"]), 4),
            "program_expense_ratio": round(float(latest["program_expense_ratio"]), 4),
            "liabilities_to_assets": round(float(latest["liabilities_to_assets"]), 4),
            "revenue_volatility": round(float(latest["revenue_volatility"]), 4),
            "risk_probability": round(float(latest["risk_probability"]), 4),
            "weighted_health_score": _weighted_health_score(latest, config["weights"]),
            "descriptor": _descriptor(continuity_months, float(latest["risk_probability"]), config["thresholds"]),
        },
        "history": feature_frame[["year", *FEATURE_COLUMNS, "risk_probability"]].round(4).replace({np.nan: None}).to_dict(orient="records"),
        "scored_at": datetime.now().isoformat(),
    }
    write_json(out_file, result)
    logging.info("Scored %s into %s", entity_type, out_file)
    return result
