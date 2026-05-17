import logging
import math
import os
import pickle
import json
from hashlib import sha256
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

_MODEL_CACHE: dict[str, Pipeline] = {}


def _descriptor(continuity_months: float | None, risk_probability: float, thresholds: dict[str, float]) -> str:
    if continuity_months is None:
        return "Unknown"
    if continuity_months >= thresholds["continuity_low"] and risk_probability < thresholds["risk_probability_moderate"]:
        return "Low Risk (Excellent)"
    if continuity_months >= thresholds["continuity_moderate"] and risk_probability < thresholds["risk_probability_high"]:
        return "Moderate Risk (Acceptable)"
    return "High Risk (Insufficient)"


def _resolve_thresholds(config: dict[str, Any], entity_type: str | None) -> dict[str, float]:
    thresholds = dict(config.get("thresholds", {}))
    if entity_type:
        entity_thresholds = config.get("entity_thresholds", {}).get(entity_type, {})
        thresholds.update(entity_thresholds)
    return thresholds


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
    # Prevent pathological continuity explosions when expenses approach zero.
    frame["continuity_months"] = frame["continuity_months"].clip(lower=-24, upper=120)
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
    revenue = frame["revenue"].astype(float)
    pct_vol = revenue.pct_change().replace([np.inf, -np.inf], np.nan).abs()
    log_rev = np.log(revenue.where(revenue > 0))
    log_vol = log_rev.diff().rolling(window=3, min_periods=2).std()
    frame["revenue_volatility"] = log_vol.fillna(pct_vol).fillna(0).clip(lower=0, upper=2)
    return frame


def _label_records(frame: pd.DataFrame, thresholds: dict[str, float], config: dict[str, Any]) -> pd.Series:
    rules = config.get("labeling", {})
    points_cutoff = int(rules.get("risk_points_cutoff", 3))
    severe_margin = float(rules.get("severe_margin", -0.1))
    weak_program_ratio = float(rules.get("weak_program_ratio", 0.55))
    high_volatility = float(rules.get("high_revenue_volatility", 0.35))

    continuity = frame["continuity_months"].fillna(0)
    margin = frame["operating_margin"].fillna(0)
    leverage = frame["liabilities_to_assets"].fillna(0)
    program_ratio = frame["program_expense_ratio"].fillna(0)
    volatility = frame["revenue_volatility"].fillna(0)

    points = (
        (continuity < float(thresholds["continuity_moderate"])).astype(int) * 2
        + (continuity < float(thresholds["continuity_low"])).astype(int)
        + (margin < 0).astype(int) * 2
        + (margin < severe_margin).astype(int)
        + (leverage > 0.9).astype(int) * 2
        + (leverage > 1.0).astype(int)
        + (program_ratio < weak_program_ratio).astype(int)
        + (volatility > high_volatility).astype(int)
    )
    return (points >= points_cutoff).astype(int)


def _load_reference_profiles(config: dict[str, Any]) -> pd.DataFrame:
    profiles = list(config.get("model", {}).get("reference_profiles", []))
    profiles_file = config.get("model", {}).get("reference_profiles_file")
    if profiles_file:
        resolved = os.path.abspath(os.path.expanduser(str(profiles_file)))
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Reference profiles file not found: {resolved}")
        suffix = os.path.splitext(resolved)[1].lower()
        if suffix == ".csv":
            external = pd.read_csv(resolved).to_dict(orient="records")
        elif suffix == ".json":
            with open(resolved, encoding="utf-8") as handle:
                raw = json.load(handle)
            if isinstance(raw, list):
                external = raw
            elif isinstance(raw, dict) and isinstance(raw.get("profiles"), list):
                external = raw["profiles"]
            else:
                raise ValueError("Reference profiles JSON must be a list or an object with a 'profiles' list.")
        else:
            raise ValueError("Reference profiles file must be CSV or JSON.")
        profiles.extend(external)

    if not profiles:
        return pd.DataFrame(columns=[*FEATURE_COLUMNS, "risk_label"])

    reference = pd.DataFrame(profiles)
    if "risk_label" not in reference.columns:
        raise ValueError("Reference profiles require a risk_label column.")
    return reference


def _model_cache_key(training: pd.DataFrame, config: dict[str, Any]) -> str:
    stable = training[[*FEATURE_COLUMNS, "risk_label"]].round(6).sort_values(FEATURE_COLUMNS).to_json(orient="records")
    model_opts = {
        "random_state": config["model"].get("random_state", 42),
        "max_iter": config["model"].get("max_iter", 1000),
        "class_weight": config["model"].get("class_weight", "balanced"),
    }
    return sha256((stable + str(model_opts)).encode("utf-8")).hexdigest()


def _load_pretrained_model(config: dict[str, Any]) -> Pipeline | None:
    path = config.get("model", {}).get("pretrained_model_path")
    if not path:
        return None
    resolved = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.exists(resolved):
        return None
    with open(resolved, "rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, Pipeline):
        raise ValueError("Pretrained model must be a sklearn Pipeline.")
    return model


def _persist_model_if_requested(model: Pipeline, config: dict[str, Any]) -> None:
    path = config.get("model", {}).get("pretrained_model_path")
    save_if_missing = bool(config.get("model", {}).get("save_trained_if_missing", False))
    if not path or not save_if_missing:
        return
    resolved = os.path.abspath(os.path.expanduser(str(path)))
    if os.path.exists(resolved):
        return
    parent = os.path.dirname(resolved)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(resolved, "wb") as handle:
        pickle.dump(model, handle)


def _train_model(feature_frame: pd.DataFrame, config: dict[str, Any], thresholds: dict[str, float]) -> Pipeline:
    pretrained = _load_pretrained_model(config)
    if pretrained is not None:
        return pretrained

    training = feature_frame[FEATURE_COLUMNS].copy()
    training["risk_label"] = _label_records(feature_frame, thresholds, config)
    reference = _load_reference_profiles(config)
    if not reference.empty:
        training = pd.concat([training, reference], ignore_index=True, sort=False)

    training = training.dropna(subset=["risk_label"]).copy()
    training["risk_label"] = training["risk_label"].astype(int)

    # Ensure at least two classes are available for logistic regression.
    if training["risk_label"].nunique() < 2:
        fallback = pd.DataFrame(
            [
                {"continuity_months": 2.0, "operating_margin": -0.1, "program_expense_ratio": 0.5, "liabilities_to_assets": 1.0, "revenue_volatility": 0.5, "risk_label": 1},
                {"continuity_months": 9.0, "operating_margin": 0.08, "program_expense_ratio": 0.8, "liabilities_to_assets": 0.35, "revenue_volatility": 0.1, "risk_label": 0},
            ]
        )
        training = pd.concat([training, fallback], ignore_index=True, sort=False)

    cache_enabled = bool(config.get("model", {}).get("cache_models", True))
    cache_key = _model_cache_key(training, config)
    if cache_enabled and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=int(config["model"].get("max_iter", 1000)),
                    class_weight=config["model"].get("class_weight", "balanced"),
                    random_state=int(config["model"].get("random_state", 42)),
                ),
            ),
        ]
    )
    model.fit(training[FEATURE_COLUMNS], training["risk_label"])
    if cache_enabled:
        _MODEL_CACHE[cache_key] = model
    _persist_model_if_requested(model, config)
    return model


def _sigmoid(value: float) -> float:
    bounded = max(min(value, 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-bounded))


def _normalization_settings(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "continuity_months": {"center": 6.0, "scale": 3.0, "low": -24.0, "high": 120.0, "positive": True},
        "operating_margin": {"center": 0.03, "scale": 0.08, "low": -0.5, "high": 0.5, "positive": True},
        "program_expense_ratio": {"center": 0.7, "scale": 0.12, "low": 0.0, "high": 1.0, "positive": True},
        "liabilities_to_assets": {"center": 0.6, "scale": 0.2, "low": 0.0, "high": 2.0, "positive": False},
        "revenue_volatility": {"center": 0.2, "scale": 0.12, "low": 0.0, "high": 2.0, "positive": False},
    }
    override = config.get("normalization", {})
    merged: dict[str, Any] = {}
    for feature, settings in defaults.items():
        merged[feature] = {**settings, **override.get(feature, {})}
    return merged


def _weighted_health_score(latest_row: pd.Series, weights: dict[str, float], config: dict[str, Any]) -> tuple[float, dict[str, float]]:
    settings = _normalization_settings(config)
    normalized: dict[str, float] = {}
    for feature, weight in weights.items():
        value = float(latest_row.get(feature) or 0.0)
        spec = settings.get(feature, {})
        low = float(spec.get("low", value))
        high = float(spec.get("high", value))
        center = float(spec.get("center", 0.0))
        scale = max(float(spec.get("scale", 1.0)), 1e-6)
        positive = bool(spec.get("positive", True))

        clipped = min(max(value, low), high)
        transformed = _sigmoid((clipped - center) / scale)
        normalized[feature] = transformed if positive else 1.0 - transformed

    score = sum(normalized[column] * weight for column, weight in weights.items())
    return round(score, 4), {k: round(v, 4) for k, v in normalized.items()}


def _feature_contributions(model: Pipeline, latest_row: pd.Series) -> dict[str, Any]:
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]

    latest_df = pd.DataFrame([{feature: latest_row.get(feature) for feature in FEATURE_COLUMNS}])
    transformed = scaler.transform(imputer.transform(latest_df))[0]
    coefficients = classifier.coef_[0]
    contributions = {feature: float(transformed[idx] * coefficients[idx]) for idx, feature in enumerate(FEATURE_COLUMNS)}
    ranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    return {
        "feature_contributions": {key: round(value, 5) for key, value in contributions.items()},
        "top_drivers": [
            {
                "feature": name,
                "contribution": round(value, 5),
                "direction": "increases risk" if value > 0 else "reduces risk",
            }
            for name, value in ranked[:3]
        ],
    }


def score_risk_adjustable(
    data: dict[str, Any],
    entity_type: str,
    horizon: int,
    continuity_low: float,
    continuity_moderate: float,
) -> dict[str, Any]:
    config = load_config()
    thresholds = _resolve_thresholds(config, entity_type)
    thresholds["continuity_low"] = continuity_low
    thresholds["continuity_moderate"] = continuity_moderate
    record = {
        "year": datetime.now().year,
        "expenses": data.get("expenses", data.get("total_expenses")),
        **data,
    }
    feature_frame = _feature_frame([record])
    model = _train_model(feature_frame, config, thresholds)
    latest = feature_frame.iloc[-1]
    risk_probability = float(model.predict_proba(feature_frame[FEATURE_COLUMNS])[-1][1])
    continuity_months = round(float(latest["continuity_months"]), 2) if pd.notna(latest["continuity_months"]) else None
    weighted_score, normalized_features = _weighted_health_score(latest, config["weights"], config)
    explanation = _feature_contributions(model, latest)
    return {
        "entity_type": entity_type,
        "horizon": horizon,
        "ContinuityRiskScore": continuity_months,
        "OperatingMargin": round(float(latest["operating_margin"]), 4),
        "ProgramExpenseRatio": round(float(latest["program_expense_ratio"]), 4),
        "LiabilitiesToAssets": round(float(latest["liabilities_to_assets"]), 4),
        "RevenueVolatility": round(float(latest["revenue_volatility"]), 4),
        "ModelRiskProbability": round(risk_probability, 4),
        "WeightedHealthScore": weighted_score,
        "NormalizedFeatures": normalized_features,
        "ContinuityDescriptor": _descriptor(continuity_months, risk_probability, thresholds),
        **explanation,
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
    thresholds = _resolve_thresholds(config, entity_type)
    model = _train_model(feature_frame, config, thresholds)
    feature_frame["risk_probability"] = model.predict_proba(feature_frame[FEATURE_COLUMNS])[:, 1]

    latest = feature_frame.iloc[-1]
    continuity_months = round(float(latest["continuity_months"]), 2) if pd.notna(latest["continuity_months"]) else None
    weighted_score, normalized_features = _weighted_health_score(latest, config["weights"], config)
    explanation = _feature_contributions(model, latest)
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
            "weighted_health_score": weighted_score,
            "normalized_features": normalized_features,
            "descriptor": _descriptor(continuity_months, float(latest["risk_probability"]), thresholds),
            **explanation,
        },
        "history": feature_frame[["year", *FEATURE_COLUMNS, "risk_probability"]].round(4).replace({np.nan: None}).to_dict(orient="records"),
        "scored_at": datetime.now().isoformat(),
    }
    write_json(out_file, result)
    logging.info("Scored %s into %s", entity_type, out_file)
    return result
