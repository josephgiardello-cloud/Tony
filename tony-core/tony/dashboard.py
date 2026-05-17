import json

import pandas as pd
import plotly.express as px
from flask import Flask, render_template

from .utils import read_json


def _build_charts(payload: dict) -> dict[str, str]:
    history = pd.DataFrame(payload.get("history", []))
    if history.empty:
        return {"trend": "{}", "mix": "{}", "risk": "{}"}

    trend = px.line(
        history,
        x="year",
        y=["continuity_months", "operating_margin", "program_expense_ratio"],
        markers=True,
        title="Financial resilience trends",
    )
    mix = px.bar(
        history,
        x="year",
        y=["liabilities_to_assets", "revenue_volatility"],
        barmode="group",
        title="Balance sheet pressure",
    )
    risk = px.area(history, x="year", y="risk_probability", title="Model risk probability")
    for figure in (trend, mix, risk):
        figure.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=60, b=20))
    return {
        "trend": trend.to_json(),
        "mix": mix.to_json(),
        "risk": risk.to_json(),
    }


def create_app(data_path: str) -> Flask:
    app = Flask(__name__, template_folder="templates")

    @app.route("/")
    def index() -> str:
        payload = read_json(data_path)
        charts = _build_charts(payload)
        return render_template(
            "dashboard.html",
            summary=payload.get("summary", {}),
            metadata=payload.get("metadata", {}),
            history=json.dumps(payload.get("history", [])),
            charts=charts,
        )

    return app


def main(data_path: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    app = create_app(data_path)
    app.run(host=host, port=port, debug=True)