import json
import os
import tempfile

import pandas as pd
import plotly.express as px
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from . import ingest, score
from .utils import parse_years
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


def _empty_payload() -> dict:
    return {
        "summary": {},
        "metadata": {},
        "history": [],
    }


def _load_payload(data_path: str) -> dict:
    try:
        payload = read_json(data_path)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return _empty_payload()


def create_app(data_path: str) -> Flask:
    app = Flask(__name__, template_folder="templates")

    @app.route("/", methods=["GET", "POST"])
    def index() -> str:
        payload = _load_payload(data_path)
        error_message = ""
        status_message = ""
        form_state = {
            "entity_type": "nonprofit",
            "horizon": "12",
            "ein": "",
            "years": "",
        }

        if request.method == "POST":
            form_state["entity_type"] = request.form.get("entity_type", "nonprofit")
            form_state["horizon"] = request.form.get("horizon", "12")
            form_state["ein"] = request.form.get("ein", "")
            form_state["years"] = request.form.get("years", "")
            action = request.form.get("action", "").strip().lower()

            try:
                entity_type = (form_state["entity_type"] or "nonprofit").strip() or "nonprofit"
                horizon = int(form_state["horizon"] or "12")

                with tempfile.TemporaryDirectory(prefix="tony_dashboard_") as work_dir:
                    normalized_path = os.path.join(work_dir, "normalized.json")
                    scored_path = os.path.join(work_dir, "scored.json")

                    if action == "upload":
                        uploaded = request.files.get("data_file")
                        if not uploaded or not uploaded.filename:
                            raise ValueError("Select a CSV, Excel, or PDF file to upload.")

                        filename = secure_filename(uploaded.filename) or "uploaded_data.csv"
                        source_path = os.path.join(work_dir, filename)
                        uploaded.save(source_path)
                        ingest.run(source_path, None, [], normalized_path)
                    elif action == "propublica":
                        ein = (form_state["ein"] or "").strip()
                        if not ein:
                            raise ValueError("EIN is required for ProPublica lookup.")

                        years_raw = (form_state["years"] or "").strip()
                        years = parse_years(years_raw) if years_raw else []
                        ingest.run("propublica", ein, years, normalized_path)
                    else:
                        raise ValueError("Unsupported dashboard action.")

                    payload = score.run(normalized_path, entity_type, horizon, scored_path)
                    status_message = "Dashboard updated with fresh scoring output."
            except Exception as exc:
                error_message = str(exc)

        charts = _build_charts(payload)
        return render_template(
            "dashboard.html",
            summary=payload.get("summary", {}),
            metadata=payload.get("metadata", {}),
            history=json.dumps(payload.get("history", [])),
            charts=charts,
            error_message=error_message,
            status_message=status_message,
            form_state=form_state,
        )

    return app


def main(data_path: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    app = create_app(data_path)
    app.run(host=host, port=port, debug=True)