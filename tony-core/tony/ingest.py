import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from PyPDF2 import PdfReader

from .config import load_config
from .utils import coalesce_number, resolve_path, write_json


def _find_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    normalized = {column.strip().lower(): column for column in frame.columns}
    for alias in aliases:
        column = normalized.get(alias.lower())
        if column:
            return column
    return None


def _normalize_frame(frame: pd.DataFrame, config: dict[str, Any], source_name: str, ein: str | None) -> list[dict[str, Any]]:
    aliases = config["column_aliases"]
    columns = {key: _find_column(frame, values) for key, values in aliases.items()}
    year_column = columns["year"]
    if not year_column:
        raise ValueError("Input data must contain a year column.")

    normalized_records: list[dict[str, Any]] = []
    working = frame.copy().where(pd.notnull(frame), None)
    for record in working.to_dict(orient="records"):
        year = int(record[year_column])
        revenue = coalesce_number(record.get(columns["revenue"])) or 0.0
        expenses = coalesce_number(record.get(columns["expenses"])) or 0.0
        assets = coalesce_number(record.get(columns["assets"]))
        liabilities = coalesce_number(record.get(columns["liabilities"]))
        net_assets = coalesce_number(record.get(columns["net_assets"]))
        if net_assets is None and assets is not None and liabilities is not None:
            net_assets = assets - liabilities
        program_expenses = coalesce_number(record.get(columns["program_expenses"]))
        normalized_records.append(
            {
                "year": year,
                "revenue": revenue,
                "expenses": expenses,
                "assets": assets,
                "liabilities": liabilities,
                "unrestricted_net_assets": net_assets,
                "program_expenses": program_expenses,
                "source": source_name,
                "ein": ein,
            }
        )

    return sorted(normalized_records, key=lambda item: item["year"])


def _load_table_file(source: str) -> pd.DataFrame:
    path = resolve_path(source)
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported tabular file type: {suffix}")


def _load_pdf_tables(source: str) -> pd.DataFrame:
    path = resolve_path(source)
    frames: list[pd.DataFrame] = []

    try:
        import camelot  # type: ignore

        tables = camelot.read_pdf(path, pages="all")
        frames.extend(table.df for table in tables if not table.df.empty)
    except Exception:
        pass

    if not frames:
        try:
            import tabula  # type: ignore

            tabula_frames = tabula.read_pdf(path, pages="all", multiple_tables=True)
            frames.extend(frame for frame in tabula_frames if not frame.empty)
        except Exception:
            pass

    if not frames:
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        raise ValueError(
            "No tabular PDF parser is installed or no tables were detected. "
            f"Extracted text length: {len(text)}. Install camelot-py or tabula-py to parse filing PDFs."
        )

    return pd.concat(frames, ignore_index=True)


def _normalize_propublica_payload(payload: dict[str, Any], years: list[int]) -> tuple[str | None, list[dict[str, Any]]]:
    organization = payload.get("organization", payload)
    filings = payload.get("filings_with_data") or organization.get("filings_with_data") or []
    ein = str(organization.get("ein")) if organization.get("ein") else None
    records: list[dict[str, Any]] = []
    for filing in filings:
        year = int(filing.get("tax_prd_yr"))
        if years and year not in years:
            continue
        records.append(
            {
                "year": year,
                "revenue": coalesce_number(filing.get("totrevenue")) or 0.0,
                "expenses": coalesce_number(filing.get("totfuncexpns")) or 0.0,
                "assets": coalesce_number(filing.get("totassetsend")),
                "liabilities": coalesce_number(filing.get("totliabend")),
                "unrestricted_net_assets": coalesce_number(filing.get("totnetassetsend")),
                "program_expenses": None,
                "source": "propublica",
                "ein": ein,
            }
        )
    return ein, sorted(records, key=lambda item: item["year"])


def _fetch_propublica(ein: str, config: dict[str, Any]) -> dict[str, Any]:
    base_url = config["sources"]["propublica"]["base_url"].rstrip("/")
    response = requests.get(f"{base_url}/organizations/{ein}.json", timeout=30)
    response.raise_for_status()
    return response.json()


def run(
    source: str,
    ein: str | None,
    years: list[int],
    out_file: str,
    config_path: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    source_value = source.strip()
    if source_value.lower() == "propublica":
        if not ein:
            raise ValueError("EIN is required when source is 'propublica'.")
        payload = _fetch_propublica(ein, config)
        normalized_ein, records = _normalize_propublica_payload(payload, years)
        metadata = {
            "source": "propublica",
            "ein": normalized_ein or ein,
            "organization": payload.get("organization", {}).get("name"),
        }
    else:
        resolved = resolve_path(source_value)
        suffix = Path(resolved).suffix.lower()
        frame = _load_pdf_tables(resolved) if suffix == ".pdf" else _load_table_file(resolved)
        records = _normalize_frame(frame, config, resolved, ein)
        if years:
            records = [record for record in records if record["year"] in years]
        metadata = {"source": resolved, "ein": ein}

    result = {
        "metadata": {
            **metadata,
            "ingested_at": datetime.now().isoformat(),
            "record_count": len(records),
            "years": [record["year"] for record in records],
        },
        "records": records,
    }
    write_json(out_file, result)
    logging.info("Ingested %s records into %s", len(records), out_file)
    return result
