import csv
import json
from pathlib import Path
from typing import Any

import pytest
from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    path = tmp_path / "filings.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["year", "revenue", "expenses", "assets", "liabilities", "unrestricted_net_assets", "program_expenses"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "year": 2022,
                    "revenue": 1200000,
                    "expenses": 1100000,
                    "assets": 900000,
                    "liabilities": 300000,
                    "unrestricted_net_assets": 600000,
                    "program_expenses": 825000,
                },
                {
                    "year": 2023,
                    "revenue": 1350000,
                    "expenses": 1150000,
                    "assets": 970000,
                    "liabilities": 280000,
                    "unrestricted_net_assets": 690000,
                    "program_expenses": 860000,
                },
            ]
        )
    return path


@pytest.fixture
def normalized_payload(tmp_path: Path) -> Path:
    payload: dict[str, Any] = {
        "metadata": {"source": "fixture", "years": [2022, 2023], "record_count": 2},
        "records": [
            {
                "year": 2022,
                "revenue": 1200000,
                "expenses": 1100000,
                "assets": 900000,
                "liabilities": 300000,
                "unrestricted_net_assets": 600000,
                "program_expenses": 825000,
            },
            {
                "year": 2023,
                "revenue": 1350000,
                "expenses": 1150000,
                "assets": 970000,
                "liabilities": 280000,
                "unrestricted_net_assets": 690000,
                "program_expenses": 860000,
            },
        ],
    }
    path = tmp_path / "normalized.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def propublica_payload() -> dict[str, Any]:
    return {
        "organization": {"ein": "123456789", "name": "Example Nonprofit"},
        "filings_with_data": [
            {
                "tax_prd_yr": 2021,
                "totrevenue": 1000000,
                "totfuncexpns": 950000,
                "totassetsend": 800000,
                "totliabend": 250000,
                "totnetassetsend": 550000,
            },
            {
                "tax_prd_yr": 2022,
                "totrevenue": 1250000,
                "totfuncexpns": 1000000,
                "totassetsend": 850000,
                "totliabend": 230000,
                "totnetassetsend": 620000,
            },
        ],
    }