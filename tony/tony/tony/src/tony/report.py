from __future__ import annotations
import csv
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from .glossary import GLOSSARY

def print_brief_ledger(ledger: Dict[str, Any], with_defs: bool = False, with_summary: bool = False) -> None:
    console = Console()
    console.rule("[bold]TONY Brief Ledger[/bold]")
    meta = ledger["meta"]
    console.print(f"EIN: [bold]{meta.get('ein')}[/bold]  Years: {meta.get('years')}")
    
    table = Table(show_header=True, header_style="bold")
    for col in ["Year","SP","e","ATTR","DRP","VO","TOTAL"]:
        table.add_column(col)
    for r in ledger["scores"]:
        table.add_row(
            str(r["year"]), f"{r['sp']:.3f}", f"{r['e']:.3f}", f"{r['attr']:.3f}",
            f"{r['drp']:.3f}", f"{r['vo']:.3f}", f"{r['total']:.3f}",
        )
    console.print(table)
    console.print(f"[bold]Overall:[/bold] {ledger['overall']:.3f}")
    
    if with_defs:
        console.print("\n[bold]Variable Definitions:[/bold]")
        for key, desc in GLOSSARY.items():
            console.print(f"[cyan]{key}[/cyan]: {desc}")
    
    if with_summary:
        console.print("\n[bold]What This Means:[/bold]")
        console.print(interpret_scores(ledger["scores"]))

def interpret_scores(records: list[Dict[str, Any]]) -> str:
    if not records:
        return "No records to interpret."
    avg_scores = {}
    for var in ["sp","e","attr","drp","vo"]:
        avg_scores[var] = sum(r[var] for r in records) / len(records)
    top_vars = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    top_vars = [v for v in top_vars if v[1] > 0.2]
    if not top_vars:
        return "No significant distortion patterns detected."
    parts = [GLOSSARY[v[0]] for v in top_vars[:2]]
    return "Key drivers: " + "; ".join(parts)

def export_csv(ledger: Dict[str, Any], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EIN", ledger["meta"].get("ein")]); w.writerow([])
        w.writerow(["year","sp","epsilon","attr","drp","vo","total"])
        for r in ledger["scores"]:
            w.writerow([r["year"], r["sp"], r["e"], r["attr"], r["drp"], r["vo"], r["total"]])
        w.writerow([]); w.writerow(["overall", ledger["overall"]])
