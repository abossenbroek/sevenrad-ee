"""Test evidence classification fix with known negative companies.

This script validates that the LLM-based evidence classifier correctly
handles the 6 known negative companies that were previously misclassified.
"""

import json
import os
from pathlib import Path

import dspy
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from sevenrad_ee.ai.evidence_classifier import GrowLightEvidenceClassifier

# Load environment variables from .env file
load_dotenv()

console = Console()

# Known negative companies (should NOT be classified as POSITIVE)
KNOWN_NEGATIVES = [
    "P.J.J.M._Verbeek_en_P.H.M._Verbeek_Maasland.json",
    "Kwekerij_Ted_Vijverberg_B.V._De Lier.json",
    "Kwekerij_Figaro_B.V._Naaldwijk.json",
    "Fachjan_Project_Plants_Honselersdijk.json",
    "Van_Onselen_Aubergines_B.V._'s Gravenzande.json",
    "Van_der_Sar_Plants_'s Gravenzande.json",
]


def test_evidence_classifier() -> None:
    """Test evidence classifier with known negative companies."""
    # Configure DSPy with Gemini 2.5 Pro
    console.print("\n[cyan]Configuring DSPy with Gemini 2.5 Pro...[/cyan]")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]Error: GEMINI_API_KEY not found in environment[/red]")
        console.print("[yellow]Tip:[/yellow] Add to .env file: GEMINI_API_KEY=your-key")
        return

    lm = dspy.LM("gemini/gemini-2.5-pro", api_key=api_key, temperature=0.0)
    dspy.configure(lm=lm)

    classifier = dspy.Predict(GrowLightEvidenceClassifier)

    # Test each known negative
    data_dir = Path("data/research")
    results = []

    console.print(f"\n[cyan]Testing {len(KNOWN_NEGATIVES)} known negatives...[/cyan]\n")

    for filename in KNOWN_NEGATIVES:
        filepath = data_dir / filename
        if not filepath.exists():
            console.print(f"[yellow]⚠ Skipping {filename} (not found)[/yellow]")
            continue

        # Load research data
        with open(filepath) as f:
            data = json.load(f)

        company_name = data["company"]
        console.print(f"[dim]Testing:[/dim] {company_name}")

        # Test query 3 (negative signals query) - most important for negatives
        query3 = next((q for q in data["queries"] if "onbelichte teelt" in q["query"]), None)
        if not query3:
            console.print(f"  [yellow]⚠ No query 3 found[/yellow]")
            continue

        # Classify the evidence
        result = classifier(company_name=company_name, evidence_text=query3["content"])

        # Record result
        # Success = NOT classified as POSITIVE (NEGATIVE, NEUTRAL, or AMBIGUOUS all OK)
        is_correct = result.evidence_category != "POSITIVE"
        is_optimal = result.evidence_category in ["NEGATIVE", "NEUTRAL"]

        results.append(
            {
                "company": company_name,
                "category": result.evidence_category,
                "confidence": result.confidence,
                "reasoning": result.reasoning[:100] + "..." if len(result.reasoning) > 100 else result.reasoning,
                "correct": is_correct,
                "optimal": is_optimal,
            }
        )

        # Display result
        if is_optimal:
            status = "✅"
            color = "green"
        elif is_correct:
            status = "⚠️"  # Acceptable but not ideal
            color = "yellow"
        else:
            status = "❌"
            color = "red"
        console.print(f"  {status} [{color}]{result.evidence_category}[/{color}] (confidence: {result.confidence})")
        if not is_optimal and is_correct:
            console.print(f"    [dim]{result.reasoning[:80]}...[/dim]")

    # Summary table
    console.print("\n[cyan bold]Results Summary[/cyan bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Company", style="white", width=40)
    table.add_column("Category", justify="center", width=12)
    table.add_column("Confidence", justify="center", width=10)
    table.add_column("Status", justify="center", width=8)

    for r in results:
        if r["optimal"]:
            status = "✅"
            color = "green"
        elif r["correct"]:
            status = "⚠️"
            color = "yellow"
        else:
            status = "❌"
            color = "red"
        table.add_row(
            r["company"],
            f"[{color}]{r['category']}[/{color}]",
            f"{r['confidence']:.2f}",
            status,
        )

    console.print(table)

    # Final statistics
    total = len(results)
    correct = sum(1 for r in results if r["correct"])  # Not POSITIVE
    optimal = sum(1 for r in results if r["optimal"])  # NEGATIVE or NEUTRAL
    false_positives = sum(1 for r in results if r["category"] == "POSITIVE")

    console.print(f"\n[cyan]Results:[/cyan]")
    console.print(f"  Not POSITIVE: {correct}/{total} ({correct/total*100:.1f}%)")
    console.print(f"  Optimal (NEGATIVE/NEUTRAL): {optimal}/{total} ({optimal/total*100:.1f}%)")
    console.print(f"  False Positives: {false_positives}/{total}")

    if false_positives == 0:
        console.print("\n[green bold]✅ CRITICAL FIX SUCCESSFUL![/green bold]")
        console.print("[green]No negatives classified as POSITIVE (was 100% before fix)[/green]")
        if optimal == total:
            console.print("[green]All negatives optimally classified as NEGATIVE or NEUTRAL![/green]")
        else:
            console.print(f"[yellow]{total - optimal} classified as AMBIGUOUS (acceptable, not ideal)[/yellow]")
    else:
        console.print(f"\n[red bold]❌ FIX INCOMPLETE[/red bold]")
        console.print(f"[red]{false_positives} negatives still misclassified as POSITIVE[/red]")


if __name__ == "__main__":
    test_evidence_classifier()
