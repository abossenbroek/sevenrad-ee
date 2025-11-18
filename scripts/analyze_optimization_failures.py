#!/usr/bin/env python3
"""Analyze optimization results to identify failure patterns and weak areas.

This script examines CV results, fold-level performance, and individual company
predictions to identify systematic weaknesses that should guide the next round
of data collection.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def load_cv_results(results_dir: Path) -> dict[str, Any]:
    """Load cross-validation results."""
    cv_file = results_dir / "cv_results.json"
    if not cv_file.exists():
        console.print(f"[red]✗[/red] CV results not found: {cv_file}")
        sys.exit(1)

    with open(cv_file) as f:
        return json.load(f)


def load_company_data(data_dir: Path) -> list[dict[str, Any]]:
    """Load all company research data."""
    companies = []
    for json_file in data_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            companies.append(data)
    return companies


def analyze_fold_performance(cv_results: dict[str, Any]) -> dict[str, Any]:
    """Analyze performance across CV folds."""
    fold_scores = cv_results.get("fold_scores", [])

    if not fold_scores:
        return {}

    return {
        "best_fold": max(fold_scores),
        "worst_fold": min(fold_scores),
        "range": max(fold_scores) - min(fold_scores),
        "below_mean": sum(1 for s in fold_scores if s < cv_results["mean_f1"]),
        "above_mean": sum(1 for s in fold_scores if s >= cv_results["mean_f1"]),
    }


def analyze_misclassifications(
    companies: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze patterns in misclassified companies."""
    # Classify companies by evidence strength
    weak_positive = []  # POSITIVE but low evidence
    weak_negative = []  # NEGATIVE but low evidence
    contradictory = []  # Both positive and negative evidence
    no_evidence = []  # No evidence found

    for company in companies:
        evidence = company.get("evidence", {})
        pos_count = len(evidence.get("positive", []))
        neg_count = len(evidence.get("negative", []))
        total_evidence = pos_count + neg_count
        confidence = company.get("confidence_score", 0)
        classification = company.get("classification_suggestion", "")

        # Categorize
        if total_evidence == 0:
            no_evidence.append(company)
        elif pos_count > 0 and neg_count > 0:
            contradictory.append(company)
        elif "POSITIVE" in classification and confidence < 0.7:
            weak_positive.append(company)
        elif "NEGATIVE" in classification and confidence < 0.7:
            weak_negative.append(company)

    return {
        "weak_positive": weak_positive,
        "weak_negative": weak_negative,
        "contradictory": contradictory,
        "no_evidence": no_evidence,
    }


def analyze_evidence_patterns(companies: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze evidence quality and source patterns."""
    tier_counts = Counter()
    dutch_term_counts = Counter()
    supplier_mentions = Counter()

    for company in companies:
        evidence = company.get("evidence", {})

        # Count evidence tiers
        for item in (
            evidence.get("positive", [])
            + evidence.get("negative", [])
            + evidence.get("ambiguous", [])
        ):
            tier = item.get("tier", "unknown")
            tier_counts[tier] += 1

        # Count Dutch terms
        dutch_terms = company.get("dutch_terms_found", [])
        for term in dutch_terms:
            dutch_term_counts[term] += 1

        # Extract supplier mentions from queries
        for query in company.get("queries", []):
            content = query.get("content", "").lower()
            if "signify" in content or "philips" in content:
                supplier_mentions["Signify/Philips"] += 1
            if "hortilux" in content:
                supplier_mentions["Hortilux"] += 1
            if "gavita" in content:
                supplier_mentions["Gavita"] += 1

    return {
        "tier_distribution": dict(tier_counts.most_common()),
        "common_dutch_terms": dict(dutch_term_counts.most_common(10)),
        "supplier_mentions": dict(supplier_mentions.most_common()),
    }


def identify_underrepresented_patterns(
    companies: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Identify patterns that need more examples."""
    # Extract patterns
    locations = Counter()
    crop_types = Counter()

    for company in companies:
        location = company.get("location", "Unknown")
        locations[location] += 1

        # Try to extract crop type from notes or queries
        notes = company.get("notes", "").lower()
        for query in company.get("queries", []):
            content = query.get("content", "").lower()
            notes += " " + content

        # Detect crop types
        if "roses" in notes or "rozen" in notes:
            crop_types["Roses"] += 1
        elif "orchid" in notes or "orchidee" in notes:
            crop_types["Orchids"] += 1
        elif "tomato" in notes or "tomaten" in notes:
            crop_types["Tomatoes"] += 1
        elif "pepper" in notes or "paprika" in notes:
            crop_types["Peppers"] += 1
        elif "gerbera" in notes:
            crop_types["Gerbera"] += 1
        elif "chrysant" in notes:
            crop_types["Chrysanthemums"] += 1
        else:
            crop_types["Other/Unknown"] += 1

    # Identify underrepresented (< 5 examples)
    underrepresented = {
        "locations": [loc for loc, count in locations.items() if count < 5],
        "crop_types": [crop for crop, count in crop_types.items() if count < 5],
    }

    return underrepresented


def generate_recommendations(
    cv_results: dict[str, Any],
    misclassifications: dict[str, Any],
    evidence_patterns: dict[str, Any],
    underrepresented: dict[str, list[str]],
) -> list[str]:
    """Generate actionable recommendations for next iteration."""
    recommendations = []

    mean_f1 = cv_results.get("mean_f1", 0)
    std_f1 = cv_results.get("std_f1", 0)

    # Performance-based recommendations
    if mean_f1 < 0.75:
        recommendations.append(
            f"⚠️  Mean F1 ({mean_f1:.1%}) below target (75%) - collect 35-50 more examples"
        )
    elif mean_f1 < 0.85:
        recommendations.append(
            f"⚙️  Mean F1 ({mean_f1:.1%}) progressing - collect 20-30 targeted examples"
        )
    else:
        recommendations.append(
            f"✅ Mean F1 ({mean_f1:.1%}) strong - collect 10-15 edge cases for polish"
        )

    if std_f1 > 0.08:
        recommendations.append(
            f"⚠️  High variance (std={std_f1:.1%}) - add more examples for consistency"
        )

    # Evidence-based recommendations
    no_evidence_count = len(misclassifications["no_evidence"])
    if no_evidence_count > 10:
        recommendations.append(
            f"📚 {no_evidence_count} companies with no evidence - prioritize well-documented companies"
        )

    contradictory_count = len(misclassifications["contradictory"])
    if contradictory_count > 5:
        recommendations.append(
            f"⚖️  {contradictory_count} companies with contradictory evidence - "
            "collect examples with clear POSITIVE/NEGATIVE signals"
        )

    weak_count = len(misclassifications["weak_positive"]) + len(
        misclassifications["weak_negative"]
    )
    if weak_count > 10:
        recommendations.append(
            f"💪 {weak_count} weak classifications - prioritize high-confidence cases"
        )

    # Pattern-based recommendations
    underrep_crops = underrepresented.get("crop_types", [])
    if underrep_crops:
        recommendations.append(
            f"🌱 Underrepresented crops: {', '.join(underrep_crops[:3])} - collect more examples"
        )

    underrep_locs = underrepresented.get("locations", [])
    if len(underrep_locs) > 5:
        recommendations.append(
            f"📍 {len(underrep_locs)} locations with <5 examples - diversify geography"
        )

    return recommendations


def display_analysis(
    cv_results: dict[str, Any],
    fold_analysis: dict[str, Any],
    misclassifications: dict[str, Any],
    evidence_patterns: dict[str, Any],
    underrepresented: dict[str, list[str]],
    recommendations: list[str],
) -> None:
    """Display comprehensive analysis with Rich formatting."""
    console.print(
        Panel.fit(
            "[bold cyan]Optimization Failure Analysis[/bold cyan]\n"
            "Identify weak areas to guide next data collection iteration",
            border_style="cyan",
        )
    )

    # Performance summary
    console.print("\n[bold]📊 Performance Summary[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Status", style="white")

    mean_f1 = cv_results.get("mean_f1", 0)
    std_f1 = cv_results.get("std_f1", 0)
    gap = cv_results.get("overfitting_gap", 0)

    table.add_row(
        "Mean F1",
        f"{mean_f1:.2%}",
        "✅" if mean_f1 >= 0.75 else "⚠️" if mean_f1 >= 0.65 else "❌",
    )
    table.add_row(
        "Std F1",
        f"{std_f1:.2%}",
        "✅" if std_f1 <= 0.05 else "⚠️" if std_f1 <= 0.08 else "❌",
    )
    table.add_row(
        "Overfitting Gap",
        f"{gap:+.2%}",
        "✅" if abs(gap) < 0.05 else "⚠️" if abs(gap) < 0.10 else "❌",
    )

    if fold_analysis:
        table.add_row("Best Fold", f"{fold_analysis['best_fold']:.2%}", "")
        table.add_row("Worst Fold", f"{fold_analysis['worst_fold']:.2%}", "")
        table.add_row("Range", f"{fold_analysis['range']:.2%}", "")

    console.print(table)

    # Misclassification analysis
    console.print("\n[bold]⚠️  Misclassification Patterns[/bold]")
    misc_table = Table(show_header=True, header_style="bold magenta")
    misc_table.add_column("Category", style="cyan")
    misc_table.add_column("Count", style="white")
    misc_table.add_column("Examples", style="dim")

    for category, companies in misclassifications.items():
        count = len(companies)
        examples = ", ".join(c["company"][:30] for c in companies[:3])
        if count > 3:
            examples += f" (+{count-3} more)"
        misc_table.add_row(
            category.replace("_", " ").title(), str(count), examples if count > 0 else "-"
        )

    console.print(misc_table)

    # Evidence patterns
    console.print("\n[bold]📚 Evidence Quality Analysis[/bold]")
    ev_table = Table(show_header=True, header_style="bold magenta")
    ev_table.add_column("Source Tier", style="cyan")
    ev_table.add_column("Count", style="white")

    tier_dist = evidence_patterns.get("tier_distribution", {})
    for tier, count in sorted(tier_dist.items(), key=lambda x: x[1], reverse=True):
        ev_table.add_row(tier.replace("_", " ").title(), str(count))

    console.print(ev_table)

    # Underrepresented patterns
    console.print("\n[bold]🎯 Underrepresented Patterns (<5 examples)[/bold]")
    for pattern_type, patterns in underrepresented.items():
        if patterns:
            console.print(
                f"  [cyan]{pattern_type.replace('_', ' ').title()}:[/cyan] "
                f"{', '.join(patterns[:5])}"
            )
            if len(patterns) > 5:
                console.print(f"    [dim](+{len(patterns)-5} more)[/dim]")

    # Recommendations
    console.print("\n[bold]💡 Recommendations for Next Iteration[/bold]")
    for i, rec in enumerate(recommendations, 1):
        console.print(f"  {i}. {rec}")


def main() -> int:
    """Run failure analysis on optimization results."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze optimization failures to guide next iteration"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/65examples"),
        help="Directory containing optimization results",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/research"),
        help="Directory containing company research data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for recommendations (optional)",
    )

    args = parser.parse_args()

    # Load data
    console.print("[dim]Loading results...[/dim]")
    cv_results = load_cv_results(args.results_dir)
    companies = load_company_data(args.data_dir)

    # Analyze
    console.print("[dim]Analyzing failures...[/dim]")
    fold_analysis = analyze_fold_performance(cv_results)
    misclassifications = analyze_misclassifications(companies)
    evidence_patterns = analyze_evidence_patterns(companies)
    underrepresented = identify_underrepresented_patterns(companies)
    recommendations = generate_recommendations(
        cv_results, misclassifications, evidence_patterns, underrepresented
    )

    # Display
    display_analysis(
        cv_results,
        fold_analysis,
        misclassifications,
        evidence_patterns,
        underrepresented,
        recommendations,
    )

    # Save recommendations
    if args.output:
        analysis = {
            "cv_results": cv_results,
            "fold_analysis": fold_analysis,
            "misclassifications": {
                k: len(v) for k, v in misclassifications.items()
            },
            "evidence_patterns": evidence_patterns,
            "underrepresented": underrepresented,
            "recommendations": recommendations,
        }

        with open(args.output, "w") as f:
            json.dump(analysis, f, indent=2)

        console.print(f"\n[green]✓[/green] Analysis saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
