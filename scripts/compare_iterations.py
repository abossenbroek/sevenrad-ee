#!/usr/bin/env python3
"""Compare optimization results across iterations.

Generates comprehensive comparison report showing improvements from
dataset expansion and targeted data collection.
"""

import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def load_iteration_results(results_dir: Path) -> dict[str, Any]:
    """Load results from an optimization iteration."""
    cv_file = results_dir / "cv_results.json"
    if not cv_file.exists():
        return None

    with open(cv_file) as f:
        cv_results = json.load(f)

    # Count examples
    example_count = 0
    data_dir = Path("data/research")
    if data_dir.exists():
        example_count = len(list(data_dir.glob("*.json")))

    return {
        "name": results_dir.name,
        "cv_results": cv_results,
        "example_count": example_count,
        "results_dir": str(results_dir),
    }


def calculate_improvements(
    baseline: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Calculate improvements from baseline to current iteration."""
    baseline_cv = baseline["cv_results"]
    current_cv = current["cv_results"]

    improvements = {
        "mean_f1_delta": current_cv["mean_f1"] - baseline_cv["mean_f1"],
        "mean_f1_pct_change": (
            (current_cv["mean_f1"] - baseline_cv["mean_f1"]) / baseline_cv["mean_f1"]
        )
        * 100,
        "std_f1_delta": current_cv["std_f1"] - baseline_cv["std_f1"],
        "gap_delta": current_cv.get("overfitting_gap", 0) - baseline_cv.get(
            "overfitting_gap", 0
        ),
        "example_delta": current["example_count"] - baseline["example_count"],
        "example_multiplier": current["example_count"] / baseline["example_count"],
    }

    return improvements


def display_comparison(iterations: list[dict[str, Any]]) -> None:
    """Display side-by-side comparison of all iterations."""
    console.print(
        Panel.fit(
            "[bold cyan]Iteration Comparison[/bold cyan]\n"
            f"Comparing {len(iterations)} optimization iterations",
            border_style="cyan",
        )
    )

    # Summary table
    console.print("\n[bold]📊 Performance Comparison[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Iteration", style="cyan")
    table.add_column("Examples", style="white")
    table.add_column("Mean F1", style="white")
    table.add_column("Std F1", style="white")
    table.add_column("Overfit Gap", style="white")
    table.add_column("Status", style="white")

    for iteration in iterations:
        cv = iteration["cv_results"]
        mean_f1 = cv["mean_f1"]
        std_f1 = cv["std_f1"]
        gap = cv.get("overfitting_gap", 0)

        # Status indicator
        if mean_f1 >= 0.85:
            status = "🟢 Excellent"
        elif mean_f1 >= 0.75:
            status = "🟡 Good"
        elif mean_f1 >= 0.65:
            status = "🟠 Fair"
        else:
            status = "🔴 Needs Work"

        table.add_row(
            iteration["name"],
            str(iteration["example_count"]),
            f"{mean_f1:.2%}",
            f"{std_f1:.2%}",
            f"{gap:+.2%}",
            status,
        )

    console.print(table)

    # Improvements from baseline
    if len(iterations) > 1:
        console.print("\n[bold]📈 Improvements from Baseline[/bold]")
        baseline = iterations[0]

        for iteration in iterations[1:]:
            improvements = calculate_improvements(baseline, iteration)

            console.print(f"\n  [cyan]{iteration['name']}[/cyan] vs [dim]{baseline['name']}[/dim]:")
            console.print(
                f"    Dataset size: {baseline['example_count']} → {iteration['example_count']} "
                f"({improvements['example_multiplier']:.1f}x, +{improvements['example_delta']})"
            )
            console.print(
                f"    Mean F1: {baseline['cv_results']['mean_f1']:.2%} → "
                f"{iteration['cv_results']['mean_f1']:.2%} "
                f"({improvements['mean_f1_delta']:+.2%}, {improvements['mean_f1_pct_change']:+.1f}%)"
            )
            console.print(
                f"    Std F1: {baseline['cv_results']['std_f1']:.2%} → "
                f"{iteration['cv_results']['std_f1']:.2%} "
                f"({improvements['std_f1_delta']:+.2%})"
            )
            console.print(
                f"    Overfit Gap: {baseline['cv_results'].get('overfitting_gap', 0):+.2%} → "
                f"{iteration['cv_results'].get('overfitting_gap', 0):+.2%} "
                f"({improvements['gap_delta']:+.2%})"
            )

    # Learning curve analysis
    if len(iterations) >= 2:
        console.print("\n[bold]📉 Learning Curve Analysis[/bold]")

        # Calculate marginal improvements
        for i in range(1, len(iterations)):
            prev = iterations[i - 1]
            curr = iterations[i]

            example_added = curr["example_count"] - prev["example_count"]
            f1_gained = curr["cv_results"]["mean_f1"] - prev["cv_results"]["mean_f1"]
            efficiency = (
                f1_gained / example_added if example_added > 0 else 0
            )

            console.print(
                f"  {prev['name']} → {curr['name']}: "
                f"+{example_added} examples → {f1_gained:+.2%} F1 "
                f"({efficiency:.4%} F1/example)"
            )

        # Diminishing returns analysis
        if len(iterations) >= 3:
            console.print("\n  [dim]Efficiency trend:")
            efficiencies = []
            for i in range(1, len(iterations)):
                prev = iterations[i - 1]
                curr = iterations[i]
                example_added = curr["example_count"] - prev["example_count"]
                f1_gained = (
                    curr["cv_results"]["mean_f1"] - prev["cv_results"]["mean_f1"]
                )
                if example_added > 0:
                    efficiencies.append(f1_gained / example_added)

            if len(efficiencies) >= 2:
                trend = "declining" if efficiencies[-1] < efficiencies[0] else "stable"
                console.print(f"    {trend} (as expected with larger datasets)")


def generate_markdown_report(
    iterations: list[dict[str, Any]], output_file: Path
) -> None:
    """Generate markdown comparison report."""
    lines = [
        "# Optimization Iteration Comparison\n",
        f"**Iterations Compared**: {len(iterations)}\n",
        f"**Date**: {Path().absolute().name}\n",
        "\n## Performance Summary\n",
        "\n| Iteration | Examples | Mean F1 | Std F1 | Overfit Gap | Status |",
        "|-----------|----------|---------|--------|-------------|--------|",
    ]

    for iteration in iterations:
        cv = iteration["cv_results"]
        mean_f1 = cv["mean_f1"]
        std_f1 = cv["std_f1"]
        gap = cv.get("overfitting_gap", 0)

        if mean_f1 >= 0.85:
            status = "✅ Excellent"
        elif mean_f1 >= 0.75:
            status = "🟡 Good"
        elif mean_f1 >= 0.65:
            status = "🟠 Fair"
        else:
            status = "❌ Needs Work"

        lines.append(
            f"| {iteration['name']} | {iteration['example_count']} | "
            f"{mean_f1:.2%} | {std_f1:.2%} | {gap:+.2%} | {status} |"
        )

    # Improvements section
    if len(iterations) > 1:
        lines.append("\n## Improvements from Baseline\n")
        baseline = iterations[0]

        for iteration in iterations[1:]:
            improvements = calculate_improvements(baseline, iteration)

            lines.extend(
                [
                    f"\n### {iteration['name']} vs {baseline['name']}\n",
                    f"- **Dataset size**: {baseline['example_count']} → {iteration['example_count']} "
                    f"({improvements['example_multiplier']:.1f}x, +{improvements['example_delta']})",
                    f"- **Mean F1**: {baseline['cv_results']['mean_f1']:.2%} → "
                    f"{iteration['cv_results']['mean_f1']:.2%} "
                    f"({improvements['mean_f1_delta']:+.2%}, {improvements['mean_f1_pct_change']:+.1f}%)",
                    f"- **Std F1**: {baseline['cv_results']['std_f1']:.2%} → "
                    f"{iteration['cv_results']['std_f1']:.2%} "
                    f"({improvements['std_f1_delta']:+.2%})",
                    f"- **Overfit Gap**: {baseline['cv_results'].get('overfitting_gap', 0):+.2%} → "
                    f"{iteration['cv_results'].get('overfitting_gap', 0):+.2%} "
                    f"({improvements['gap_delta']:+.2%})",
                ]
            )

    # Learning curve
    if len(iterations) >= 2:
        lines.append("\n## Learning Curve\n")

        for i in range(1, len(iterations)):
            prev = iterations[i - 1]
            curr = iterations[i]

            example_added = curr["example_count"] - prev["example_count"]
            f1_gained = curr["cv_results"]["mean_f1"] - prev["cv_results"]["mean_f1"]
            efficiency = (
                f1_gained / example_added if example_added > 0 else 0
            )

            lines.append(
                f"- {prev['name']} → {curr['name']}: "
                f"+{example_added} examples → {f1_gained:+.2%} F1 "
                f"({efficiency:.4%} F1/example)"
            )

    # Write report
    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    console.print(f"\n[green]✓[/green] Markdown report saved to {output_file}")


def main() -> int:
    """Compare optimization iterations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare optimization results across iterations"
    )
    parser.add_argument(
        "iteration_dirs",
        nargs="+",
        type=Path,
        help="Directories containing iteration results (in order)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/iteration_comparison.md"),
        help="Output markdown report file",
    )

    args = parser.parse_args()

    # Load all iterations
    console.print("[dim]Loading iteration results...[/dim]")
    iterations = []
    for results_dir in args.iteration_dirs:
        result = load_iteration_results(results_dir)
        if result is None:
            console.print(f"[yellow]⚠[/yellow] No CV results found in {results_dir}")
            continue
        iterations.append(result)

    if not iterations:
        console.print("[red]✗[/red] No valid iterations found")
        return 1

    # Display comparison
    display_comparison(iterations)

    # Generate markdown report
    generate_markdown_report(iterations, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
