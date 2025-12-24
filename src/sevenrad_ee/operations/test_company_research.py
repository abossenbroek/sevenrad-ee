"""
Test complete Phase 1.2 + 1.3 infrastructure with sample companies.

Documentation Type: How-to Guide

This tool validates the complete research infrastructure by researching 3 Dutch
greenhouse companies using the 3-query Perplexity strategy with evidence categorization.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from sevenrad_ee.ai.company_research_models import CompanyResearchResult
from sevenrad_ee.ai.company_researcher import CompanyResearcher
from sevenrad_ee.ai.perplexity_cache import CacheManager
from sevenrad_ee.ai.perplexity_client import PerplexityClient

# Load environment variables from .env file
load_dotenv()

console = Console()

# Test companies from IMPROVE_PROMPT.md
TEST_COMPANIES = [
    ("Porta Nova", "Waddinxveen"),
    ("Kwekerij Overgaag", "Maasland"),
    ("BM Roses", "Maasland"),
]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test company research infrastructure (Phase 1.2 + 1.3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Research 3 sample companies
  uv run python -m sevenrad_ee.operations.test_company_research

  # Clear cache before researching
  uv run python -m sevenrad_ee.operations.test_company_research --clear-cache

  # Show research results directory
  uv run python -m sevenrad_ee.operations.test_company_research --show-results
        """,
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache before running research",
    )

    parser.add_argument(
        "--show-results",
        action="store_true",
        help="Display saved research results and exit",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/research"),
        help="Output directory for research results (default: data/research)",
    )

    return parser.parse_args()


def display_research_results(output_dir: Path) -> None:
    """
    Display saved research results.

    Args:
        output_dir: Directory containing research results

    """
    if not output_dir.exists():
        console.print(f"[yellow]No results found in {output_dir}[/yellow]")
        return

    result_files = list(output_dir.glob("*.json"))

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Company", style="green")
    table.add_column("File", style="white")
    table.add_column("Size", style="dim")

    for result_file in sorted(result_files):
        company_name = result_file.stem.replace("_", " ")
        size_kb = result_file.stat().st_size / 1024
        table.add_row(
            company_name,
            result_file.name,
            f"{size_kb:.1f} KB",
        )

    console.print(f"\n[bold]Research Results[/bold] ({output_dir})")
    console.print(table)
    console.print(f"\nTotal: {len(result_files)} companies researched")


def display_result_summary(result: CompanyResearchResult) -> None:
    """
    Display summary of a single research result.

    Args:
        result: Company research result to display

    """
    console.print(f"\n[bold cyan]┌─ {result.company}, {result.location}[/bold cyan]")

    # Determine color based on suggestion
    is_positive = "POSITIVE" in result.classification_suggestion.value
    suggestion_color = "green" if is_positive else "yellow"

    console.print(
        f"[dim]│[/dim] Suggestion: "
        f"[{suggestion_color}]{result.classification_suggestion.value}[/]"
    )
    console.print(f"[dim]│[/dim] Confidence: {result.confidence_score:.0%}")
    console.print(f"[dim]│[/dim] Queries: {len(result.queries)}")
    console.print(
        f"[dim]│[/dim] Evidence: "
        f"[green]{len(result.evidence.positive)} positive[/green], "
        f"[red]{len(result.evidence.negative)} negative[/red], "
        f"[yellow]{len(result.evidence.ambiguous)} ambiguous[/yellow]"
    )
    console.print(f"[dim]│[/dim] Dutch terms: {len(result.dutch_terms_found)}")
    console.print(
        f"[dim]│[/dim] Tier-2 evidence: {result.tier2_evidence_count()} "
        f"({result.tier2_percentage():.0f}%)"
    )
    console.print("[dim]└─[/dim]")


def run_research_test(
    researcher: CompanyResearcher,
    companies: list[tuple[str, str]],
    output_dir: Path,
) -> int:
    """
    Run research test with sample companies.

    Args:
        researcher: Company researcher instance
        companies: List of (company_name, location) tuples
        output_dir: Output directory for results

    Returns:
        Exit code (0 for success, 1 for failure)

    """
    console.print(f"\n[bold]Researching {len(companies)} Companies[/bold]")
    console.print(f"Using 3-query Perplexity strategy with evidence categorization\n")

    results = []
    failed_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Researching companies...", total=len(companies))

        for company_name, location in companies:
            try:
                result = researcher.research_company(company_name, location)
                results.append(result)

                # Save result
                output_path = researcher.save_result(result, output_dir)
                console.print(f"  [green]✓[/green] Saved: {output_path.name}")

                progress.advance(task)

            except Exception as e:
                console.print(f"  [red]✗[/red] Failed: {company_name} - {e}")
                failed_count += 1
                progress.advance(task)

    # Display results summary
    console.print("\n[bold]Research Results Summary[/bold]")

    for result in results:
        display_result_summary(result)

    # Overall statistics
    console.print(f"\n[bold]Overall Statistics[/bold]")
    stats_table = Table(show_header=False)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")

    stats_table.add_row("Companies researched", str(len(results)))
    stats_table.add_row("Failed", str(failed_count))
    stats_table.add_row(
        "Positive suggestions",
        str(sum(1 for r in results if "POSITIVE" in r.classification_suggestion.value)),
    )
    stats_table.add_row(
        "Negative suggestions",
        str(sum(1 for r in results if "NEGATIVE" in r.classification_suggestion.value)),
    )
    stats_table.add_row(
        "Needs review",
        str(sum(1 for r in results if "REVIEW" in r.classification_suggestion.value)),
    )
    stats_table.add_row(
        "Avg confidence",
        f"{sum(r.confidence_score for r in results) / len(results):.0%}"
        if results
        else "N/A",
    )
    stats_table.add_row(
        "Avg Dutch terms",
        f"{sum(len(r.dutch_terms_found) for r in results) / len(results):.1f}"
        if results
        else "N/A",
    )

    console.print(stats_table)

    # Success/failure status
    if len(results) == len(companies) and failed_count == 0:
        console.print("\n[green]✓ All companies researched successfully![/green]")
        console.print(f"[dim]Results saved to {output_dir}[/dim]")
        return 0
    else:
        console.print(f"\n[yellow]⚠ {failed_count} companies failed[/yellow]")
        return 1


def main() -> int:
    """Run Phase 1.2 + 1.3 validation test."""
    # Display header
    console.print(
        Panel.fit(
            "[bold cyan]Company Research Infrastructure Test[/bold cyan]\n"
            "Phase 1.2 (Caching) + Phase 1.3 (Research Models)\n"
            "3-Query Perplexity Strategy with Evidence Categorization",
            border_style="cyan",
        )
    )

    args = parse_arguments()

    # Handle --show-results
    if args.show_results:
        display_research_results(args.output_dir)
        return 0

    # Initialize components
    cache_manager = CacheManager()

    # Handle --clear-cache
    if args.clear_cache:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Clearing cache...", total=None)
            count = cache_manager.clear_cache()
            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Cleared {count} cached queries\n")

    # Initialize client and researcher
    try:
        client = PerplexityClient(cache_manager=cache_manager)
        researcher = CompanyResearcher(client)
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}", style="bold red")
        console.print(
            "\n[yellow]Tip:[/yellow] Create a .env file with:",
            style="italic",
        )
        console.print("  PERPLEXITY_API_KEY='your-api-key-here'")
        return 1

    # Run research test
    return run_research_test(researcher, TEST_COMPANIES, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
