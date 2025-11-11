"""
Batch company research tool for Phase 1 Step 3 data expansion.

Documentation Type: How-to Guide

This tool processes multiple Dutch greenhouse companies in batch using the
3-query Perplexity strategy with evidence categorization and tier classification.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import dspy
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


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch research Dutch greenhouse companies (Phase 1 Step 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List companies from data file
  uv run python -m sevenrad_ee.operations.batch_company_research --list-companies

  # Research all pending companies
  uv run python -m sevenrad_ee.operations.batch_company_research

  # Research specific companies
  uv run python -m sevenrad_ee.operations.batch_company_research \\
      --filter "Porta Nova,BM Roses"

  # Clear cache and re-research all
  uv run python -m sevenrad_ee.operations.batch_company_research \\
      --clear-cache --force

  # Use custom companies file
  uv run python -m sevenrad_ee.operations.batch_company_research \\
      --companies-file data/my_companies.json
        """,
    )

    parser.add_argument(
        "--list-companies",
        action="store_true",
        help="List companies from file with status (researched/pending) and exit",
    )

    parser.add_argument(
        "--list-results",
        action="store_true",
        help="Display already-researched companies and exit",
    )

    parser.add_argument(
        "--companies-file",
        type=Path,
        default=Path("data/companies_to_research.json"),
        help="Path to companies JSON file (default: data/companies_to_research.json)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/research"),
        help="Output directory for research results (default: data/research)",
    )

    parser.add_argument(
        "--filter",
        type=str,
        help=(
            "Comma-separated list of company names to research "
            "(e.g., 'Porta Nova,BM Roses')"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-research companies even if already researched",
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear Perplexity API cache before running research",
    )

    parser.add_argument(
        "--summary-file",
        type=Path,
        help="Optional path to save summary report (markdown format)",
    )

    return parser.parse_args()


def load_companies(companies_file: Path) -> list[dict[str, Any]]:
    """
    Load companies from JSON file.

    Args:
        companies_file: Path to companies JSON file

    Returns:
        List of company dictionaries

    Raises:
        FileNotFoundError: If companies file doesn't exist
        ValueError: If JSON is invalid

    """
    if not companies_file.exists():
        console.print(
            f"[red]✗[/red] Companies file not found: {companies_file}",
            style="bold red",
        )
        console.print(
            "\n[yellow]Tip:[/yellow] Create a JSON file with company data:",
            style="italic",
        )
        console.print(
            '  [{"company": "Name", "location": "City", '
            '"expected": "POSITIVE|NEGATIVE", "researched": false}]'
        )
        msg = f"Companies file not found: {companies_file}"
        raise FileNotFoundError(msg)

    try:
        data: list[dict[str, Any]] = json.loads(companies_file.read_text())
        return data
    except json.JSONDecodeError as e:
        console.print(
            f"[red]✗[/red] Invalid JSON in {companies_file}",
            style="bold red",
        )
        console.print(f"\n[yellow]Error:[/yellow] {e}")
        msg = f"Invalid JSON in {companies_file}"
        raise ValueError(msg) from e


def list_companies_table(companies: list[dict[str, Any]], output_dir: Path) -> None:
    """
    Display companies with research status.

    Args:
        companies: List of company dictionaries
        output_dir: Directory where results are saved

    """
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Company", style="white")
    table.add_column("Location", style="cyan")
    table.add_column("Expected", style="yellow")
    table.add_column("Status", style="green")

    for idx, company in enumerate(companies, 1):
        # Check if already researched
        result_file = (
            output_dir
            / f"{company['company'].replace(' ', '_')}_{company['location']}.json"
        )
        is_researched = result_file.exists() or company.get("researched", False)

        status = (
            "[green]✓ Researched[/green]" if is_researched else "[dim]Pending[/dim]"
        )

        table.add_row(
            str(idx),
            company["company"],
            company["location"],
            company.get("expected", "UNKNOWN"),
            status,
        )

    console.print("\n[bold]Companies List[/bold]")
    console.print(table)

    # Summary statistics
    researched_count = sum(
        1
        for c in companies
        if (
            output_dir / f"{c['company'].replace(' ', '_')}_{c['location']}.json"
        ).exists()
        or c.get("researched", False)
    )
    pending_count = len(companies) - researched_count

    console.print(
        f"\nTotal: {len(companies)} companies | "
        f"[green]{researched_count} researched[/green] | "
        f"[dim]{pending_count} pending[/dim]"
    )


def list_research_results(output_dir: Path) -> None:
    """
    Display already-researched companies.

    Args:
        output_dir: Directory containing research results

    """
    if not output_dir.exists():
        console.print(f"[yellow]No results found in {output_dir}[/yellow]")
        return

    result_files = list(output_dir.glob("*.json"))

    if not result_files:
        console.print(f"[yellow]No research results in {output_dir}[/yellow]")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Company", style="green")
    table.add_column("Location", style="cyan")
    table.add_column("Suggestion", style="yellow")
    table.add_column("Confidence", style="white")
    table.add_column("Evidence", style="dim")

    for result_file in sorted(result_files):
        try:
            result_data = json.loads(result_file.read_text())
            result = CompanyResearchResult(**result_data)

            # Format suggestion
            suggestion = result.classification_suggestion.value
            if "POSITIVE" in suggestion:
                suggestion_display = "[green]POSITIVE[/green]"
            elif "NEGATIVE" in suggestion:
                suggestion_display = "[red]NEGATIVE[/red]"
            else:
                suggestion_display = "[yellow]NEEDS REVIEW[/yellow]"

            # Evidence counts
            evidence_str = (
                f"{len(result.evidence.positive)}+"
                f"/{len(result.evidence.negative)}-"
                f"/{len(result.evidence.ambiguous)}?"
            )

            table.add_row(
                result.company,
                result.location,
                suggestion_display,
                f"{result.confidence_score:.0%}",
                evidence_str,
            )

        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not parse {result_file.name}: {e}[/yellow]"
            )
            continue

    console.print(f"\n[bold]Research Results[/bold] ({output_dir})")
    console.print(table)
    console.print(
        f"\nTotal: {len(result_files)} companies researched\n"
        f"[dim]Evidence format: positive+/negative-/ambiguous?[/dim]"
    )


def filter_companies(
    companies: list[dict[str, Any]],
    filter_str: str | None,
    output_dir: Path,
    force: bool,
) -> list[dict[str, Any]]:
    """
    Filter companies based on criteria.

    Args:
        companies: List of company dictionaries
        filter_str: Comma-separated company names to include (or None for all)
        output_dir: Directory where results are saved
        force: Re-research even if already researched

    Returns:
        Filtered list of companies to research

    """
    filtered = companies

    # Apply name filter if provided
    if filter_str:
        filter_names = {name.strip().lower() for name in filter_str.split(",")}
        filtered = [c for c in filtered if c["company"].lower() in filter_names]
        console.print(f"[cyan]Filter:[/cyan] {len(filtered)} companies matching filter")

    # Skip already-researched unless --force
    if not force:
        pending = []
        for company in filtered:
            result_file = (
                output_dir
                / f"{company['company'].replace(' ', '_')}_{company['location']}.json"
            )
            if not result_file.exists() and not company.get("researched", False):
                pending.append(company)

        console.print(
            f"[cyan]Status:[/cyan] {len(pending)} pending, "
            f"{len(filtered) - len(pending)} already researched"
        )
        filtered = pending

    return filtered


def run_batch_research(
    researcher: CompanyResearcher,
    companies: list[dict[str, Any]],
    output_dir: Path,
) -> list[CompanyResearchResult]:
    """
    Run batch research for companies.

    Args:
        researcher: Company researcher instance
        companies: List of company dictionaries
        output_dir: Output directory for results

    Returns:
        List of research results

    """
    if not companies:
        console.print("[yellow]No companies to research[/yellow]")
        return []

    console.print(f"\n[bold]Researching {len(companies)} Companies[/bold]")
    console.print("Using 3-query Perplexity strategy with evidence categorization\n")

    results = []
    failed = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Researching companies...", total=len(companies))

        for company_data in companies:
            company_name = company_data["company"]
            location = company_data["location"]

            try:
                result = researcher.research_company(company_name, location)
                results.append(result)

                # Save result
                output_path = researcher.save_result(result, output_dir)
                console.print(f"  [green]✓[/green] Saved: {output_path.name}")

                progress.advance(task)

            except Exception as e:
                console.print(f"  [red]✗[/red] Failed: {company_name} - {e}")
                failed.append((company_name, location, str(e)))
                progress.advance(task)

    return results


def generate_summary_report(
    results: list[CompanyResearchResult],
    summary_file: Path | None,
) -> None:
    """
    Generate and display summary report.

    Args:
        results: List of research results
        summary_file: Optional path to save markdown report

    """
    if not results:
        return

    console.print("\n[bold]Research Summary Report[/bold]")

    # Overall statistics
    stats_table = Table(show_header=False, box=None)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")

    positive_count = sum(
        1 for r in results if "POSITIVE" in r.classification_suggestion.value
    )
    negative_count = sum(
        1 for r in results if "NEGATIVE" in r.classification_suggestion.value
    )
    review_count = sum(
        1 for r in results if "REVIEW" in r.classification_suggestion.value
    )

    stats_table.add_row("Companies researched", str(len(results)))
    stats_table.add_row("Positive suggestions", f"[green]{positive_count}[/green]")
    stats_table.add_row("Negative suggestions", f"[red]{negative_count}[/red]")
    stats_table.add_row("Needs manual review", f"[yellow]{review_count}[/yellow]")
    stats_table.add_row(
        "Avg confidence",
        f"{sum(r.confidence_score for r in results) / len(results):.1%}",
    )
    stats_table.add_row(
        "Avg Dutch terms",
        f"{sum(len(r.dutch_terms_found) for r in results) / len(results):.1f}",
    )
    stats_table.add_row(
        "Avg tier-2 evidence",
        f"{sum(r.tier2_percentage() for r in results) / len(results):.1f}%",
    )

    console.print(stats_table)

    # Companies needing manual review
    if review_count > 0:
        console.print("\n[bold yellow]Companies Requiring Manual Review:[/bold yellow]")
        for result in results:
            if "REVIEW" in result.classification_suggestion.value:
                console.print(
                    f"  • {result.company}, {result.location} "
                    f"[dim](Confidence: {result.confidence_score:.0%}, "
                    f"Evidence: {len(result.evidence.positive)}+/"
                    f"{len(result.evidence.negative)}-)[/dim]"
                )

    # Save markdown report if requested
    if summary_file:
        markdown = generate_markdown_report(results)
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(markdown)
        console.print(f"\n[green]✓[/green] Summary saved to: {summary_file}")


def generate_markdown_report(
    results: list[CompanyResearchResult],
) -> str:
    """
    Generate markdown summary report.

    Args:
        results: List of research results

    Returns:
        Markdown-formatted report

    """
    positive_count = sum(
        1 for r in results if "POSITIVE" in r.classification_suggestion.value
    )
    negative_count = sum(
        1 for r in results if "NEGATIVE" in r.classification_suggestion.value
    )
    review_count = sum(
        1 for r in results if "REVIEW" in r.classification_suggestion.value
    )

    pos_pct = positive_count / len(results) * 100
    neg_pct = negative_count / len(results) * 100
    review_pct = review_count / len(results) * 100
    avg_conf = sum(r.confidence_score for r in results) / len(results)
    avg_dutch = sum(len(r.dutch_terms_found) for r in results) / len(results)
    avg_tier2 = sum(r.tier2_percentage() for r in results) / len(results)

    lines = [
        "# Batch Company Research Summary",
        "",
        "## Overall Statistics",
        "",
        f"- **Total Companies Researched:** {len(results)}",
        f"- **Positive Suggestions:** {positive_count} ({pos_pct:.1f}%)",
        f"- **Negative Suggestions:** {negative_count} ({neg_pct:.1f}%)",
        f"- **Needs Manual Review:** {review_count} ({review_pct:.1f}%)",
        f"- **Average Confidence:** {avg_conf:.1%}",
        f"- **Average Dutch Terms:** {avg_dutch:.1f}",
        f"- **Average Tier-2 Evidence:** {avg_tier2:.1f}%",
        "",
        "## Detailed Results",
        "",
        "| Company | Location | Suggestion | Confidence | "
        "Evidence (P/N/A) | Tier-2 % | Dutch Terms |",
        "|---------|----------|------------|------------|"
        "------------------|----------|-------------|",
    ]

    for result in results:
        suggestion = result.classification_suggestion.value
        if "POSITIVE" in suggestion:
            suggestion_md = "✅ POSITIVE"
        elif "NEGATIVE" in suggestion:
            suggestion_md = "❌ NEGATIVE"
        else:
            suggestion_md = "⚠️ NEEDS REVIEW"

        evidence_str = (
            f"{len(result.evidence.positive)}/"
            f"{len(result.evidence.negative)}/"
            f"{len(result.evidence.ambiguous)}"
        )

        lines.append(
            f"| {result.company} | {result.location} | {suggestion_md} | "
            f"{result.confidence_score:.0%} | {evidence_str} | "
            f"{result.tier2_percentage():.0f}% | {len(result.dutch_terms_found)} |"
        )

    if review_count > 0:
        lines.extend(
            [
                "",
                "## Companies Requiring Manual Review",
                "",
            ]
        )
        for result in results:
            if "REVIEW" in result.classification_suggestion.value:
                lines.append(f"- **{result.company}, {result.location}**")
                lines.append(f"  - Confidence: {result.confidence_score:.0%}")
                lines.append(
                    f"  - Evidence: {len(result.evidence.positive)} positive, "
                    f"{len(result.evidence.negative)} negative, "
                    f"{len(result.evidence.ambiguous)} ambiguous"
                )
                lines.append(f"  - Dutch terms: {len(result.dutch_terms_found)}")
                lines.append(f"  - Tier-2 evidence: {result.tier2_percentage():.0f}%")
                lines.append("")

    return "\n".join(lines)


def main() -> int:
    """Run batch company research."""
    # Display header
    console.print(
        Panel.fit(
            "[bold cyan]Batch Company Research Tool[/bold cyan]\n"
            "Phase 1 Step 3: Data Expansion\n"
            "3-Query Perplexity Strategy with Evidence Categorization",
            border_style="cyan",
        )
    )

    args = parse_arguments()

    # Load companies
    try:
        companies = load_companies(args.companies_file)
    except (FileNotFoundError, ValueError):
        return 1

    # Handle --list-companies
    if args.list_companies:
        list_companies_table(companies, args.output_dir)
        return 0

    # Handle --list-results
    if args.list_results:
        list_research_results(args.output_dir)
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

    # Configure DSPy with Gemini for evidence classification
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        console.print(
            "[red]✗[/red] GEMINI_API_KEY not found in environment",
            style="bold red",
        )
        console.print(
            "\n[yellow]Tip:[/yellow] Create a .env file with:",
            style="italic",
        )
        console.print("  GEMINI_API_KEY='your-api-key-here'")
        console.print("  PERPLEXITY_API_KEY='your-api-key-here'")
        return 1

    # Configure DSPy LM for semantic evidence classification
    gemini_lm = dspy.LM("gemini/gemini-2.5-pro", api_key=gemini_api_key, temperature=0.0)
    dspy.configure(lm=gemini_lm)
    console.print("[green]✓[/green] DSPy configured with Gemini 2.5 Pro\n")

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
        console.print("  GEMINI_API_KEY='your-api-key-here'")
        return 1

    # Filter companies to research
    to_research = filter_companies(
        companies,
        args.filter,
        args.output_dir,
        args.force,
    )

    if not to_research:
        console.print("\n[green]✓ All companies already researched![/green]")
        console.print("[dim]Use --force to re-research[/dim]")
        return 0

    # Run batch research
    results = run_batch_research(researcher, to_research, args.output_dir)

    # Generate summary report
    if results:
        generate_summary_report(results, args.summary_file)
        console.print("\n[green]✓ Batch research complete![/green]")
        console.print(f"[dim]Results saved to {args.output_dir}[/dim]")
        return 0

    console.print("\n[yellow]⚠ No companies successfully researched[/yellow]")
    return 1


if __name__ == "__main__":
    sys.exit(main())
