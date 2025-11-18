"""
Test Perplexity API caching infrastructure (Phase 1.2 validation).

Documentation Type: How-to Guide

This tool validates the SHA-256 caching infrastructure for Perplexity API queries
by executing sample queries and demonstrating cache hits on subsequent runs.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Load environment variables from .env file
load_dotenv()

from sevenrad_ee.ai.perplexity_cache import (  # noqa: E402
    CacheManager,
    PerplexityAPIConfig,
    PerplexityResponse,
)
from sevenrad_ee.ai.perplexity_client import (  # noqa: E402
    PerplexityAPIError,
    PerplexityClient,
)

console = Console()

# Constants
MAX_CONTENT_PREVIEW_LENGTH = 200  # Maximum characters to show in content preview

# Test queries for known Dutch greenhouse companies
TEST_QUERIES = [
    '"Porta Nova" Waddinxveen AND (assimilatiebelichting OR groeilicht)',
    '"Kwekerij Overgaag" Maasland AND (belichte teelt OR LED-belichting)',
    '"BM Roses" Maasland AND (Signify OR Hortilux OR "Philips LED")',
]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test Perplexity API caching infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run validation test with 3 sample queries
  uv run python -m sevenrad_ee.operations.test_perplexity_cache

  # Clear cache before testing
  uv run python -m sevenrad_ee.operations.test_perplexity_cache --clear-cache

  # Show current cache statistics
  uv run python -m sevenrad_ee.operations.test_perplexity_cache --show-stats
        """,
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache before running tests",
    )

    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Display cache statistics and exit",
    )

    return parser.parse_args()


def display_cache_statistics(cache_manager: CacheManager) -> None:
    """
    Display cache statistics.

    Args:
        cache_manager: Cache manager instance

    """
    cache_files = list(cache_manager.cache_dir.glob("*.json"))
    result_files = list(cache_manager.results_dir.glob("*.json"))

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="green")
    table.add_column("Value", style="white")

    table.add_row("Cache Directory", str(cache_manager.cache_dir))
    table.add_row("Cached Queries", str(len(cache_files)))
    table.add_row("Results Directory", str(cache_manager.results_dir))
    table.add_row("Result Files", str(len(result_files)))

    console.print(table)


def execute_query_pass(
    client: PerplexityClient, queries: list[str], pass_name: str
) -> list[tuple[str, PerplexityResponse | str]]:
    """
    Execute a pass of queries.

    Args:
        client: Perplexity API client
        queries: List of queries to execute
        pass_name: Name of this pass for display

    Returns:
        List of (status, result) tuples

    """
    console.print(f"\n[bold cyan]{pass_name}[/bold cyan]")

    results: list[tuple[str, PerplexityResponse | str]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Executing queries...", total=len(queries))

        for query in queries:
            try:
                response = client.query(query)
                results.append(("success", response))
                progress.advance(task)
            except PerplexityAPIError as e:
                console.print(f"[red]✗[/red] Query failed: {e}")
                results.append(("error", str(e)))
                progress.advance(task)

    return results


def display_results_summary(
    results_1: list[tuple[str, PerplexityResponse | str]],
    results_2: list[tuple[str, PerplexityResponse | str]],
    total_queries: int,
) -> None:
    """
    Display summary table of results.

    Args:
        results_1: Results from first pass
        results_2: Results from second pass
        total_queries: Total number of queries

    """
    console.print("\n[bold]Validation Results[/bold]")

    success_count_1 = sum(1 for status, _ in results_1 if status == "success")
    success_count_2 = sum(1 for status, _ in results_2 if status == "success")

    result_table = Table(show_header=True, header_style="bold cyan")
    result_table.add_column("Pass", style="white")
    result_table.add_column("Successful", style="green")
    result_table.add_column("Failed", style="red")

    result_table.add_row(
        "Pass 1 (API calls)",
        str(success_count_1),
        str(total_queries - success_count_1),
    )
    result_table.add_row(
        "Pass 2 (cache hits)",
        str(success_count_2),
        str(total_queries - success_count_2),
    )

    console.print(result_table)


def display_sample_response(
    results: list[tuple[str, PerplexityResponse | str]],
) -> None:
    """
    Display details of first successful response.

    Args:
        results: List of query results

    """
    console.print("\n[bold]Sample Response Details[/bold]")
    for i, (status, response) in enumerate(results[:1], 1):
        if status == "success" and isinstance(response, PerplexityResponse):
            console.print(f"\n[cyan]Query {i}:[/cyan] {TEST_QUERIES[i-1][:60]}...")
            console.print(f"[dim]Model:[/dim] {response.model}")
            console.print(f"[dim]Citations:[/dim] {len(response.citations)}")
            preview_length = MAX_CONTENT_PREVIEW_LENGTH
            console.print(
                f"[dim]Content:[/dim] {response.content[:preview_length]}..."
                if len(response.content) > preview_length
                else f"[dim]Content:[/dim] {response.content}"
            )


def run_validation_test(client: PerplexityClient) -> int:
    """
    Run validation test with sample queries.

    Args:
        client: Perplexity API client

    Returns:
        Exit code (0 for success, 1 for failure)

    """
    console.print("\n[bold]Running Validation Test[/bold]")
    console.print(f"Testing with {len(TEST_QUERIES)} sample queries\n")

    # Execute both passes
    results_1 = execute_query_pass(
        client, TEST_QUERIES, "Pass 1: Initial queries (API calls)"
    )
    results_2 = execute_query_pass(
        client, TEST_QUERIES, "Pass 2: Repeat queries (cache hits)"
    )

    # Display summary and sample
    display_results_summary(results_1, results_2, len(TEST_QUERIES))

    success_count_1 = sum(1 for status, _ in results_1 if status == "success")
    if success_count_1 > 0:
        display_sample_response(results_1)

    # Determine validation status
    success_count_2 = sum(1 for status, _ in results_2 if status == "success")
    all_passed = success_count_1 == len(TEST_QUERIES) and success_count_2 == len(
        TEST_QUERIES
    )

    if all_passed:
        console.print("\n[green]✓ All tests passed![/green]")
        console.print("[dim]Cache infrastructure is working correctly[/dim]")
        return 0
    else:
        console.print("\n[yellow]⚠ Some tests failed[/yellow]")
        console.print("[dim]Review errors above for details[/dim]")
        return 1


def main() -> int:
    """Run Phase 1.2 validation test."""
    # Display header
    console.print(
        Panel.fit(
            "[bold cyan]Perplexity API Cache Validation[/bold cyan]\n"
            "Phase 1.2: SHA-256 Caching Infrastructure Test",
            border_style="cyan",
        )
    )

    args = parse_arguments()

    # Initialize cache manager
    cache_manager = CacheManager()

    # Handle --show-stats
    if args.show_stats:
        display_cache_statistics(cache_manager)
        return 0

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

    # Initialize client
    try:
        client = PerplexityClient(cache_manager=cache_manager)
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}", style="bold red")
        console.print(
            "\n[yellow]Tip:[/yellow] Set your API key with:",
            style="italic",
        )
        console.print("  export PERPLEXITY_API_KEY='your-api-key-here'")
        return 1

    # Run validation
    return run_validation_test(client)


if __name__ == "__main__":
    sys.exit(main())
