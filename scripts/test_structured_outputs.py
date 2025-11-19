"""
Test script for Perplexity structured outputs.

This script validates that the refactored PerplexityClient correctly uses
Perplexity's native response_format parameter with Pydantic models.

Usage:
    uv run python scripts/test_structured_outputs.py
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sevenrad_ee.ai.perplexity_cache import PerplexityAPIConfig
from sevenrad_ee.ai.perplexity_client import PerplexityClient

console = Console()
logging.basicConfig(level=logging.INFO)


# Define a simple Pydantic model for greenhouse classification
class SimplifiedGreenhouseAnalysis(BaseModel):
    """Simplified greenhouse analysis with structured output."""

    is_greenhouse: bool = Field(
        ...,
        description="Is this location an actual greenhouse facility?",
    )
    uses_growlight: Literal["YES", "NO", "UNKNOWN"] = Field(
        ...,
        description=(
            "Does this greenhouse use artificial lighting? "
            "Must be YES, NO, or UNKNOWN."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classification confidence from 0.0 to 1.0",
    )
    reasoning: str = Field(
        ...,
        description="Step-by-step reasoning for the classification",
    )


def test_structured_output() -> int:
    """
    Test Perplexity structured output with SimplifiedGreenhouseAnalysis.

    Returns:
        Exit code (0 for success, 1 for failure)

    """
    console.print(
        Panel.fit(
            "[bold cyan]Perplexity Structured Output Test[/bold cyan]\n"
            "Validating native response_format parameter with Pydantic",
            border_style="cyan",
        )
    )

    # Initialize client
    try:
        client = PerplexityClient()
        console.print("[green]✓[/green] PerplexityClient initialized")
    except ValueError as e:
        console.print(f"[red]✗[/red] Failed to initialize client: {e}")
        console.print(
            "\n[yellow]Set PERPLEXITY_API_KEY environment variable[/yellow]"
        )
        return 1

    # Test query with structured output
    test_query = (
        "Analyze Marjoland in Waddinxveen, Netherlands. "
        "Is it a greenhouse facility? Does it use artificial grow lights?"
    )

    console.print(f"\n[cyan]Test Query:[/cyan] {test_query[:80]}...")

    try:
        # Configure Perplexity API
        config = PerplexityAPIConfig(
            model="sonar-pro",  # Use sonar-pro for structured output support
            temperature=0.0,
            max_tokens=1500,
        )

        console.print("\n[yellow]Executing structured query...[/yellow]")
        console.print(
            "[dim]Note: First request may take 10-30s for schema compilation[/dim]\n"
        )

        # Execute query with structured output
        result = client.query(
            query=test_query,
            config=config,
            response_model=SimplifiedGreenhouseAnalysis,
        )

        # Validate result is correctly typed
        if not isinstance(result, SimplifiedGreenhouseAnalysis):
            console.print(
                f"[red]✗[/red] Expected SimplifiedGreenhouseAnalysis, "
                f"got {type(result).__name__}"
            )
            return 1

        console.print("[green]✓[/green] Structured output received!\n")

        # Display results in table
        results_table = Table(
            title="Structured Output Results",
            show_header=True,
            header_style="bold cyan",
        )
        results_table.add_column("Field", style="cyan", width=20)
        results_table.add_column("Value", style="white")

        results_table.add_row("is_greenhouse", str(result.is_greenhouse))
        results_table.add_row("uses_growlight", result.uses_growlight)
        results_table.add_row("confidence", f"{result.confidence:.2%}")
        results_table.add_row(
            "reasoning",
            result.reasoning[:100] + "..."
            if len(result.reasoning) > 100
            else result.reasoning,
        )

        console.print(results_table)

        # Validate Pydantic constraints
        validation_table = Table(
            title="Pydantic Validation Checks",
            show_header=True,
            header_style="bold cyan",
        )
        validation_table.add_column("Check", style="cyan")
        validation_table.add_column("Status", justify="center")

        # Check 1: is_greenhouse is bool
        is_bool = isinstance(result.is_greenhouse, bool)
        validation_table.add_row(
            "is_greenhouse is bool",
            "[green]✓[/green]" if is_bool else "[red]✗[/red]",
        )

        # Check 2: uses_growlight is Literal
        is_literal = result.uses_growlight in ["YES", "NO", "UNKNOWN"]
        validation_table.add_row(
            "uses_growlight is Literal['YES', 'NO', 'UNKNOWN']",
            "[green]✓[/green]" if is_literal else "[red]✗[/red]",
        )

        # Check 3: confidence in range [0, 1]
        in_range = 0.0 <= result.confidence <= 1.0
        validation_table.add_row(
            "confidence in [0.0, 1.0]",
            "[green]✓[/green]" if in_range else "[red]✗[/red]",
        )

        console.print("\n")
        console.print(validation_table)

        if all([is_bool, is_literal, in_range]):
            console.print(
                "\n[bold green]✓ All validation checks passed![/bold green]"
            )
            console.print(
                "[green]Perplexity structured outputs working correctly[/green]"
            )
            return 0
        else:
            console.print(
                "\n[bold red]✗ Some validation checks failed[/bold red]"
            )
            return 1

    except Exception as e:
        console.print(f"\n[red]✗[/red] Test failed: {e}")
        logging.exception("Structured output test failed")
        return 1


def main() -> int:
    """
    Run structured output test.

    Returns:
        Exit code

    """
    return test_structured_output()


if __name__ == "__main__":
    import sys

    sys.exit(main())
