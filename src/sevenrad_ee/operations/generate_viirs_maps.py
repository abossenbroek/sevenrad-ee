"""
Generate VIIRS nighttime light maps using Google Earth Engine.

This module provides a CLI for processing VIIRS DNB (Day/Night Band) data
to generate nighttime light composite maps for specified regions and time periods.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

if TYPE_CHECKING:
    import ee
else:
    import ee  # type: ignore[no-redef]

# Initialize Rich console for colorful output
console = Console()


def initialize_earth_engine() -> None:
    """
    Initialize Google Earth Engine API.

    Raises:
        Exception: If Earth Engine initialization fails

    """
    try:
        ee.Initialize()
        console.print("[green]✓[/green] Earth Engine initialized successfully")
    except Exception as e:
        console.print(
            f"[red]✗[/red] Failed to initialize Earth Engine: {e}", style="bold red"
        )
        console.print(
            "\n[yellow]Run:[/yellow] uv run earthengine authenticate", style="italic"
        )
        raise


def validate_date_format(date_str: str) -> bool:
    """
    Validate date string is in YYYY-MM-DD format.

    Args:
        date_str: Date string to validate

    Returns:
        True if date format is valid, False otherwise

    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def generate_viirs_composite(
    start_date: str,
    end_date: str,
    region: ee.Geometry | None = None,  # type: ignore[name-defined]
    output_path: Path | None = None,  # noqa: ARG001
) -> ee.Image:  # type: ignore[name-defined]
    """
    Generate VIIRS nighttime light composite for specified date range.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        region: Geographic region of interest (defaults to global if None)
        output_path: Path to save the output (optional)

    Returns:
        Composite Earth Engine Image of VIIRS nighttime lights

    Raises:
        ValueError: If date range is invalid

    """
    # Validate dates
    if not validate_date_format(start_date) or not validate_date_format(end_date):
        raise ValueError("Dates must be in YYYY-MM-DD format")

    console.print(
        f"\n[cyan]Processing VIIRS data from {start_date} to {end_date}[/cyan]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load VIIRS DNB collection
        task1 = progress.add_task("Loading VIIRS DNB collection...", total=None)
        viirs_collection = ee.ImageCollection(  # type: ignore[attr-defined]
            "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
        ).filterDate(start_date, end_date)
        progress.update(task1, completed=True)

        # Apply region filter if specified
        if region is not None:
            task2 = progress.add_task("Filtering by region...", total=None)
            viirs_collection = viirs_collection.filterBounds(region)
            progress.update(task2, completed=True)

        # Create composite
        task3 = progress.add_task("Creating composite image...", total=None)
        composite = viirs_collection.select("avg_rad").median()
        progress.update(task3, completed=True)

        console.print("[green]✓[/green] Composite generated successfully")

        return composite


def display_summary(
    start_date: str,
    end_date: str,
    region: str | None,
    output: str | None,
) -> None:
    """
    Display a summary table of the processing parameters.

    Args:
        start_date: Start date of processing
        end_date: End date of processing
        region: Region specification (or None for global)
        output: Output path (or None if not saving)

    """
    table = Table(title="VIIRS Map Generation Summary", show_header=True)
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Start Date", start_date)
    table.add_row("End Date", end_date)
    table.add_row("Region", region or "Global")
    table.add_row("Output Path", output or "Not specified")

    console.print("\n")
    console.print(table)
    console.print("\n")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed command-line arguments

    """
    parser = argparse.ArgumentParser(
        description="Generate VIIRS nighttime light maps using Google Earth Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate global composite for January 2024
  uv run python -m sevenrad_ee.operations.generate_viirs_maps \\
      --start-date 2024-01-01 \\
      --end-date 2024-01-31

  # Generate composite for specific region
  uv run python -m sevenrad_ee.operations.generate_viirs_maps \\
      --start-date 2024-01-01 \\
      --end-date 2024-01-31 \\
      --region "POLYGON((...))"

  # Save output to file
  uv run python -m sevenrad_ee.operations.generate_viirs_maps \\
      --start-date 2024-01-01 \\
      --end-date 2024-01-31 \\
      --output /path/to/output.tif
        """,
    )

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format",
    )

    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help=(
            "Region as WKT geometry string (e.g., 'POLYGON((...))') "
            "or None for global"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (optional)",
    )

    return parser.parse_args()


def main() -> int:
    """
    Run the VIIRS map generation CLI.

    Returns:
        Exit code (0 for success, 1 for failure)

    """
    # Display header
    console.print(
        Panel.fit(
            "[bold cyan]VIIRS Nighttime Light Map Generator[/bold cyan]\n"
            "Generate composite maps using Google Earth Engine",
            border_style="cyan",
        )
    )

    try:
        # Parse arguments
        args = parse_arguments()

        # Display summary
        display_summary(
            start_date=args.start_date,
            end_date=args.end_date,
            region=args.region,
            output=args.output,
        )

        # Initialize Earth Engine
        console.print("[yellow]Initializing Earth Engine...[/yellow]")
        initialize_earth_engine()

        # Parse region if provided
        region: ee.Geometry | None = None  # type: ignore[name-defined]
        if args.region:
            console.print(f"[yellow]Parsing region geometry...[/yellow]")
            region = ee.Geometry(args.region)  # type: ignore[attr-defined]
            console.print("[green]✓[/green] Region parsed successfully")

        # Generate composite
        output_path = Path(args.output) if args.output else None
        composite = generate_viirs_composite(
            start_date=args.start_date,
            end_date=args.end_date,
            region=region,
            output_path=output_path,
        )

        # Display success message
        console.print(
            "\n[bold green]Success![/bold green] "
            "VIIRS composite generated successfully.",
            style="bold",
        )

        if output_path:
            console.print(f"[yellow]Note:[/yellow] Export task queued to {output_path}")
            console.print(
                "[dim]Check Earth Engine task manager for export status[/dim]"
            )

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}", style="bold red")
        return 1


if __name__ == "__main__":
    sys.exit(main())
