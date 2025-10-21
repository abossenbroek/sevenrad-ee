"""
Generate VIIRS nighttime light maps using Google Earth Engine.

This module provides a CLI for processing VIIRS DNB (Day/Night Band) data
to generate nighttime light composite maps for specified regions and time periods.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from sevenrad_ee.data.js_config_parser import (
    KnownRegion,
    RegionConfig,
    load_viirs_config,
)

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
    region_name: str,
    region: RegionConfig,
) -> ee.Image:  # type: ignore[name-defined]
    """
    Generate VIIRS nighttime light composite for specified date range and region.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        region_name: Name of the region being processed
        region: Region configuration with bounds

    Returns:
        Composite Earth Engine Image of VIIRS nighttime lights

    Raises:
        ValueError: If date range is invalid

    """
    # Validate dates
    if not validate_date_format(start_date) or not validate_date_format(end_date):
        raise ValueError("Dates must be in YYYY-MM-DD format")

    console.print(
        f"\n[cyan]Processing '{region_name}' from {start_date} to {end_date}[/cyan]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Create region geometry
        task_region = progress.add_task(f"Creating region geometry...", total=None)
        ee_region = ee.Geometry.Rectangle(  # type: ignore[attr-defined]
            region.to_ee_rectangle()
        )
        progress.update(task_region, completed=True)

        # Load VIIRS DNB collection
        task1 = progress.add_task("Loading VIIRS DNB collection...", total=None)
        viirs_collection = ee.ImageCollection(  # type: ignore[attr-defined]
            "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
        ).filterDate(start_date, end_date)
        progress.update(task1, completed=True)

        # Apply region filter
        task2 = progress.add_task("Filtering by region...", total=None)
        viirs_collection = viirs_collection.filterBounds(ee_region)
        progress.update(task2, completed=True)

        # Create composite
        task3 = progress.add_task("Creating composite image...", total=None)
        composite = viirs_collection.select("avg_rad").median()
        progress.update(task3, completed=True)

        console.print(f"[green]✓[/green] Composite for '{region_name}' generated")

        return composite


def list_available_maps(config_file: Path | None = None) -> None:
    """
    Display available map regions with descriptions.

    Args:
        config_file: Optional path to config file to show actual availability

    """
    console.print("\n[bold cyan]Available Map Regions:[/bold cyan]\n")

    # Show known regions with display names
    display_names = KnownRegion.display_names()
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Region Code", style="green", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Status", style="yellow")

    # If config file provided, check which are actually available
    available_regions = set()
    if config_file and config_file.exists():
        regions, _ = load_viirs_config(config_file)
        available_regions = set(regions.keys())

    for region_code in KnownRegion.all_regions():
        description = display_names.get(region_code, region_code)
        if config_file:
            status = (
                "✓ Available" if region_code in available_regions else "✗ Not in config"
            )
        else:
            status = "Check config file"
        table.add_row(region_code, description, status)

    console.print(table)
    console.print(
        "\n[dim]Use --maps to select: e.g., --maps drc,us or --maps all[/dim]\n"
    )


def display_summary(
    start_date: str,
    end_date: str,
    maps: list[str],
    config_file: Path,
    output_dir: Path | None,
) -> None:
    """
    Display a summary table of the processing parameters.

    Args:
        start_date: Start date of processing
        end_date: End date of processing
        maps: List of map names to process
        config_file: Path to configuration file
        output_dir: Output directory (or None if not saving)

    """
    table = Table(title="VIIRS Map Generation Summary", show_header=True)
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Start Date", start_date)
    table.add_row("End Date", end_date)
    table.add_row("Maps", ", ".join(maps))
    table.add_row("Config File", str(config_file))
    table.add_row("Output Directory", str(output_dir) if output_dir else "Not saving")

    console.print("\n")
    console.print(table)
    console.print("\n")


@click.command(name="maps")
@click.option(
    "--start-date",
    type=str,
    default=None,
    help="Start date (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    type=str,
    default=None,
    help="End date (YYYY-MM-DD)",
)
@click.option(
    "--regions",
    type=str,
    default=None,
    help="Region codes: 'all' or comma-separated",
)
@click.option(
    "--list-maps",
    is_flag=True,
    default=False,
    help="List available map regions with descriptions and exit",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=None,
    help="Path to JavaScript config file (default: ./extract_geotiffs.js)",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory for saving results",
)
def maps(  # noqa: C901, PLR0912, PLR0915, PLR0913
    start_date: str | None,
    end_date: str | None,
    regions: str | None,
    list_maps: bool,
    config_path: Path | None,
    output_dir: Path | None,
) -> None:
    r"""
    Generate VIIRS DNB composite maps for specified regions and time periods.

    Examples:
      # List available maps
      uv run viirs maps --list-maps

      # Generate all maps for 2020
      uv run viirs maps --start-date 2020-01-01 \
          --end-date 2020-12-31 --maps all

      # Generate specific maps
      uv run viirs maps --start-date 2020-01-01 \
          --end-date 2020-12-31 --maps us,europe

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
        # Determine config file path
        config_file = config_path if config_path else Path.cwd() / "extract_geotiffs.js"

        # Handle --list-maps option
        if list_maps:
            list_available_maps(config_file if config_file.exists() else None)
            raise click.exceptions.Exit(0)

        # Validate required arguments when not listing maps
        if not start_date or not end_date or not regions:
            console.print(
                "[red]✗[/red] --start-date, --end-date, and --regions are required",
                style="bold red",
            )
            console.print(
                "\n[yellow]Tip:[/yellow] Use --list-maps to see available regions"
            )
            raise click.exceptions.Exit(1)

        if not config_file.exists():
            console.print(
                f"[red]✗[/red] Config file not found: {config_file}",
                style="bold red",
            )
            console.print(
                "\n[yellow]Tip:[/yellow] Specify config file with --config option"
            )
            raise click.exceptions.Exit(1)

        # Load configuration
        console.print(f"[yellow]Loading configuration from {config_file}...[/yellow]")
        regions_config, vis_config = load_viirs_config(config_file)

        if not regions_config:
            console.print(
                "[red]✗[/red] No regions found in config file", style="bold red"
            )
            raise click.exceptions.Exit(1)

        console.print(
            f"[green]✓[/green] Found {len(regions_config)} region(s) in config"
        )

        # Determine which maps to process
        if regions.lower() == "all":
            maps_to_process = list(regions_config.keys())
        else:
            maps_to_process = [m.strip() for m in regions.split(",")]

            # Validate map names
            invalid_maps = [m for m in maps_to_process if m not in regions_config]
            if invalid_maps:
                console.print(
                    f"[red]✗[/red] Invalid map name(s): {', '.join(invalid_maps)}",
                    style="bold red",
                )
                available = ", ".join(sorted(regions_config.keys()))
                console.print(f"\n[yellow]Available in config:[/yellow] {available}")
                console.print(
                    "\n[dim]Run with --list-maps to see full descriptions[/dim]"
                )
                raise click.exceptions.Exit(1)

        # Prepare output directory
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Display summary
        display_summary(
            start_date=start_date,
            end_date=end_date,
            maps=maps_to_process,
            config_file=config_file,
            output_dir=output_dir,
        )

        # Initialize Earth Engine
        console.print("[yellow]Initializing Earth Engine...[/yellow]")
        initialize_earth_engine()

        # Process each map
        console.print(
            f"\n[bold cyan]Processing {len(maps_to_process)} map(s)...[/bold cyan]\n"
        )

        for map_name in maps_to_process:
            region_config = regions_config[map_name]
            composite = generate_viirs_composite(
                start_date=start_date,
                end_date=end_date,
                region_name=map_name,
                region=region_config,
            )

            # TODO: Add export functionality when output_dir is specified
            if output_dir:
                console.print(
                    f"[dim]Note: Export to {output_dir} not yet implemented[/dim]"
                )

        # Display success message
        console.print(
            f"\n[bold green]Success![/bold green] "
            f"Generated {len(maps_to_process)} VIIRS composite(s).",
            style="bold",
        )

        if vis_config:
            console.print(
                f"\n[dim]Visualization: min={vis_config.min_value}, "
                f"max={vis_config.max_value}, "
                f"palette colors={len(vis_config.palette)}[/dim]"
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise click.exceptions.Exit(1) from None
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}", style="bold red")
        import traceback

        console.print(f"\n[dim]{traceback.format_exc()}[/dim]")
        raise click.exceptions.Exit(1) from e
