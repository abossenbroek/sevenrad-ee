"""
CLI for VIIRS Top Polluters feature.

Identifies top light emitters using VIIRS DNB satellite data
with optional enrichment from Google Maps APIs.
"""

from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import enrichment, export
from .cache import cache
from .earth_engine import get_top_emitters
from .models import TopEmitter

app = typer.Typer()
console = Console()


@app.command()
def main(  # noqa: PLR0913, C901
    region: Path = typer.Option(
        ...,
        "--region",
        help="Path to GeoJSON file defining the geographic region",
        exists=True,
        dir_okay=False,
    ),
    start_date: str = typer.Option(
        ...,
        "--start-date",
        help="Start date in YYYY-MM-DD format",
    ),
    end_date: str = typer.Option(
        ...,
        "--end-date",
        help="End date in YYYY-MM-DD format",
    ),
    n: int = typer.Option(
        20,
        "--n",
        help="Number of top emitters to find (max 30)",
        min=1,
        max=30,
    ),
    geocode: bool = typer.Option(
        True,
        "--geocode/--skip-geocode",
        help="Enrich with reverse geocoding",
    ),
    businesses: bool = typer.Option(
        False,
        "--businesses",
        is_flag=True,
        help="Find nearby businesses",
    ),
    streetview: bool = typer.Option(
        False,
        "--streetview",
        is_flag=True,
        help="Download Street View imagery",
    ),
    output: Path = typer.Option(
        Path("results.yml"),
        "--output",
        help="Output YAML file path",
    ),
    clear_cache: bool = typer.Option(
        False,
        "--clear-cache",
        is_flag=True,
        help="Clear all caches before running",
    ),
) -> None:
    """
    Find top VIIRS DNB light emitters in a geographic region.

    Queries NOAA VIIRS DNB monthly satellite data to identify the brightest
    500m x 500m patches, with optional enrichment from Google Maps APIs.
    """
    # Display header
    console.print(
        Panel.fit(
            "[bold cyan]VIIRS Top Polluters[/bold cyan]\n"
            "Identify top light emitters from satellite data",
            border_style="cyan",
        )
    )

    # Clear cache if requested
    if clear_cache:
        console.print("[yellow]Clearing cache...[/yellow]")
        cache.clear()
        console.print("[green]✓[/green] Cache cleared")

    # Parse and validate dates
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except ValueError as e:
        console.print(f"[red]✗[/red] Invalid date format: {e}", style="bold red")
        console.print(
            "\n[yellow]Tip:[/yellow] Use YYYY-MM-DD format (e.g., 2024-01-01)"
        )
        raise typer.Exit(1) from e

    if end < start:
        console.print(
            f"[red]✗[/red] End date {end_date} must be >= start date {start_date}",
            style="bold red",
        )
        raise typer.Exit(1)

    # Validate region file
    if not region.exists():
        console.print(
            f"[red]✗[/red] Region file not found: {region}",
            style="bold red",
        )
        console.print("\n[yellow]Tip:[/yellow] Provide a valid GeoJSON file path")
        raise typer.Exit(1)

    # Step 1: Query Earth Engine
    console.print(f"\n[bold]1. Querying VIIRS DNB ({start_date} to {end_date})[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading VIIRS DNB data...", total=None)
        try:
            emitters = get_top_emitters(region, start, end, n)
            progress.update(
                task, description=f"[green]✓[/green] Found {len(emitters)} emitters"
            )
        except Exception as e:
            progress.update(task, description=f"[red]✗[/red] Query failed: {e}")
            console.print(f"\n[red]Error:[/red] {e}", style="bold red")
            raise typer.Exit(1) from e

    if not emitters:
        console.print("[yellow]No emitters found in region[/yellow]")
        raise typer.Exit(0)

    # Step 2: Geocoding (optional)
    if geocode:
        console.print(f"\n[bold]2. Geocoding {len(emitters)} locations[/bold]")
        _enrich_geocoding(emitters)

    # Step 3: Business lookup (optional)
    if businesses:
        console.print(f"\n[bold]3. Finding nearby businesses[/bold]")
        _enrich_businesses(emitters)

    # Step 4: Street View (optional)
    if streetview:
        console.print(f"\n[bold]4. Downloading Street View imagery[/bold]")
        _enrich_streetview(emitters)

    # Step 5: Export to YAML
    console.print(f"\n[bold]5. Exporting to {output}[/bold]")
    try:
        export.export_yaml(emitters, output)
        console.print(f"[green]✓[/green] Exported to {output}")
    except Exception as e:
        console.print(f"[red]✗[/red] Export failed: {e}", style="bold red")
        raise typer.Exit(1) from e

    # Summary
    _display_summary(emitters)


def _enrich_geocoding(emitters: list[TopEmitter]) -> None:
    """Enrich emitters with geocoding data."""
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Geocoding 0/{len(emitters)}", total=len(emitters))

        for i, emitter in enumerate(emitters, 1):
            try:
                address = enrichment.geocode_coordinates(emitter.coordinates)
                emitter.address = address
                progress.update(
                    task,
                    advance=1,
                    description=f"Geocoding {i}/{len(emitters)}",
                )
            except Exception as e:
                console.print(
                    f"[yellow]Warning:[/yellow] Geocoding failed for rank {emitter.rank}: {e}"  # noqa: E501
                )
                progress.advance(task)

    geocoded_count = sum(1 for e in emitters if e.address is not None)
    console.print(
        f"[green]✓[/green] Geocoded {geocoded_count}/{len(emitters)} locations"
    )


def _enrich_businesses(emitters: list[TopEmitter]) -> None:
    """Enrich emitters with nearby business data."""
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Finding businesses 0/{len(emitters)}", total=len(emitters)
        )

        for i, emitter in enumerate(emitters, 1):
            try:
                businesses_list = enrichment.find_nearby_businesses(emitter.coordinates)
                emitter.businesses = businesses_list[:10]  # Top 10
                progress.update(
                    task,
                    advance=1,
                    description=f"Finding businesses {i}/{len(emitters)}",
                )
            except Exception as e:
                console.print(
                    f"[yellow]Warning:[/yellow] Business search failed for rank {emitter.rank}: {e}"  # noqa: E501
                )
                progress.advance(task)

    total_businesses = sum(len(e.businesses) for e in emitters)
    console.print(f"[green]✓[/green] Found {total_businesses} businesses")


def _enrich_streetview(emitters: list[TopEmitter]) -> None:
    """Enrich emitters with Street View imagery."""
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Downloading imagery 0/{len(emitters)}", total=len(emitters)
        )

        for i, emitter in enumerate(emitters, 1):
            try:
                streetview_images = enrichment.get_street_view_images(
                    emitter.coordinates, emitter.rank
                )
                emitter.streetview = streetview_images
                progress.update(
                    task,
                    advance=1,
                    description=f"Downloading imagery {i}/{len(emitters)}",
                )
            except Exception as e:
                console.print(
                    f"[yellow]Warning:[/yellow] Street View failed for rank {emitter.rank}: {e}"  # noqa: E501
                )
                progress.advance(task)

    available_count = sum(
        1 for e in emitters if e.streetview and e.streetview.available
    )
    console.print(
        f"[green]✓[/green] Downloaded imagery for {available_count}/{len(emitters)} locations"  # noqa: E501
    )


def _display_summary(emitters: list[TopEmitter]) -> None:
    """Display summary table of results."""
    console.print("\n[bold cyan]Summary[/bold cyan]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Rank", style="green", width=6)
    table.add_column("Coordinates", style="white")
    table.add_column("Radiance", style="yellow")
    table.add_column("Address", style="cyan")

    for emitter in emitters[:10]:  # Show top 10
        address_str = (
            emitter.address.formatted[:40] + "..."
            if emitter.address and len(emitter.address.formatted) > 40  # noqa: PLR2004
            else (emitter.address.formatted if emitter.address else "N/A")
        )

        table.add_row(
            str(emitter.rank),
            f"{emitter.coordinates.lat:.4f}, {emitter.coordinates.lon:.4f}",
            f"{emitter.avg_radiance:.2f}",
            address_str,
        )

    console.print(table)

    if len(emitters) > 10:  # noqa: PLR2004
        console.print(f"\n[dim]... and {len(emitters) - 10} more emitters[/dim]")


if __name__ == "__main__":
    app()
