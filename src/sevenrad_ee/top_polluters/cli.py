"""
CLI for VIIRS Top Polluters feature.

Identifies top light emitters using VIIRS DNB satellite data
with optional enrichment from Google Maps APIs.
"""

import asyncio
import logging
import re
from datetime import date
from pathlib import Path

import click
import ee
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from sevenrad_ee.data.js_config_parser import (
    KnownRegion,
    load_viirs_config,
)

from . import enrichment, export
from .attribution import HIGH_LIKELIHOOD_THRESHOLD, scan_pixel_for_attribution
from .cache import cache
from .earth_engine import get_top_emitters
from .models import Business, TopEmitter

console = Console()
logger = logging.getLogger(__name__)


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
        "\n[dim]Use --regions to select: e.g., --regions drc or --regions moerkapelle[/dim]\n"  # noqa: E501
    )


def initialize_earth_engine() -> None:
    """
    Initialize Google Earth Engine API.

    Raises:
        Exception: If Earth Engine initialization fails

    """
    try:
        ee.Initialize()
    except Exception as e:
        console.print(
            f"[red]✗[/red] Failed to initialize Earth Engine: {e}", style="bold red"
        )
        console.print(
            "\n[yellow]Run:[/yellow] uv run earthengine authenticate", style="italic"
        )
        raise


@click.command(name="top-polluters")
@click.option(
    "--regions",
    type=str,
    default=None,
    help="Region code (e.g., 'drc', 'moerkapelle'). "
    "Mutually exclusive with --region-file.",
)
@click.option(
    "--region-file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    default=None,
    help="Path to GeoJSON file defining custom region. "
    "Mutually exclusive with --regions.",
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
    "--start-date",
    type=str,
    default=None,
    help="Start date in YYYY-MM-DD format",
)
@click.option(
    "--end-date",
    type=str,
    default=None,
    help="End date in YYYY-MM-DD format",
)
@click.option(
    "-n",
    "--n",
    "n_emitters",
    type=click.IntRange(1, 30),
    default=20,
    show_default=True,
    help="Number of top emitters to find.",
)
@click.option(
    "--geocode/--no-geocode",
    "geocode",
    default=True,
    show_default=True,
    help="Enrich with reverse geocoding.",
)
@click.option(
    "--businesses",
    is_flag=True,
    default=False,
    help="Find nearby businesses.",
)
@click.option(
    "--streetview",
    is_flag=True,
    default=False,
    help="Download Street View imagery for attributed greenhouses.",
)
@click.option(
    "--attribute-greenhouses",
    is_flag=True,
    default=False,
    help="Use AI to identify greenhouse operations causing light pollution.",
)
@click.option(
    "--max-businesses",
    type=click.IntRange(1, 20),
    default=5,
    show_default=True,
    help="Maximum businesses to analyze per pixel (for attribution).",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=Path("results.yml"),
    show_default=True,
    help="Output YAML file path.",
)
@click.option(
    "--clear-cache",
    is_flag=True,
    default=False,
    help="Clear all caches before running.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging for debugging.",
)
@click.option(
    "--images-dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=Path("images"),
    show_default=True,
    help="Directory for Street View images.",
)
def top_polluters(  # noqa: PLR0913, C901, PLR0915, PLR0912
    regions: str | None,
    region_file: Path | None,
    list_maps: bool,
    config_path: Path | None,
    start_date: str | None,
    end_date: str | None,
    n_emitters: int,
    geocode: bool,
    businesses: bool,
    streetview: bool,
    attribute_greenhouses: bool,
    max_businesses: int,
    output: Path,
    clear_cache: bool,
    verbose: bool,
    images_dir: Path,
) -> None:
    """
    Find top VIIRS DNB light emitters in a geographic region.

    Queries NOAA VIIRS DNB monthly satellite data to identify the brightest
    500m x 500m patches, with optional enrichment from Google Maps APIs.

    Specify region using either --regions (named region from config) or
    --region-file (path to custom GeoJSON file).
    """
    # Configure logging
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[RichHandler(console=console, rich_tracebacks=True)],
            force=True,
        )
        logger.info("Verbose logging enabled (DEBUG level)")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[RichHandler(console=console, show_time=False, show_path=False)],
            force=True,
        )

    logger.info("Starting VIIRS Top Polluters CLI")
    logger.debug(
        "Parameters: regions=%s, region_file=%s, n=%d, geocode=%s, businesses=%s, "
        "streetview=%s, attribute_greenhouses=%s",
        regions,
        region_file,
        n_emitters,
        geocode,
        businesses,
        streetview,
        attribute_greenhouses,
    )

    # Display header
    console.print(
        Panel.fit(
            "[bold cyan]VIIRS Top Polluters[/bold cyan]\n"
            "Identify top light emitters from satellite data",
            border_style="cyan",
        )
    )

    # Determine config file path
    config_file = config_path if config_path else Path.cwd() / "extract_geotiffs.js"

    # Handle --list-maps option
    if list_maps:
        list_available_maps(config_file if config_file.exists() else None)
        raise click.exceptions.Exit(0)

    # Validate that exactly one of --regions or --region-file is provided
    if not regions and not region_file:
        console.print(
            "[red]✗[/red] Must provide either --regions or --region-file",
            style="bold red",
        )
        console.print(
            "\n[yellow]Tip:[/yellow] Use --list-maps to see available regions"
        )
        raise click.exceptions.Exit(1)

    if regions and region_file:
        console.print(
            "[red]✗[/red] Cannot use both --regions and --region-file",
            style="bold red",
        )
        console.print(
            "\n[yellow]Tip:[/yellow] Choose one: --regions for named regions, "
            "--region-file for custom GeoJSON"
        )
        raise click.exceptions.Exit(1)

    # Validate required date parameters
    if not start_date or not end_date:
        console.print(
            "[red]✗[/red] --start-date and --end-date are required",
            style="bold red",
        )
        raise click.exceptions.Exit(1)

    # Initialize Earth Engine
    try:
        initialize_earth_engine()
    except Exception as e:
        raise click.exceptions.Exit(1) from e

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
        raise click.exceptions.Exit(1) from e

    if end < start:
        console.print(
            f"[red]✗[/red] End date {end_date} must be >= start date {start_date}",
            style="bold red",
        )
        raise click.exceptions.Exit(1)

    # Prepare region (named region or custom GeoJSON file)
    if regions:
        # Handle named region from config file
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
        regions_config, _ = load_viirs_config(config_file)

        if not regions_config:
            console.print(
                "[red]✗[/red] No regions found in config file", style="bold red"
            )
            raise click.exceptions.Exit(1)

        console.print(
            f"[green]✓[/green] Found {len(regions_config)} region(s) in config"
        )

        # Parse and validate region codes
        # Note: Unlike viirs maps, top-polluters requires a SINGLE region
        region_codes = [r.strip() for r in regions.split(",")]
        if len(region_codes) > 1:
            console.print(
                "[red]✗[/red] top-polluters requires a single region, not multiple",
                style="bold red",
            )
            console.print(
                f"\n[yellow]Tip:[/yellow] Use --regions {region_codes[0]} "
                "(process one region at a time)"
            )
            raise click.exceptions.Exit(1)

        region_code = region_codes[0]
        if region_code not in regions_config:
            console.print(
                f"[red]✗[/red] Unknown region: {region_code}",
                style="bold red",
            )
            available = ", ".join(sorted(regions_config.keys()))
            console.print(f"\n[yellow]Available in config:[/yellow] {available}")
            console.print("\n[dim]Run with --list-maps to see full descriptions[/dim]")
            raise click.exceptions.Exit(1)

        # Convert RegionConfig to ee.Geometry
        region_config = regions_config[region_code]
        region_geometry = ee.Geometry.Rectangle(region_config.to_ee_rectangle())
        region_name = region_code  # For display/logging
    else:
        # Use custom GeoJSON file (existing behavior)
        # region_file is guaranteed to be Path here (validated earlier)
        assert region_file is not None  # noqa: S101
        region_geometry = region_file
        region_name = region_file.stem  # For display/logging

    # Step 1: Query Earth Engine
    console.print(
        f"\n[bold]1. Querying VIIRS DNB for '{region_name}' "
        f"({start_date} to {end_date})[/bold]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading VIIRS DNB data...", total=None)
        try:
            emitters = get_top_emitters(region_geometry, start, end, n_emitters)
            progress.update(
                task, description=f"[green]✓[/green] Found {len(emitters)} emitters"
            )
        except Exception as e:
            progress.update(task, description=f"[red]✗[/red] Query failed: {e}")
            console.print(f"\n[red]Error:[/red] {e}", style="bold red")
            raise click.exceptions.Exit(1) from e

    if not emitters:
        console.print("[yellow]No emitters found in region[/yellow]")
        raise click.exceptions.Exit(0)

    # Step 2: Geocoding (optional)
    if geocode:
        console.print(f"\n[bold]2. Geocoding {len(emitters)} locations[/bold]")
        _enrich_geocoding(emitters)

    # Step 3: Business lookup (optional, but required for attribution)
    if businesses or attribute_greenhouses:
        console.print(f"\n[bold]3. Finding nearby businesses[/bold]")
        _enrich_businesses(emitters)

    # Step 4: Greenhouse Attribution (optional, AI-powered)
    if attribute_greenhouses:
        step_num = 4 if businesses or geocode else 2
        console.print(f"\n[bold]{step_num}. AI Greenhouse Attribution Analysis[/bold]")
        _run_attribution_analysis(emitters, region_name, max_businesses)

    # Step 5: Conditional Street View (optional)
    if streetview:
        if not attribute_greenhouses:
            console.print(
                "[yellow]Warning:[/yellow] --streetview requires "
                "--attribute-greenhouses to identify targets. Skipping."
            )
        else:
            step_num = 5 if businesses or geocode else 3
            console.print(
                f"\n[bold]{step_num}. Downloading Street View for "
                "high-confidence greenhouses[/bold]"
            )
            _enrich_attributed_streetview(emitters, images_dir)

    # Final Step: Export to YAML
    final_step = (
        (2 if geocode else 1)
        + (1 if businesses or attribute_greenhouses else 0)
        + (1 if attribute_greenhouses else 0)
        + (1 if streetview and attribute_greenhouses else 0)
    )
    console.print(f"\n[bold]{final_step}. Exporting to {output}[/bold]")
    try:
        export.export_yaml(emitters, output)
        console.print(f"[green]✓[/green] Exported to {output}")
    except Exception as e:
        console.print(f"[red]✗[/red] Export failed: {e}", style="bold red")
        raise click.exceptions.Exit(1) from e

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


def _enrich_attributed_streetview(emitters: list[TopEmitter], images_dir: Path) -> None:
    """Download Street View for businesses with high grow light likelihood."""
    targets: list[Business] = []
    for emitter in emitters:
        for business in emitter.businesses:
            if business.perplexity_analysis:
                likelihood = business.perplexity_analysis.get("grow_light_likelihood")
                if likelihood and likelihood >= HIGH_LIKELIHOOD_THRESHOLD:
                    targets.append(business)

    if not targets:
        console.print("[dim]No high-likelihood greenhouses found to capture.[/dim]")
        return

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Downloading imagery 0/{len(targets)}", total=len(targets)
        )
        for i, business in enumerate(targets, 1):
            if not business.coordinates:
                progress.advance(task)
                continue
            try:
                # Create a unique, filesystem-safe name for the subdirectory
                safe_name = re.sub(
                    r"[^a-zA-Z0-9_-]", "", business.name.replace(" ", "_")
                )
                sub_dir_name = f"{safe_name}_{business.place_id[:8]}"

                streetview_images = enrichment.get_street_view_images(
                    business.coordinates,
                    sub_dir_name=sub_dir_name,
                    images_dir=images_dir,
                )
                business.streetview = streetview_images
                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"Downloading imagery {i}/{len(targets)}: {business.name}"
                    ),
                )
            except Exception as e:
                console.print(
                    f"[yellow]Warning:[/yellow] Street View failed for "
                    f"{business.name}: {e}"
                )
                progress.advance(task)

    available_count = sum(
        1
        for business in targets
        if business.streetview and business.streetview.available
    )
    console.print(
        f"[green]✓[/green] Downloaded imagery for {available_count}/"
        f"{len(targets)} businesses"
    )


def _sanitize_location_context(region_name: str) -> str:
    """
    Sanitize region name to prevent prompt injection attacks.

    Allows only letters, numbers, spaces, commas, and hyphens to prevent
    malicious instructions from being injected into Perplexity prompts.

    Args:
        region_name: Raw region name from GeoJSON filename

    Returns:
        Sanitized location context safe for use in prompts

    """
    # Convert underscores to spaces and titlecase
    readable = region_name.replace("_", " ").title()
    # Remove any characters that could be used for prompt injection
    sanitized = re.sub(r"[^a-zA-Z0-9\s,\-]", "", readable)
    # Limit length to prevent extremely long inputs
    return sanitized[:100]


def _run_attribution_analysis(
    emitters: list[TopEmitter], region_name: str, max_businesses: int
) -> None:
    """
    Run AI-powered greenhouse attribution analysis on top emitters.

    Args:
        emitters: List of top emitters to analyze
        region_name: Name of region for context (e.g., "moerkapelle")
        max_businesses: Maximum number of businesses to analyze per pixel

    """
    logger.info("Starting attribution analysis for %d emitters", len(emitters))
    logger.info("Region context: %s", region_name)
    logger.info("Max businesses to analyze per pixel: %d", max_businesses)
    logger.debug("Will analyze top 5 emitters maximum to conserve API quota")

    # Sanitize location context to prevent prompt injection
    safe_location = _sanitize_location_context(region_name)
    logger.debug("Sanitized location context: %s", safe_location)

    console.print(
        "[cyan]Running AI-powered attribution analysis using Perplexity...[/cyan]\n"
    )

    async def analyze_emitters() -> None:
        for emitter in emitters[:5]:  # Analyze top 5 to conserve API quota
            logger.info(
                "Processing emitter rank %d: %.6f°N, %.6f°E (radiance: %.2f)",
                emitter.rank,
                emitter.coordinates.lat,
                emitter.coordinates.lon,
                emitter.avg_radiance,
            )
            console.print(f"\n[yellow]─[/yellow] Rank {emitter.rank}")

            # Capture attribution results (FIX: was discarding return value)
            attribution_results = await scan_pixel_for_attribution(
                pixel_center=emitter.coordinates,
                pixel_radiance=emitter.avg_radiance,
                location_context=safe_location,
                confidence_threshold=0.85,
                max_businesses_to_analyze=max_businesses,
            )

            # Convert AttributionResult → Business and attach to emitter
            if attribution_results:
                logger.info(
                    "Attaching %d attribution results to emitter rank %d",
                    len(attribution_results),
                    emitter.rank,
                )
                updated_businesses = []
                for result in attribution_results:
                    # Unpack existing Business data (including place_id)
                    business_data = result.business.model_dump()
                    # Update distance with actual pixel-to-business distance
                    business_data["distance_m"] = result.distance_from_pixel_m
                    # Add attribution confidence score
                    business_data["confidence_score"] = result.confidence_score
                    # Add raw Perplexity analysis (if available)
                    if result.perplexity_analysis:
                        business_data["perplexity_analysis"] = (
                            result.perplexity_analysis.model_dump()
                        )
                    updated_businesses.append(Business(**business_data))

                # Results already sorted by confidence (highest first)
                emitter.businesses = updated_businesses
                logger.debug(
                    "Top business: %s (place_id: %s, confidence: %.2f)",
                    updated_businesses[0].name,
                    updated_businesses[0].place_id,
                    updated_businesses[0].confidence_score or 0.0,
                )
            else:
                logger.warning(
                    "No attribution results for emitter rank %d", emitter.rank
                )

            logger.debug("Completed analysis for emitter rank %d", emitter.rank)

    try:
        logger.debug("Starting async event loop for attribution analysis...")
        asyncio.run(analyze_emitters())
        logger.info("Attribution analysis complete for all emitters")
    except KeyboardInterrupt:
        logger.warning("Attribution analysis interrupted by user")
        console.print("\n[yellow]Attribution analysis interrupted by user[/yellow]")
    except Exception as e:
        logger.exception("Attribution analysis failed with exception: %s", e)
        console.print(f"\n[red]Attribution analysis failed:[/red] {e}")
        console.print("[yellow]Tip:[/yellow] Ensure PERPLEXITY_API_KEY is set in .env")


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
    top_polluters()
