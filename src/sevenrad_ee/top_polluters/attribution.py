"""
Greenhouse attribution using Google Places + Perplexity AI.

This module implements a novel algorithm to identify greenhouse operations
responsible for VIIRS nighttime light pollution by combining:
1. Geospatial business search (Google Places API)
2. AI-powered lighting practice analysis (Perplexity AI)
3. Multi-factor confidence scoring

The scanner uses a single-radius search optimization and structured JSON
prompts to minimize API calls while maximizing attribution accuracy.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from perplexity import AsyncPerplexity
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .cache import cache
from .config import settings
from .enrichment import find_nearby_businesses, haversine_distance
from .models import Business, Coordinates

console = Console()
logger = logging.getLogger(__name__)

# Constants for confidence scoring
HIGH_LIKELIHOOD_THRESHOLD = 0.7  # Threshold for "high likelihood" status display


# Pydantic models for structured Perplexity responses


class PerplexityAnalysis(BaseModel):
    """
    Structured analysis of greenhouse operation from Perplexity AI.

    This model is used to parse and validate JSON output from the LLM,
    ensuring we handle None values gracefully when data is unavailable.
    """

    grow_light_likelihood: Optional[float] = None  # 0.0-1.0 probability
    lighting_evidence: Optional[str] = None
    primary_crops: list[str] = []
    size_hectares: Optional[float] = None
    instagram_handle: Optional[str] = None  # Instagram account (without @)
    sources: list[str] = []


@dataclass
class AttributionResult:
    """
    Result of attributing a VIIRS pixel to a specific greenhouse operation.

    Contains the business information, Perplexity analysis, confidence score,
    and detailed scoring factors for transparency.
    """

    business: Business
    distance_from_pixel_m: float
    perplexity_analysis: Optional[PerplexityAnalysis]
    confidence_score: float
    confidence_factors: dict[str, float] = field(default_factory=dict)
    api_error: Optional[str] = None


class PerplexityGreenhouseAnalyzer:
    """
    Analyzer using official Perplexity AI SDK for greenhouse operations.

    This class handles:
    - Structured JSON prompt construction
    - Robust JSON parsing with markdown cleaning
    - Retry logic with exponential backoff
    - Rate limit handling
    """

    def __init__(self, api_key: str) -> None:
        """
        Initialize Perplexity analyzer.

        Args:
            api_key: Perplexity API key (or use PERPLEXITY_API_KEY env var)

        """
        self.client = AsyncPerplexity(api_key=api_key)

    @staticmethod
    def _clean_json_string(raw_string: str) -> str:
        """
        Extract JSON object from markdown-wrapped or mixed-content response.

        The LLM sometimes wraps JSON in ```json``` blocks or adds commentary.
        This function extracts just the JSON object.

        Args:
            raw_string: Raw response from Perplexity

        Returns:
            Cleaned JSON string or empty string if no JSON found

        """
        # Find first '{' and last '}'
        start_idx = raw_string.find("{")
        end_idx = raw_string.rfind("}")

        if start_idx == -1 or end_idx == -1:
            return ""

        return raw_string[start_idx : end_idx + 1]

    async def analyze_business(
        self,
        business: Business,
        location_context: str,
        max_retries: int = 3,
    ) -> Optional[PerplexityAnalysis]:
        """
        Analyze greenhouse operation using Perplexity AI with structured JSON.

        Uses official perplexityai SDK with sonar-small-32k-online model
        for real-time web search and factual data retrieval.

        Args:
            business: Business to analyze from Google Places
            location_context: Location context (e.g., "Moerkapelle, Netherlands")
            max_retries: Maximum retry attempts for API errors

        Returns:
            PerplexityAnalysis model or None if analysis fails

        """
        logger.info(
            "Starting Perplexity analysis for business: %s (place_id: %s)",
            business.name,
            business.place_id,
        )
        logger.debug("Business types: %s", business.types)
        logger.debug("Location context: %s", location_context)

        # Check cache first
        cache_key = f"perplexity_greenhouse:{business.place_id}"
        cached = cache.get(cache_key)
        if cached:
            logger.info("Cache HIT for business: %s", business.name)
            return cached  # type: ignore[no-any-return]

        logger.info(
            "Cache MISS for business: %s - querying Perplexity API", business.name
        )

        # Construct structured JSON prompt
        business_info = business.name
        location_info = location_context
        prompt = (
            f"Analyze greenhouse operation: {business_info} near {location_info}.\n\n"
            "Focus ONLY on verifiable information from company websites, "
            "agricultural databases, or industry reports.\n\n"
            "For grow_light_likelihood: Estimate the probability (0.0-1.0) that this "
            "operation uses artificial grow lights based on:\n"
            "- Crop type (flowers like gerbera, roses require high light; "
            "tomatoes, peppers need supplemental lighting)\n"
            "- Explicit mentions of LED/HPS lighting, controlled "
            "environment, or year-round production\n"
            "- DO NOT speculate about schedules or practices unless "
            "explicitly stated\n\n"
            "For lighting_evidence: Quote or paraphrase ONLY factual "
            "statements from sources. Do not infer practices.\n\n"
            "For instagram_handle: Find official Instagram account if available "
            "(return handle WITHOUT @ symbol, "
            "e.g., 'summitgerbera' not '@summitgerbera').\n\n"
            "Respond ONLY with a JSON object using this schema. "
            "If information is not found, use null.\n"
            "Do not add commentary outside the JSON block.\n\n"
            "{\n"
            '  "grow_light_likelihood": float,  // 0.0-1.0 probability\n'
            '  "lighting_evidence": "...",  // FACTUAL quotes only, or null\n'
            '  "primary_crops": ["...", "..."],\n'
            '  "size_hectares": float,  // or null\n'
            '  "instagram_handle": "...",  // Without @ symbol, or null\n'
            '  "sources": ["...", "..."]  // URLs only\n'
            "}"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Agricultural operations analyst. "
                    "Respond only with requested JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            logger.debug(
                "Perplexity API call attempt %d/%d for business: %s",
                attempt + 1,
                max_retries,
                business.name,
            )

            try:
                logger.debug("Sending request to Perplexity sonar-pro model...")
                response = await self.client.chat.completions.create(
                    model="sonar-pro",
                    messages=messages,  # type: ignore[arg-type]
                )

                logger.debug("Received response from Perplexity API")
                raw_content = response.choices[0].message.content
                logger.debug(
                    "Raw response length: %d characters", len(str(raw_content))
                )
                logger.debug("Raw response preview: %s...", raw_content[:200])

                cleaned_json = self._clean_json_string(str(raw_content))

                if not cleaned_json:
                    logger.warning(
                        "No JSON in Perplexity response for business: %s",
                        business.name,
                    )
                    logger.debug("Full raw response: %s", raw_content)
                    return None

                logger.debug("Extracted JSON: %s", cleaned_json[:300])

                # Parse with Pydantic for validation
                logger.debug("Validating JSON with Pydantic schema...")
                analysis = PerplexityAnalysis.model_validate_json(cleaned_json)

                logger.info("✓ Successfully analyzed business: %s", business.name)
                if analysis.grow_light_likelihood is not None:
                    logger.info(
                        "  - Grow light likelihood: %.2f%%",
                        analysis.grow_light_likelihood * 100,
                    )
                logger.info("  - Crops: %s", analysis.primary_crops)
                logger.info("  - Size: %s hectares", analysis.size_hectares)
                logger.info("  - Sources: %d found", len(analysis.sources))

                # Cache successful result
                cache.set(cache_key, analysis)
                logger.debug("Cached analysis result for future requests")

                return analysis

            except ValidationError as e:
                logger.error(
                    "JSON validation failed for business %s: %s", business.name, e
                )
                logger.debug("Raw content was: %s", raw_content[:500])
                await asyncio.sleep(1 * (attempt + 1))
                continue  # Retry

            except Exception as e:
                # Catch all API errors (rate limits, connection errors, etc.)
                wait_time = 2 ** (attempt + 1)  # Exponential backoff
                logger.warning(
                    "API error for business %s (attempt %d/%d): %s. " "Retrying in %ds",
                    business.name,
                    attempt + 1,
                    max_retries,
                    e,
                    wait_time,
                )
                await asyncio.sleep(wait_time)

        logger.error(
            "Failed to analyze business %s after %d retries", business.name, max_retries
        )
        return None


def calculate_confidence(  # noqa: PLR0915
    perplexity: Optional[PerplexityAnalysis],
    distance_m: float,
    business_types: list[str],
    max_distance: float = 1000,
) -> tuple[float, dict[str, float]]:
    """
    Calculate attribution confidence using improved multi-factor algorithm.

    Uses continuous distance decay (not step functions) and treats night
    lighting as a veto condition. Perplexity signals act as multipliers
    on the base score.

    Args:
        perplexity: Perplexity analysis result (None if failed)
        distance_m: Distance from VIIRS pixel center in meters
        business_types: List of business types from Google Places
        max_distance: Maximum search distance (default 1000m)

    Returns:
        Tuple of (total_confidence_score, factor_breakdown_dict)

    """
    logger.debug("Calculating confidence score...")
    logger.debug("  Distance: %.1fm", distance_m)
    logger.debug("  Business types: %s", business_types)

    factors: dict[str, float] = {}

    # NO VETO - allow all businesses to contribute proportionally
    if perplexity is None:
        logger.info("No Perplexity analysis - confidence = 0.0")
        return 0.0, {"analysis_failed": 0.0}

    # CONTINUOUS DISTANCE SCORE (Zen's recommendation: not step function)
    distance_score = 0.3 * (1 - (distance_m / max_distance))
    factors["distance"] = round(distance_score, 3)

    # BUSINESS TYPE SCORE
    greenhouse_types = {"greenhouse", "nursery", "horticulture"}
    agri_types = {"agricultural", "farming", "agri"}

    if any(t.lower() in greenhouse_types for t in business_types):
        type_score = 0.20
    elif any(t.lower() in agri_types for t in business_types):
        type_score = 0.10
    else:
        type_score = 0.0
    factors["business_type"] = type_score

    # BASE SCORE from distance + type
    base_score = distance_score + type_score

    # GROW LIGHT LIKELIHOOD SCORE (distance-weighted)
    # Use distance as weighting factor: closer = higher weight
    distance_weight = 1 - (distance_m / max_distance)  # 1.0 at center, 0.0 at boundary

    if perplexity.grow_light_likelihood is not None:
        # Direct likelihood score (0.0-1.0) weighted by distance
        # Maximum contribution is 0.5 when at pixel center
        likelihood_score = 0.5 * perplexity.grow_light_likelihood * distance_weight
        factors["grow_light_likelihood"] = round(perplexity.grow_light_likelihood, 3)
        factors["grow_light_score"] = round(likelihood_score, 3)
        logger.debug(
            "Grow light likelihood: %.3f x distance_weight: %.3f = score: %.3f",
            perplexity.grow_light_likelihood,
            distance_weight,
            likelihood_score,
        )
    else:
        # No lighting information - score remains 0
        likelihood_score = 0.0
        factors["no_lighting_info"] = 0.0
        logger.debug("No lighting information available → score: 0.000")

    # MULTIPLIERS (Zen's suggestion: signals amplify base score)
    multiplier = 1.0

    # Energy-intensive crops boost
    energy_crops = {"tomato", "pepper", "cucumber", "flower", "rose"}
    if any(crop.lower() in energy_crops for crop in perplexity.primary_crops):
        multiplier *= 1.15
        factors["energy_crops"] = 0.15

    # Large operation boost (>5 hectares)
    large_greenhouse_threshold_hectares = 5.0
    if (
        perplexity.size_hectares
        and perplexity.size_hectares > large_greenhouse_threshold_hectares
    ):
        multiplier *= 1.10
        factors["large_operation"] = 0.10

    # Calculate total score
    total_score = min((base_score + likelihood_score) * multiplier, 1.0)
    factors["total"] = round(total_score, 3)

    logger.info("Confidence calculation complete:")
    logger.info("  Distance score: %.3f", factors["distance"])
    logger.info("  Business type score: %.3f", factors["business_type"])
    if "grow_light_likelihood" in factors:
        logger.info(
            "  Grow light likelihood (raw): %.3f", factors["grow_light_likelihood"]
        )
        logger.info(
            "  Grow light score (weighted): %.3f", factors.get("grow_light_score", 0.0)
        )
    else:
        logger.info("  No lighting info: %.3f", factors.get("no_lighting_info", 0.0))
    logger.info("  Base score: %.3f", base_score)
    logger.info("  Multiplier: %.3f", multiplier)
    logger.info("  TOTAL CONFIDENCE: %.3f", total_score)

    return total_score, factors


async def scan_pixel_for_attribution(  # noqa: PLR0915
    pixel_center: Coordinates,
    pixel_radiance: float,
    location_context: str = "Netherlands",
    confidence_threshold: float = 0.85,
    max_businesses_to_analyze: int = 5,
) -> list[AttributionResult]:
    """
    Scan VIIRS pixel for greenhouse attribution using AI-powered analysis.

    This is the main orchestrator that coordinates:
    1. Single-radius business search (1km)
    2. Perplexity AI analysis for each business
    3. Confidence scoring and ranking
    4. Early stopping when high-confidence match found

    Args:
        pixel_center: VIIRS pixel center coordinates
        pixel_radiance: Radiance value in nW/cm²/sr
        location_context: Location context for Perplexity queries
        confidence_threshold: Stop when confidence >= this (default 0.85)
        max_businesses_to_analyze: Max businesses to query (API limit)

    Returns:
        List of AttributionResult sorted by confidence (highest first)

    """
    logger.info("=" * 70)
    logger.info("Starting greenhouse attribution scan")
    logger.info("Pixel coordinates: %.6f°N, %.6f°E", pixel_center.lat, pixel_center.lon)
    logger.info("Pixel radiance: %.2f nW/cm²/sr", pixel_radiance)
    logger.info("Location context: %s", location_context)
    logger.info("Confidence threshold: %.2f", confidence_threshold)
    logger.info("Max businesses to analyze: %d", max_businesses_to_analyze)
    logger.info("=" * 70)

    # Header
    console.print("\n" + "=" * 60)
    console.print(
        Panel.fit(
            "[bold cyan]Greenhouse Attribution Scanner[/bold cyan]\n"
            "AI-Powered Light Pollution Source Identification",
            border_style="cyan",
        )
    )

    console.print(
        f"[cyan]Pixel Center:[/cyan] {pixel_center.lat:.4f}°N, "
        f"{pixel_center.lon:.4f}°E"
    )
    console.print(f"[cyan]Radiance:[/cyan] {pixel_radiance:.2f} nW/cm²/sr")
    console.print("=" * 60 + "\n")

    # Phase 1: Search businesses (single API call optimization)
    logger.info("Phase 1: Business search starting...")
    console.print(
        "[yellow]Phase 1:[/yellow] Searching nearby businesses (1km radius)..."
    )

    # Use existing find_nearby_businesses (single call at 1000m)
    logger.debug("Calling find_nearby_businesses with 1000m radius...")
    businesses = find_nearby_businesses(pixel_center, radius=1000)
    logger.info("Business search returned %d results", len(businesses))

    if not businesses:
        logger.warning("No businesses found within 1km radius")
        console.print("[red]✗ No businesses found within 1km[/red]")
        return []

    # Calculate distances and sort
    logger.debug("Sorting businesses by distance from pixel center...")
    businesses_with_distance: list[tuple[Business, float]] = []
    for business in businesses:
        distance = business.distance_m if business.distance_m else 0.0
        businesses_with_distance.append((business, distance))
        logger.debug(
            "  %s: %.1fm (types: %s)",
            business.name,
            distance,
            ", ".join(business.types[:3]),
        )

    businesses_with_distance.sort(key=lambda x: x[1])
    logger.info(
        "Businesses sorted. Closest: %s (%.1fm)",
        businesses_with_distance[0][0].name,
        businesses_with_distance[0][1],
    )

    console.print(
        f"[green]✓ Found {len(businesses_with_distance)} businesses[/green]\n"
    )

    # Phase 2 & 3: Analyze with Perplexity + Score
    if not settings.perplexity_api_key:
        logger.error("PERPLEXITY_API_KEY not configured")
        console.print(
            "[yellow]Warning: PERPLEXITY_API_KEY not set. "
            "Attribution analysis cannot proceed.[/yellow]\n"
        )
        return []

    logger.info("Phase 2 & 3: AI Analysis and Confidence Scoring starting...")
    logger.debug("Initializing Perplexity analyzer...")
    analyzer = PerplexityGreenhouseAnalyzer(api_key=settings.perplexity_api_key)
    results: list[AttributionResult] = []

    for idx, (business, distance) in enumerate(businesses_with_distance, 1):
        if idx > max_businesses_to_analyze:
            logger.info(
                "Reached max businesses limit (%d), stopping analysis",
                max_businesses_to_analyze,
            )
            console.print(
                f"[yellow]Stopping: analyzed {idx - 1} businesses "
                f"(max limit)[/yellow]"
            )
            break

        logger.info(
            "Analyzing business %d/%d: %s (%.1fm from pixel)",
            idx,
            min(len(businesses_with_distance), max_businesses_to_analyze),
            business.name,
            distance,
        )
        console.print(f"\n[cyan]Analyzing:[/cyan] {business.name} ({distance:.0f}m)")

        # Query Perplexity
        logger.debug("Starting Perplexity API query...")
        analysis = await analyzer.analyze_business(business, location_context)
        logger.debug("Perplexity query complete")

        if analysis:
            logger.info("Analysis successful, calculating confidence...")
            # Calculate confidence
            confidence, factors = calculate_confidence(
                analysis, distance, business.types
            )

            result = AttributionResult(
                business=business,
                distance_from_pixel_m=distance,
                perplexity_analysis=analysis,
                confidence_score=confidence,
                confidence_factors=factors,
            )

            logger.info("Attribution result created with confidence: %.2f", confidence)
            logger.debug("Confidence factors: %s", factors)

            console.print(f"  [green]✓[/green] Analysis complete")
            console.print(f"  Confidence: {confidence:.2f}")

            if analysis.primary_crops:
                console.print(f"  Crops: {', '.join(analysis.primary_crops)}")

        else:
            logger.warning("Analysis failed for business: %s", business.name)
            result = AttributionResult(
                business=business,
                distance_from_pixel_m=distance,
                perplexity_analysis=None,
                confidence_score=0.0,
                confidence_factors={},
                api_error="Failed to get Perplexity analysis",
            )
            console.print("  [red]✗[/red] Analysis failed")

        results.append(result)
        logger.debug("Result added to results list (total: %d)", len(results))

        # Early stopping on high confidence
        if result.confidence_score >= confidence_threshold:
            logger.info(
                "HIGH CONFIDENCE MATCH (%.2f >= %.2f) - stopping early",
                result.confidence_score,
                confidence_threshold,
            )
            console.print(
                f"\n[green]✓ HIGH CONFIDENCE MATCH FOUND! " f"Stopping scan.[/green]"
            )
            break

    # Sort by confidence (highest first)
    logger.info("Sorting %d results by confidence score...", len(results))
    results.sort(key=lambda r: r.confidence_score, reverse=True)

    if results:
        logger.info(
            "Top result: %s with confidence %.2f",
            results[0].business.name,
            results[0].confidence_score,
        )
        logger.debug(
            "All confidence scores: %s", [f"{r.confidence_score:.2f}" for r in results]
        )

    # Display summary
    console.print("\n" + "=" * 60)
    _display_attribution_summary(results)
    console.print("=" * 60 + "\n")

    logger.info("=" * 70)
    logger.info("Attribution scan complete - returning %d results", len(results))
    logger.info("=" * 70)

    return results


def _display_attribution_summary(results: list[AttributionResult]) -> None:  # noqa: C901
    """Display attribution results in a formatted table."""
    if not results:
        console.print("[yellow]No attribution results to display[/yellow]")
        return

    console.print("[bold]ATTRIBUTION RESULTS[/bold]\n")

    # Create table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Rank", style="yellow")
    table.add_column("Business", style="white")
    table.add_column("Distance", style="cyan")
    table.add_column("Confidence", style="green")
    table.add_column("Status", style="white")

    for rank, result in enumerate(results[:5], 1):  # Top 5
        distance_str = f"{result.distance_from_pixel_m:.0f}m"
        confidence_str = f"{result.confidence_score:.2f}"

        if result.api_error:
            status = "✗ Failed"
        elif (
            result.perplexity_analysis
            and result.perplexity_analysis.grow_light_likelihood
            and result.perplexity_analysis.grow_light_likelihood
            >= HIGH_LIKELIHOOD_THRESHOLD
        ):
            status = "✓ High likelihood"
        else:
            status = "? Unknown"

        table.add_row(
            str(rank), result.business.name, distance_str, confidence_str, status
        )

    console.print(table)

    # Show top result details
    if results[0].confidence_score > 0:
        top = results[0]
        console.print(f"\n[bold]Most Likely Culprit:[/bold] {top.business.name}")
        console.print(f"  Confidence: {top.confidence_score:.0%}")
        console.print(f"  Distance: {top.distance_from_pixel_m:.0f}m from pixel center")

        if top.perplexity_analysis:
            if top.perplexity_analysis.primary_crops:
                console.print(
                    f"  Crops: {', '.join(top.perplexity_analysis.primary_crops)}"
                )
            if top.perplexity_analysis.instagram_handle:
                console.print(
                    f"  Instagram: @{top.perplexity_analysis.instagram_handle}"
                )
            if top.perplexity_analysis.sources:
                console.print("  Sources:")
                for source in top.perplexity_analysis.sources[:3]:
                    console.print(f"    - {source}")
