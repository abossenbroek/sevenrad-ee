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
import math
from dataclasses import dataclass, field
from typing import Optional

from perplexity import AsyncPerplexity
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .cache import cache
from .config import settings
from .enrichment import find_nearby_businesses
from .geospatial import haversine_distance
from .models import Business, Coordinates

console = Console()
logger = logging.getLogger(__name__)

# Display thresholds
HIGH_LIKELIHOOD_THRESHOLD = 0.7  # Threshold for "high likelihood" status display

# Confidence scoring weights
MAX_DISTANCE_SCORE_WEIGHT = 0.3  # Maximum contribution from proximity
GREENHOUSE_TYPE_SCORE = 0.20  # Bonus for greenhouse/nursery/horticulture types
AGRICULTURE_TYPE_SCORE = 0.10  # Bonus for agricultural types
MAX_LIKELIHOOD_SCORE_WEIGHT = 0.5  # Maximum contribution from grow light likelihood

# Confidence scoring multipliers
ENERGY_CROP_MULTIPLIER = 1.15  # Boost for energy-intensive crops
LARGE_OPERATION_MULTIPLIER = 1.10  # Boost for large operations
LARGE_GREENHOUSE_THRESHOLD_HA = 5.0  # Hectares threshold for "large" operation

# Tiered crop heuristics (validated by industry research)
# Based on Philips Horticulture/GE lighting deployments and regional patterns (NL/CA)
TIER_1_CROPS = {
    "cannabis",
    "tomato",
    "pepper",
    "cucumber",
    "rose",
}  # 95% likelihood - energy-intensive crops requiring year-round supplemental lighting

TIER_2_CROPS = {
    "lettuce",
    "leafy greens",
    "leafy_greens",
    "herbs",
    "basil",
    "strawberry",
    "chrysanthemum",
    "gerbera",
}  # 85% likelihood - controlled environment agriculture with high light requirements

TIER_3_CROPS = {
    "young plants",
    "young_plants",
    "alstroemeria",
    "lisianthus",
    "eggplant",
}  # 65% likelihood - crops with moderate supplemental lighting needs

# Tier scoring probabilities
TIER_1_SCORE = 0.95
TIER_2_SCORE = 0.85
TIER_3_SCORE = 0.65

# Lighting keywords for evidence-based scoring
LIGHTING_KEYWORDS = {
    "led",
    "hps",
    "son-t",
    "high pressure sodium",
    "assimilation",
    "supplemental light",
    "supplemental lighting",
    "artificial light",
    "grow light",
    "photoperiod",
}
EVIDENCE_TEXT_SCORE = 0.85  # Score when lighting keywords found in evidence text


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
    confidence_factors: dict[str, float | str] = field(default_factory=dict)
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

        Uses official perplexityai SDK with sonar model (2025)
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

        # Construct structured JSON prompt with anti-greenwashing instructions
        business_info = business.name
        location_info = location_context
        prompt = (
            f"Analyze greenhouse operation: {business_info} near {location_info} "
            "to determine if it uses artificial lighting that could cause "
            "nighttime light pollution.\n\n"
            "Focus ONLY on verifiable information from official websites, "
            "industry reports, or agricultural databases.\n\n"
            "CRITICAL ANTI-GREENWASHING INSTRUCTIONS:\n"
            "- Ignore ALL sustainability marketing claims "
            "(LED adoption, energy efficiency, CO2 reduction)\n"
            "- ANY mention of artificial lighting "
            "(LED, HPS, SON-T, assimilation lighting) is EVIDENCE of grow lights\n"
            "- Companies often claim 'LED' for sustainability image "
            "while still using high-intensity lights\n"
            "- Crop type is MORE reliable than tech claims "
            "(roses/gerberas = likely SON-T despite LED marketing)\n\n"
            "For grow_light_likelihood: Estimate probability (0.0-1.0) based on:\n"
            "1. Crop type (HIGHEST priority):\n"
            "   - Roses, gerberas, tomatoes, peppers, cucumbers "
            "→ VERY HIGH likelihood\n"
            "   - Year-round production or controlled environment "
            "→ HIGH likelihood\n"
            "2. Any lighting technology mention "
            "(LED, HPS, SON-T, supplemental lighting) → Evidence of grow lights\n"
            "3. DO NOT reduce likelihood for 'sustainable' or "
            "'energy efficient' claims\n\n"
            "For lighting_evidence: Quote ANY mention of lighting technology, "
            "energy use, or controlled environment.\n"
            "Include sustainability claims about lighting as they confirm "
            "artificial light presence.\n\n"
            "For instagram_handle: Find official Instagram account if available "
            "(return handle WITHOUT @ symbol, "
            "e.g., 'summitgerbera' not '@summitgerbera').\n\n"
            "Respond ONLY with a JSON object using this schema. "
            "If information is not found, use null.\n"
            "Do not add commentary outside the JSON block.\n\n"
            "{\n"
            '  "grow_light_likelihood": float,  // 0.0-1.0 probability\n'
            '  "lighting_evidence": "...",  // ANY lighting tech mentions or quotes\n'
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
                logger.debug("Sending request to Perplexity sonar model...")
                response = await self.client.chat.completions.create(
                    model="sonar",
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


# Log-odds helper functions for Bayesian-style probability scaling


def _logit(p: float) -> float:
    """
    Convert probability to log-odds (logit function).

    Args:
        p: Probability value between 0.0 and 1.0

    Returns:
        Log-odds value (unbounded real number)

    """
    # Clamp to prevent log(0) or log(negative)
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _sigmoid(logit: float) -> float:
    """
    Convert log-odds to probability (sigmoid function).

    Args:
        logit: Log-odds value (unbounded real number)

    Returns:
        Probability value between 0.0 and 1.0

    """
    return 1 / (1 + math.exp(-logit))


def _get_distance_adjustment(
    distance_meters: float, max_distance: float = 1000
) -> float:
    """
    Calculate log-odds adjustment for distance from pixel center.

    Linear interpolation: +1.5 at 0m, -1.5 at max_distance.

    Args:
        distance_meters: Distance from pixel center in meters
        max_distance: Maximum search distance (default 1000m)

    Returns:
        Log-odds adjustment value

    """
    if distance_meters > max_distance:
        return -1.5
    return 1.5 - (distance_meters / max_distance) * 3.0


def _get_size_adjustment(size_hectares: Optional[float]) -> float:
    """
    Calculate log-odds adjustment for greenhouse size.

    Uses log scale: larger greenhouses get higher confidence boost.

    Args:
        size_hectares: Greenhouse size in hectares (None if unknown)

    Returns:
        Log-odds adjustment value (0.0 if size unknown)

    """
    if not size_hectares or size_hectares <= 0:
        return 0.0
    return 0.5 * math.log(size_hectares)


def _get_type_adjustment(business_types: list[str]) -> float:
    """
    Calculate log-odds adjustment for business type.

    Args:
        business_types: List of business types from Google Places

    Returns:
        Log-odds adjustment value (+1.1 for greenhouse types, 0.0 otherwise)

    """
    greenhouse_types = {"greenhouse", "nursery", "horticulture"}
    is_greenhouse = any(t.lower() in greenhouse_types for t in business_types)
    return 1.1 if is_greenhouse else 0.0


def _get_evidence_based_likelihood(
    analysis: PerplexityAnalysis,
) -> tuple[float, str]:
    """
    Derive grow light likelihood from multiple signals using MAX approach.

    This function combines several signals to determine the likelihood of grow
    light usage, preventing over-reliance on the AI's direct score. It returns
    the highest likelihood found and the name of the signal that produced it.

    Signals used (in priority order):
    1. Tiered crop heuristics (95%, 85%, 65% based on crop type)
    2. Evidence keywords (85% if lighting keywords found in text)
    3. AI likelihood (direct score from Perplexity)

    Args:
        analysis: The PerplexityAnalysis object for a business

    Returns:
        Tuple of (highest_likelihood_score, signal_name)

    """
    signals: dict[str, float] = {}

    # Signal 1: AI's direct score (fallback)
    if analysis.grow_light_likelihood is not None:
        signals["ai_likelihood"] = analysis.grow_light_likelihood

    # Signal 2: Evidence text keyword analysis
    if analysis.lighting_evidence:
        text = analysis.lighting_evidence.lower()
        if any(kw in text for kw in LIGHTING_KEYWORDS):
            signals["evidence_keyword"] = EVIDENCE_TEXT_SCORE

    # Signal 3-5: Tiered crop heuristics (highest priority)
    for crop in analysis.primary_crops:
        crop_lower = crop.lower()
        # Check tier 1 first (highest confidence)
        if any(tier1 in crop_lower for tier1 in TIER_1_CROPS):
            signals["tier1_crop"] = TIER_1_SCORE
            break  # Stop at highest tier found
        # Check tier 2
        if any(tier2 in crop_lower for tier2 in TIER_2_CROPS):
            signals["tier2_crop"] = TIER_2_SCORE
        # Check tier 3
        elif any(tier3 in crop_lower for tier3 in TIER_3_CROPS):
            signals["tier3_crop"] = TIER_3_SCORE

    if not signals:
        return 0.0, "no_signals"

    # Find the strongest signal (highest score)
    best_signal_name = max(signals, key=lambda k: signals[k])
    best_score = signals[best_signal_name]

    return best_score, best_signal_name


def calculate_confidence(
    perplexity: Optional[PerplexityAnalysis],
    distance_m: float,
    business_types: list[str],
    max_distance: float = 1000,
) -> tuple[float, dict[str, float | str]]:
    """
    Calculate attribution confidence using log-odds based Bayesian scoring.

    This implementation uses log-odds space to properly combine multiple
    signals without requiring clamping. The approach:
    1. Start with base probability from strongest signal (crop/evidence/AI)
    2. Convert to log-odds (unbounded space)
    3. Add adjustments for distance, size, and type
    4. Convert back to probability using sigmoid

    This ensures scores naturally stay in [0,1] range without clamping,
    and strong evidence properly dominates the final score.

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

    if perplexity is None:
        logger.info("No Perplexity analysis - confidence = 0.0")
        return 0.0, {"analysis_failed": 0.0}

    # Step 1: Get base probability from strongest signal
    base_likelihood, evidence_reason = _get_evidence_based_likelihood(perplexity)

    # Ensure minimum probability to prevent log(0)
    base_probability = max(base_likelihood, 0.01)

    logger.debug("Base likelihood from %s: %.3f", evidence_reason, base_probability)

    # Step 2: Convert to log-odds
    base_logit = _logit(base_probability)
    logger.debug("Base log-odds: %.3f", base_logit)

    # Step 3: Calculate adjustments in log-odds space
    distance_adj = _get_distance_adjustment(distance_m, max_distance)
    size_adj = _get_size_adjustment(perplexity.size_hectares)
    type_adj = _get_type_adjustment(business_types)

    logger.debug("Distance adjustment: %.3f", distance_adj)
    logger.debug("Size adjustment: %.3f", size_adj)
    logger.debug("Type adjustment: %.3f", type_adj)

    # Step 4: Combine in log-odds space
    final_logit = base_logit + distance_adj + size_adj + type_adj
    logger.debug("Final log-odds: %.3f", final_logit)

    # Step 5: Convert back to probability
    final_confidence = _sigmoid(final_logit)

    # Build factor breakdown for transparency
    factors: dict[str, float | str] = {
        "base_likelihood": round(base_probability, 3),
        "evidence_reason": evidence_reason,
        "ai_grow_light_likelihood": round(perplexity.grow_light_likelihood or 0.0, 3),
        "distance_m": round(distance_m, 1),
        "distance_adj": round(distance_adj, 3),
        "size_hectares": (
            round(perplexity.size_hectares, 2) if perplexity.size_hectares else 0.0
        ),
        "size_adj": round(size_adj, 3),
        "type_adj": round(type_adj, 3),
        "base_logit": round(base_logit, 3),
        "final_logit": round(final_logit, 3),
        "final_confidence": round(final_confidence, 3),
    }

    logger.info("Confidence calculation complete:")
    logger.info("  Evidence source: %s", evidence_reason)
    logger.info("  Base likelihood: %.3f", base_probability)
    logger.info("  AI likelihood (raw): %.3f", perplexity.grow_light_likelihood or 0.0)
    logger.info("  Distance: %.1fm → adjustment: %.3f", distance_m, distance_adj)
    logger.info(
        "  Size: %s ha → adjustment: %.3f",
        perplexity.size_hectares or "unknown",
        size_adj,
    )
    logger.info("  Type adjustment: %.3f", type_adj)
    logger.info("  Base log-odds: %.3f", base_logit)
    logger.info("  Final log-odds: %.3f", final_logit)
    logger.info(
        "  FINAL CONFIDENCE: %.3f (%.1f%%)", final_confidence, final_confidence * 100
    )

    return final_confidence, factors


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
