"""
Data enrichment services using Google Cloud APIs.

This module provides functions to:
- Geocode coordinates to addresses with quality scoring
- Find nearby businesses with relevance scoring
- Fetch Street View imagery from four cardinal directions

Includes rate limiting to respect API usage quotas.
"""

import functools
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import googlemaps
from rich.console import Console

from .cache import cache
from .config import settings
from .geospatial import haversine_distance
from .models import Address, Business, Coordinates, StreetViewImages

console = Console()

F = TypeVar("F", bound=Callable[..., Any])


def rate_limited(delay_ms: int) -> Callable[[F], F]:
    """
    Limit the call rate of a function.

    Args:
        delay_ms: Minimum delay between calls in milliseconds

    Returns:
        Decorated function with rate limiting

    """

    def decorator(func: F) -> F:
        last_called = [0.0]  # Use list to allow modification in nested scope

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            elapsed = time.monotonic() - last_called[0]
            wait_time = (delay_ms / 1000.0) - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            result = func(*args, **kwargs)
            last_called[0] = time.monotonic()
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def _calculate_address_quality(result: dict[str, Any]) -> float:
    """
    Score geocoding result quality based on Google's location_type.

    Quality scoring factors:
    - Type precision (street_address=1.0, premise=0.95, locality=0.5, etc.)
    - Partial match penalty (0.8x multiplier)
    - Component completeness bonus

    Args:
        result: Geocoding API result dictionary

    Returns:
        Quality score between 0.0 and 1.0

    """
    # Base quality score by location type
    quality_map = {
        "ROOFTOP": 1.0,
        "RANGE_INTERPOLATED": 0.9,
        "GEOMETRIC_CENTER": 0.75,
        "APPROXIMATE": 0.5,
    }
    location_type = result.get("geometry", {}).get("location_type", "APPROXIMATE")
    base_score = quality_map.get(location_type, 0.3)

    # Penalty for partial matches
    if result.get("partial_match", False):
        base_score *= 0.8

    # Bonus for completeness (has street number, route, and postal code)
    address_components = result.get("address_components", [])
    component_types = {c for comp in address_components for c in comp.get("types", [])}

    has_street_number = "street_number" in component_types
    has_route = "route" in component_types
    has_postal = "postal_code" in component_types

    completeness = sum([has_street_number, has_route, has_postal]) / 3.0
    return base_score * (0.8 + 0.2 * completeness)


def _calculate_business_relevance(
    business_data: dict[str, Any], source_coords: Coordinates
) -> tuple[float, float]:
    """
    Score business relevance based on proximity, type, and rating.

    Formula: proximity_score * type_score * rating_score

    Args:
        business_data: Google Places API result
        source_coords: Source coordinates to measure distance from

    Returns:
        Tuple of (relevance_score, distance_m)

    """
    # Calculate distance
    business_coords = Coordinates(
        lat=business_data["geometry"]["location"]["lat"],
        lon=business_data["geometry"]["location"]["lng"],
    )
    distance_m = haversine_distance(source_coords, business_coords)

    # Proximity score: exponential decay (0m=1.0, 50m≈0.37, 100m≈0.14)
    proximity_score = math.exp(-distance_m / 50.0)

    # Type weighting: high-emission types get bonus
    high_emission_types = {
        "stadium",
        "shopping_mall",
        "gas_station",
        "parking",
        "night_club",
        "casino",
        "airport",
        "industrial_facility",
        "power_plant",
        "factory",
    }
    business_types = set(business_data.get("types", []))
    type_score = 1.5 if business_types & high_emission_types else 1.0

    # Rating score: 5-star is 1.2x, 3.5-star is 1.0x, 1-star is 0.6x
    rating = business_data.get("rating")
    rating_score = 0.8 + (rating / 5.0) * 0.4 if rating is not None else 0.9

    relevance_score = proximity_score * type_score * rating_score
    return relevance_score, distance_m


@rate_limited(50)
def _geocode_coordinates_api(coords: Coordinates) -> dict[str, Any] | None:
    """
    Call Google Geocoding API for coordinates.

    Args:
        coords: Coordinates to geocode

    Returns:
        API response dict or None if error

    """
    if not settings.google_maps_api_key:
        return None

    gmaps = googlemaps.Client(key=settings.google_maps_api_key)
    try:
        results = gmaps.reverse_geocode((coords.lat, coords.lon))
        return results[0] if results else None
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Geocoding failed: {e}")
        return None


def geocode_coordinates(coords: Coordinates) -> Address | None:
    """
    Reverse geocode coordinates to an address with quality scoring.

    Uses caching to prevent redundant API calls.

    Args:
        coords: Coordinates to geocode

    Returns:
        Address model with quality score, or None if geocoding fails

    """
    cache_key = f"geocode:{coords.lat},{coords.lon}"
    cached = cache.get(cache_key)
    if cached:
        return cached  # type: ignore[no-any-return]

    result = _geocode_coordinates_api(coords)
    if not result:
        return None

    # Parse address components
    address_components = {
        comp_type: comp["long_name"]
        for comp in result.get("address_components", [])
        for comp_type in comp.get("types", [])
    }

    address = Address(
        formatted=result["formatted_address"],
        street=address_components.get("route"),
        city=address_components.get("locality"),
        state=address_components.get("administrative_area_level_1"),
        country=address_components.get("country"),
        postal_code=address_components.get("postal_code"),
        place_id=result["place_id"],
        quality_score=_calculate_address_quality(result),
        match_type=result["geometry"]["location_type"],
    )

    cache.set(cache_key, address)
    return address


@rate_limited(50)
def _find_nearby_businesses_api(
    coords: Coordinates, radius: int
) -> list[dict[str, Any]]:
    """
    Call Google Places API for nearby businesses.

    Args:
        coords: Center coordinates
        radius: Search radius in meters

    Returns:
        List of business result dicts

    """
    if not settings.google_maps_api_key:
        return []

    gmaps = googlemaps.Client(key=settings.google_maps_api_key)
    try:
        results = gmaps.places_nearby(
            location=(coords.lat, coords.lon),
            radius=radius,
        )
        return results.get("results", [])  # type: ignore[no-any-return]
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Business search failed: {e}")
        return []


def find_nearby_businesses(
    coords: Coordinates, radius: int | None = None
) -> list[Business]:
    """
    Find nearby businesses with relevance scoring.

    Uses caching to prevent redundant API calls.

    Args:
        coords: Center coordinates
        radius: Search radius in meters (uses config default if None)

    Returns:
        List of Business models sorted by relevance score (descending)

    """
    if radius is None:
        radius = settings.business_search_radius_m

    cache_key = f"businesses:{coords.lat},{coords.lon}:{radius}"
    cached = cache.get(cache_key)
    if cached:
        return cached  # type: ignore[no-any-return]

    results = _find_nearby_businesses_api(coords, radius)

    businesses = []
    for result in results:
        relevance_score, distance_m = _calculate_business_relevance(result, coords)

        # Extract business coordinates
        business_coords = Coordinates(
            lat=result["geometry"]["location"]["lat"],
            lon=result["geometry"]["location"]["lng"],
        )

        # Generate Street View link
        streetview_link = (
            f"https://www.google.com/maps/@?api=1&map_action=pano&"
            f"viewpoint={business_coords.lat},{business_coords.lon}"
        )

        businesses.append(
            Business(
                name=result["name"],
                place_id=result["place_id"],
                types=result.get("types", []),
                vicinity=result.get("vicinity"),
                distance_m=distance_m,
                rating=result.get("rating"),
                relevance_score=relevance_score,
                coordinates=business_coords,
                streetview_link=streetview_link,
            )
        )

    # Sort by relevance score (descending)
    businesses.sort(key=lambda b: b.relevance_score, reverse=True)

    cache.set(cache_key, businesses)
    return businesses


@rate_limited(100)
def _check_streetview_availability(coords: Coordinates) -> bool:
    """
    Check if Street View imagery is available at coordinates.

    Args:
        coords: Coordinates to check

    Returns:
        True if imagery available

    """
    if not settings.google_maps_api_key:
        return False

    gmaps = googlemaps.Client(key=settings.google_maps_api_key)
    try:
        metadata = gmaps.streetview_metadata((coords.lat, coords.lon))
        return metadata.get("status") == "OK"  # type: ignore[no-any-return]
    except Exception:
        return False


def get_street_view_images(
    coords: Coordinates,
    sub_dir_name: str,
    images_dir: Path | None = None,
) -> StreetViewImages:
    """
    Get Street View images from four cardinal directions.

    Saves images to a specified subdirectory.

    Args:
        coords: Center coordinates
        sub_dir_name: Name for the subdirectory to store images (e.g., business name)
        images_dir: Base directory for images (defaults to settings.images_dir)

    Returns:
        StreetViewImages model with paths to saved images

    """
    cache_key = f"streetview:{coords.lat},{coords.lon}"
    cached = cache.get(cache_key)
    if cached:
        return cached  # type: ignore[no-any-return]

    # Check availability first
    if not _check_streetview_availability(coords):
        result = StreetViewImages(available=False)
        cache.set(cache_key, result)
        return result

    # Create directory for images
    base_dir = images_dir if images_dir is not None else settings.images_dir
    image_dir = base_dir / "streetview" / sub_dir_name
    image_dir.mkdir(parents=True, exist_ok=True)

    # Download images for four cardinal directions
    if not settings.google_maps_api_key:
        return StreetViewImages(available=False)

    gmaps = googlemaps.Client(key=settings.google_maps_api_key)
    image_paths: dict[str, Path | None] = {}

    for direction, heading in [
        ("north", 0),
        ("east", 90),
        ("south", 180),
        ("west", 270),
    ]:
        try:
            image_data = gmaps.streetview(
                size=(640, 640),
                location=(coords.lat, coords.lon),
                heading=heading,
                pitch=0,
            )

            if image_data:
                image_path = image_dir / f"{direction}.jpg"
                image_path.write_bytes(image_data)
                image_paths[direction] = image_path
                time.sleep(0.1)  # Rate limiting
        except Exception:
            image_paths[direction] = None

    result = StreetViewImages(
        available=True,
        north=image_paths.get("north"),
        south=image_paths.get("south"),
        east=image_paths.get("east"),
        west=image_paths.get("west"),
    )

    cache.set(cache_key, result)
    return result
