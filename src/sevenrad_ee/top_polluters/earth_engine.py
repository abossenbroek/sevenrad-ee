"""
Earth Engine integration for VIIRS DNB top emitters query.

This module provides functions to query NOAA VIIRS DNB monthly data
and identify the top N brightest emitters in a geographic region.
"""

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ee

from .cache import cache
from .config import settings
from .geospatial import haversine_distance
from .models import Coordinates, TopEmitter


class NotEnoughEmittersError(ValueError):
    """
    Raised when insufficient spatially distinct emitters can be found.

    This exception is raised when the requested number of emitters cannot
    be found with the required minimum separation distance (VIIRS resolution).
    """

    pass


def _parse_geojson(geojson_path: Path) -> Any:  # noqa: ANN401
    """
    Parse GeoJSON file to Earth Engine Geometry.

    Handles three GeoJSON types:
    - FeatureCollection: uses first feature's geometry
    - Feature: uses the feature's geometry
    - Geometry: uses directly

    Args:
        geojson_path: Path to GeoJSON file

    Returns:
        Earth Engine Geometry object

    Raises:
        ValueError: If GeoJSON format is invalid

    """
    with geojson_path.open() as f:
        geojson = json.load(f)

    if geojson["type"] == "FeatureCollection":
        if not geojson.get("features"):
            msg = "FeatureCollection has no features"
            raise ValueError(msg)
        return ee.Geometry(geojson["features"][0]["geometry"])
    if geojson["type"] == "Feature":
        return ee.Geometry(geojson["geometry"])
    # Assume it's a raw Geometry
    return ee.Geometry(geojson)


def _generate_cache_key(
    geojson_path: Path, start_date: date, end_date: date, n: int
) -> str:
    """
    Generate deterministic cache key for EE query.

    Args:
        geojson_path: Path to GeoJSON file
        start_date: Query start date
        end_date: Query end date
        n: Number of top emitters to return

    Returns:
        SHA256 hash (truncated to 16 chars)

    """
    # Read GeoJSON content for cache key
    geojson_content = geojson_path.read_text()

    # Create key from all parameters
    key_str = f"{geojson_content}:{start_date.isoformat()}:{end_date.isoformat()}:{n}"
    hash_digest = hashlib.sha256(key_str.encode()).hexdigest()
    return f"ee_emitters:{hash_digest[:16]}"


def get_top_emitters(
    region_path: Path,
    start_date: date,
    end_date: date,
    n: int = 20,
) -> list[TopEmitter]:
    """
    Query VIIRS DNB for top N spatially distinct brightest emitters in a region.

    Uses caching to avoid repeated expensive EE queries. Applies spatial filtering
    to ensure all returned emitters are at least VIIRS_SCALE_M (750m) apart,
    reflecting the physical resolution of the VIIRS DNB sensor.

    Args:
        region_path: Path to GeoJSON file defining the region
        start_date: Start date for temporal filter
        end_date: End date for temporal filter
        n: Number of top emitters to return (default: 20)

    Returns:
        List of TopEmitter models sorted by avg_radiance (descending),
        with minimum 750m separation between all emitters

    Raises:
        ValueError: If region GeoJSON is invalid or dates are invalid
        NotEnoughEmittersError: If fewer than N spatially distinct emitters
            can be found with required minimum separation

    """
    # Validate dates
    if end_date < start_date:
        msg = f"End date {end_date} must be >= start date {start_date}"
        raise ValueError(msg)

    # Check cache first
    cache_key = _generate_cache_key(region_path, start_date, end_date, n)
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return cached_result  # type: ignore[no-any-return]

    # Parse GeoJSON
    geometry = _parse_geojson(region_path)

    # Query VIIRS DNB collection
    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
    filtered = viirs.filterDate(
        start_date.isoformat(), end_date.isoformat()
    ).filterBounds(geometry)

    # Temporal aggregation: mean radiance over date range
    mean_radiance = filtered.select("avg_rad").mean()

    # Sample at configured scale (default 500m)
    samples = mean_radiance.sample(
        region=geometry,
        scale=settings.viirs_scale_m,
        numPixels=10000,  # Max samples to collect
        geometries=True,
    )

    # Get features and sort by radiance
    features_info = samples.getInfo()
    if not features_info or "features" not in features_info:
        # No emitters found
        return []

    features = features_info["features"]
    sorted_features = sorted(
        features,
        key=lambda f: f["properties"].get("avg_rad", 0),
        reverse=True,
    )

    # Spatial filtering: Ensure minimum separation of VIIRS_SCALE_M meters
    # This reflects the physical resolution constraint of the VIIRS DNB sensor
    min_distance_m = settings.viirs_scale_m
    filtered_emitters: list[TopEmitter] = []

    for feature in sorted_features:
        # Stop if we have enough emitters
        if len(filtered_emitters) >= n:
            break

        coords_list = feature["geometry"]["coordinates"]
        lon, lat = coords_list[0], coords_list[1]
        radiance = feature["properties"]["avg_rad"]

        # Check distance to all previously selected emitters
        is_far_enough = True
        current_coords = Coordinates(lat=lat, lon=lon)
        for existing in filtered_emitters:
            distance = haversine_distance(current_coords, existing.coordinates)
            if distance < min_distance_m:
                is_far_enough = False
                break

        # Only add if sufficiently far from all existing emitters
        if is_far_enough:
            # Rank is based on position in filtered list (brightest first)
            rank = len(filtered_emitters) + 1
            emitter = TopEmitter(
                rank=rank,
                coordinates=Coordinates(lat=lat, lon=lon),
                avg_radiance=radiance,
            )
            filtered_emitters.append(emitter)

    # Check if we found enough spatially distinct emitters
    if len(filtered_emitters) < n:
        msg = (
            f"Could not find {n} spatially distinct emitters with minimum "
            f"separation of {min_distance_m}m (VIIRS sensor resolution). "
            f"Found only {len(filtered_emitters)} emitter(s). "
            f"Consider expanding the search region or reducing the number "
            f"of requested emitters (use -n {len(filtered_emitters)} or less)."
        )
        raise NotEnoughEmittersError(msg)

    # Cache the results
    cache.set(cache_key, filtered_emitters)

    return filtered_emitters
