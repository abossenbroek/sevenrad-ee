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
from .models import Coordinates, TopEmitter


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
    Query VIIRS DNB for top N brightest emitters in a region.

    Uses caching to avoid repeated expensive EE queries.

    Args:
        region_path: Path to GeoJSON file defining the region
        start_date: Start date for temporal filter
        end_date: End date for temporal filter
        n: Number of top emitters to return (default: 20)

    Returns:
        List of TopEmitter models sorted by avg_radiance (descending)

    Raises:
        ValueError: If region GeoJSON is invalid or dates are invalid

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

    # Take top N and convert to TopEmitter models
    top_features = sorted_features[:n]
    emitters = []

    for rank, feature in enumerate(top_features, 1):
        coords_list = feature["geometry"]["coordinates"]
        radiance = feature["properties"]["avg_rad"]

        emitter = TopEmitter(
            rank=rank,
            coordinates=Coordinates(lat=coords_list[1], lon=coords_list[0]),
            avg_radiance=radiance,
        )
        emitters.append(emitter)

    # Cache the results
    cache.set(cache_key, emitters)

    return emitters
