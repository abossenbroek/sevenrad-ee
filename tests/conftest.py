"""
Pytest configuration and shared fixtures for VIIRS Top Polluters tests.

Provides:
- VCR configuration for recording/replaying Google API calls
- Mock fixtures for Earth Engine objects
- Sample GeoJSON fixtures
- Cache directory fixtures with automatic cleanup
"""

import json
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from sevenrad_ee.top_polluters.models import Coordinates, TopEmitter


@pytest.fixture
def vcr_config() -> dict[str, Any]:
    """
    Configure VCR.py for recording/replaying HTTP interactions.

    Returns:
        VCR configuration dictionary

    """
    return {
        "cassette_library_dir": "tests/cassettes",
        "record_mode": "once",  # Record once, then replay
        "match_on": ["uri", "method"],
        "filter_headers": ["authorization", "x-goog-api-key"],
    }


@pytest.fixture
def sample_region_geojson(tmp_path: Path) -> Path:
    """
    Create a sample GeoJSON FeatureCollection file.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to GeoJSON file

    """
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Test Region"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.5, 37.7],
                            [-122.5, 37.9],
                            [-122.3, 37.9],
                            [-122.3, 37.7],
                            [-122.5, 37.7],
                        ]
                    ],
                },
            }
        ],
    }

    geojson_path = tmp_path / "test_region.geojson"
    geojson_path.write_text(json.dumps(geojson_data))
    return geojson_path


@pytest.fixture
def sample_feature_geojson(tmp_path: Path) -> Path:
    """
    Create a sample GeoJSON Feature file (not FeatureCollection).

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to GeoJSON file

    """
    geojson_data = {
        "type": "Feature",
        "properties": {"name": "Test Feature"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-122.5, 37.7],
                    [-122.5, 37.9],
                    [-122.3, 37.9],
                    [-122.3, 37.7],
                    [-122.5, 37.7],
                ]
            ],
        },
    }

    geojson_path = tmp_path / "test_feature.geojson"
    geojson_path.write_text(json.dumps(geojson_data))
    return geojson_path


@pytest.fixture
def sample_geometry_geojson(tmp_path: Path) -> Path:
    """
    Create a sample GeoJSON Geometry file (raw geometry).

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to GeoJSON file

    """
    geojson_data = {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.5, 37.7],
                [-122.5, 37.9],
                [-122.3, 37.9],
                [-122.3, 37.7],
                [-122.5, 37.7],
            ]
        ],
    }

    geojson_path = tmp_path / "test_geometry.geojson"
    geojson_path.write_text(json.dumps(geojson_data))
    return geojson_path


@pytest.fixture
def mock_ee_geometry() -> Mock:
    """
    Create a mock Earth Engine Geometry object.

    Returns:
        Mock ee.Geometry

    """
    mock_geom = Mock()
    mock_geom.bounds = Mock(return_value=mock_geom)
    return mock_geom


@pytest.fixture
def mock_ee_image() -> Mock:
    """
    Create a mock Earth Engine Image object.

    Returns:
        Mock ee.Image

    """
    mock_image = Mock()
    mock_image.select = Mock(return_value=mock_image)
    mock_image.mean = Mock(return_value=mock_image)
    mock_image.sample = Mock(return_value=Mock())
    return mock_image


@pytest.fixture
def mock_ee_image_collection() -> Mock:
    """
    Create a mock Earth Engine ImageCollection object.

    Returns:
        Mock ee.ImageCollection

    """
    mock_collection = Mock()
    mock_collection.filterDate = Mock(return_value=mock_collection)
    mock_collection.filterBounds = Mock(return_value=mock_collection)
    mock_collection.select = Mock(return_value=mock_collection)
    mock_collection.mean = Mock(return_value=Mock())
    return mock_collection


@pytest.fixture
def mock_ee_feature_collection_data() -> dict[str, Any]:
    """
    Create mock data returned by ee.FeatureCollection.getInfo().

    Returns:
        Dictionary mimicking EE FeatureCollection structure

    """
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-122.4, 37.8]},
                "properties": {"avg_rad": 45.5},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-122.35, 37.75]},
                "properties": {"avg_rad": 38.2},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-122.45, 37.85]},
                "properties": {"avg_rad": 52.1},
            },
        ],
    }


@pytest.fixture
def sample_coordinates() -> Coordinates:
    """
    Create sample Coordinates model.

    Returns:
        Coordinates instance

    """
    return Coordinates(lat=37.8, lon=-122.4)


@pytest.fixture
def sample_top_emitters() -> list[TopEmitter]:
    """
    Create sample list of TopEmitter models.

    Returns:
        List of TopEmitter instances

    """
    return [
        TopEmitter(
            rank=1,
            coordinates=Coordinates(lat=37.8, lon=-122.4),
            avg_radiance=52.1,
        ),
        TopEmitter(
            rank=2,
            coordinates=Coordinates(lat=37.75, lon=-122.35),
            avg_radiance=45.5,
        ),
        TopEmitter(
            rank=3,
            coordinates=Coordinates(lat=37.85, lon=-122.45),
            avg_radiance=38.2,
        ),
    ]


@pytest.fixture
def temp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Create temporary cache directory and patch settings.

    Args:
        tmp_path: Pytest temporary directory fixture
        monkeypatch: Pytest monkeypatch fixture

    Returns:
        Path to temporary cache directory

    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Patch settings to use temp cache directory
    from sevenrad_ee.top_polluters import config
    from sevenrad_ee.top_polluters.cache import cache

    monkeypatch.setattr(config.settings, "cache_dir", cache_dir)

    # Clear cache before each test
    cache.clear()

    return cache_dir


@pytest.fixture
def sample_dates() -> tuple[date, date]:
    """
    Create sample date range for testing.

    Returns:
        Tuple of (start_date, end_date)

    """
    return (date(2024, 1, 1), date(2024, 12, 31))
