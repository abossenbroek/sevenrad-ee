"""
Tests for earth_engine module.

Tests VIIRS DNB query logic, GeoJSON parsing, cache key generation,
and top emitters extraction with mocked Earth Engine API calls.
"""

import hashlib
import json
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from sevenrad_ee.top_polluters.earth_engine import (
    _generate_cache_key,
    _parse_geojson,
    get_top_emitters,
)
from sevenrad_ee.top_polluters.models import TopEmitter


class TestParseGeoJSON:
    """Test GeoJSON parsing for different formats."""

    def test_parse_feature_collection(
        self, sample_region_geojson: Path, mock_ee_geometry: Mock
    ) -> None:
        """Parse FeatureCollection extracts first feature geometry."""
        with patch("sevenrad_ee.top_polluters.earth_engine.ee.Geometry") as mock_geom:
            mock_geom.return_value = mock_ee_geometry

            result = _parse_geojson(sample_region_geojson)

            assert result == mock_ee_geometry
            # Verify it called ee.Geometry with first feature's geometry
            geojson_data = json.loads(sample_region_geojson.read_text())
            expected_geometry = geojson_data["features"][0]["geometry"]
            mock_geom.assert_called_once_with(expected_geometry)

    def test_parse_feature(
        self, sample_feature_geojson: Path, mock_ee_geometry: Mock
    ) -> None:
        """Parse Feature extracts geometry directly."""
        with patch("sevenrad_ee.top_polluters.earth_engine.ee.Geometry") as mock_geom:
            mock_geom.return_value = mock_ee_geometry

            result = _parse_geojson(sample_feature_geojson)

            assert result == mock_ee_geometry
            # Verify it called ee.Geometry with feature's geometry
            geojson_data = json.loads(sample_feature_geojson.read_text())
            expected_geometry = geojson_data["geometry"]
            mock_geom.assert_called_once_with(expected_geometry)

    def test_parse_raw_geometry(
        self, sample_geometry_geojson: Path, mock_ee_geometry: Mock
    ) -> None:
        """Parse raw Geometry uses directly."""
        with patch("sevenrad_ee.top_polluters.earth_engine.ee.Geometry") as mock_geom:
            mock_geom.return_value = mock_ee_geometry

            result = _parse_geojson(sample_geometry_geojson)

            assert result == mock_ee_geometry
            # Verify it called ee.Geometry with raw geometry
            geojson_data = json.loads(sample_geometry_geojson.read_text())
            mock_geom.assert_called_once_with(geojson_data)

    def test_parse_empty_feature_collection_raises(self, tmp_path: Path) -> None:
        """Parse FeatureCollection with no features raises ValueError."""
        empty_geojson = {"type": "FeatureCollection", "features": []}
        geojson_path = tmp_path / "empty.geojson"
        geojson_path.write_text(json.dumps(empty_geojson))

        with pytest.raises(ValueError, match="FeatureCollection has no features"):
            _parse_geojson(geojson_path)


class TestGenerateCacheKey:
    """Test deterministic cache key generation."""

    def test_cache_key_format(
        self, sample_region_geojson: Path, sample_dates: tuple[date, date]
    ) -> None:
        """Cache key has expected format with prefix and hash."""
        start_date, end_date = sample_dates
        key = _generate_cache_key(sample_region_geojson, start_date, end_date, 20)

        assert key.startswith("ee_emitters:")
        assert len(key) == len("ee_emitters:") + 16  # 16 hex chars

    def test_cache_key_deterministic(
        self, sample_region_geojson: Path, sample_dates: tuple[date, date]
    ) -> None:
        """Cache key is deterministic for same inputs."""
        start_date, end_date = sample_dates
        key1 = _generate_cache_key(sample_region_geojson, start_date, end_date, 20)
        key2 = _generate_cache_key(sample_region_geojson, start_date, end_date, 20)

        assert key1 == key2

    def test_cache_key_changes_with_inputs(
        self, sample_region_geojson: Path, sample_dates: tuple[date, date]
    ) -> None:
        """Cache key changes when inputs change."""
        start_date, end_date = sample_dates
        key1 = _generate_cache_key(sample_region_geojson, start_date, end_date, 20)
        key2 = _generate_cache_key(
            sample_region_geojson, start_date, end_date, 30
        )  # Different N
        key3 = _generate_cache_key(
            sample_region_geojson, date(2024, 6, 1), end_date, 20
        )  # Different start

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_cache_key_includes_geojson_content(
        self, tmp_path: Path, sample_dates: tuple[date, date]
    ) -> None:
        """Cache key changes when GeoJSON content changes."""
        start_date, end_date = sample_dates

        # Create two GeoJSON files with different content
        geojson1 = {
            "type": "Polygon",
            "coordinates": [[[-122.5, 37.7], [-122.3, 37.7], [-122.5, 37.7]]],
        }
        geojson2 = {
            "type": "Polygon",
            "coordinates": [[[-123.5, 38.7], [-123.3, 38.7], [-123.5, 38.7]]],
        }

        path1 = tmp_path / "region1.geojson"
        path2 = tmp_path / "region2.geojson"
        path1.write_text(json.dumps(geojson1))
        path2.write_text(json.dumps(geojson2))

        key1 = _generate_cache_key(path1, start_date, end_date, 20)
        key2 = _generate_cache_key(path2, start_date, end_date, 20)

        assert key1 != key2


class TestGetTopEmitters:
    """Test VIIRS DNB top emitters query logic."""

    def test_get_top_emitters_success(
        self,
        sample_region_geojson: Path,
        sample_dates: tuple[date, date],
        mock_ee_geometry: Mock,
        mock_ee_image_collection: Mock,
        mock_ee_feature_collection_data: dict,
        temp_cache_dir: Path,
    ) -> None:
        """Get top emitters returns sorted list of TopEmitter models."""
        start_date, end_date = sample_dates

        with (
            patch("sevenrad_ee.top_polluters.earth_engine.ee.Geometry") as mock_geom,
            patch(
                "sevenrad_ee.top_polluters.earth_engine.ee.ImageCollection"
            ) as mock_ic,
        ):
            # Setup mocks
            mock_geom.return_value = mock_ee_geometry
            mock_ic.return_value = mock_ee_image_collection

            # Mock the sample result
            mock_sample = Mock()
            mock_sample.getInfo.return_value = mock_ee_feature_collection_data

            mock_mean_image = Mock()
            mock_mean_image.sample.return_value = mock_sample
            mock_ee_image_collection.select.return_value.mean.return_value = (
                mock_mean_image
            )

            # Execute
            emitters = get_top_emitters(sample_region_geojson, start_date, end_date, 3)

            # Verify results
            assert len(emitters) == 3
            assert all(isinstance(e, TopEmitter) for e in emitters)

            # Verify sorted by radiance descending
            assert emitters[0].avg_radiance == 52.1
            assert emitters[1].avg_radiance == 45.5
            assert emitters[2].avg_radiance == 38.2

            # Verify ranks
            assert emitters[0].rank == 1
            assert emitters[1].rank == 2
            assert emitters[2].rank == 3

            # Verify coordinates
            assert emitters[0].coordinates.lat == 37.85
            assert emitters[0].coordinates.lon == -122.45

    def test_get_top_emitters_respects_n_limit(
        self,
        sample_region_geojson: Path,
        sample_dates: tuple[date, date],
        mock_ee_geometry: Mock,
        mock_ee_image_collection: Mock,
        mock_ee_feature_collection_data: dict,
        temp_cache_dir: Path,
    ) -> None:
        """Get top emitters returns only N results."""
        start_date, end_date = sample_dates

        with (
            patch("sevenrad_ee.top_polluters.earth_engine.ee.Geometry") as mock_geom,
            patch(
                "sevenrad_ee.top_polluters.earth_engine.ee.ImageCollection"
            ) as mock_ic,
        ):
            mock_geom.return_value = mock_ee_geometry
            mock_ic.return_value = mock_ee_image_collection

            mock_sample = Mock()
            mock_sample.getInfo.return_value = mock_ee_feature_collection_data

            mock_mean_image = Mock()
            mock_mean_image.sample.return_value = mock_sample
            mock_ee_image_collection.select.return_value.mean.return_value = (
                mock_mean_image
            )

            # Request only top 2
            emitters = get_top_emitters(sample_region_geojson, start_date, end_date, 2)

            assert len(emitters) == 2
            assert emitters[0].rank == 1
            assert emitters[1].rank == 2

    def test_get_top_emitters_no_results(
        self,
        sample_region_geojson: Path,
        sample_dates: tuple[date, date],
        mock_ee_geometry: Mock,
        mock_ee_image_collection: Mock,
        temp_cache_dir: Path,
    ) -> None:
        """Get top emitters returns empty list when no features found."""
        start_date, end_date = sample_dates

        with (
            patch("sevenrad_ee.top_polluters.earth_engine.ee.Geometry") as mock_geom,
            patch(
                "sevenrad_ee.top_polluters.earth_engine.ee.ImageCollection"
            ) as mock_ic,
        ):
            mock_geom.return_value = mock_ee_geometry
            mock_ic.return_value = mock_ee_image_collection

            # Mock empty result
            mock_sample = Mock()
            mock_sample.getInfo.return_value = {"features": []}

            mock_mean_image = Mock()
            mock_mean_image.sample.return_value = mock_sample
            mock_ee_image_collection.select.return_value.mean.return_value = (
                mock_mean_image
            )

            emitters = get_top_emitters(sample_region_geojson, start_date, end_date, 20)

            assert emitters == []

    def test_get_top_emitters_invalid_dates_raises(
        self,
        sample_region_geojson: Path,
        temp_cache_dir: Path,
    ) -> None:
        """Get top emitters raises ValueError for invalid date range."""
        start_date = date(2024, 12, 31)
        end_date = date(2024, 1, 1)  # Before start

        with pytest.raises(ValueError, match="End date .* must be >= start date"):
            get_top_emitters(sample_region_geojson, start_date, end_date, 20)

    def test_get_top_emitters_uses_cache(
        self,
        sample_region_geojson: Path,
        sample_dates: tuple[date, date],
        mock_ee_geometry: Mock,
        mock_ee_image_collection: Mock,
        mock_ee_feature_collection_data: dict,
        temp_cache_dir: Path,
    ) -> None:
        """Get top emitters uses cache on subsequent calls."""
        start_date, end_date = sample_dates

        with (
            patch("sevenrad_ee.top_polluters.earth_engine.ee.Geometry") as mock_geom,
            patch(
                "sevenrad_ee.top_polluters.earth_engine.ee.ImageCollection"
            ) as mock_ic,
        ):
            mock_geom.return_value = mock_ee_geometry
            mock_ic.return_value = mock_ee_image_collection

            mock_sample = Mock()
            mock_sample.getInfo.return_value = mock_ee_feature_collection_data

            mock_mean_image = Mock()
            mock_mean_image.sample.return_value = mock_sample
            mock_ee_image_collection.select.return_value.mean.return_value = (
                mock_mean_image
            )

            # First call - should hit EE
            emitters1 = get_top_emitters(sample_region_geojson, start_date, end_date, 3)
            ee_call_count = mock_ic.call_count

            # Second call - should use cache
            emitters2 = get_top_emitters(sample_region_geojson, start_date, end_date, 3)

            # Verify no additional EE calls
            assert mock_ic.call_count == ee_call_count

            # Verify results are identical
            assert len(emitters1) == len(emitters2)
            assert emitters1[0].avg_radiance == emitters2[0].avg_radiance
