"""
Tests for enrichment module.

Tests Google Maps API integrations (geocoding, business search, Street View)
with quality/relevance scoring. Uses VCR.py for deterministic HTTP replay.
"""

import math
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from sevenrad_ee.top_polluters.enrichment import (
    _calculate_address_quality,
    _calculate_business_relevance,
    find_nearby_businesses,
    geocode_coordinates,
    get_street_view_images,
    haversine_distance,
    rate_limited,
)
from sevenrad_ee.top_polluters.models import (
    Address,
    Business,
    Coordinates,
    StreetViewImages,
)


class TestHaversineDistance:
    """Test Haversine distance calculation."""

    def test_distance_same_point(self) -> None:
        """Distance between same point is zero."""
        coord = Coordinates(lat=37.8, lon=-122.4)
        distance = haversine_distance(coord, coord)
        assert distance == pytest.approx(0.0, abs=1.0)

    def test_distance_known_points(self) -> None:
        """Distance calculation matches known values."""
        # SF Downtown to SF Airport (approximately 17-18km)
        downtown = Coordinates(lat=37.7749, lon=-122.4194)
        airport = Coordinates(lat=37.6213, lon=-122.3790)

        distance = haversine_distance(downtown, airport)

        # Should be around 17-19 km (haversine gives ~17.7km)
        assert 17000 < distance < 19000

    def test_distance_symmetry(self) -> None:
        """Distance is symmetric (A to B == B to A)."""
        coord1 = Coordinates(lat=37.8, lon=-122.4)
        coord2 = Coordinates(lat=37.9, lon=-122.5)

        distance1 = haversine_distance(coord1, coord2)
        distance2 = haversine_distance(coord2, coord1)

        assert distance1 == pytest.approx(distance2)


class TestRateLimited:
    """Test rate limiting decorator."""

    def test_rate_limited_enforces_delay(self) -> None:
        """Rate limited decorator enforces minimum delay between calls."""
        call_times = []

        @rate_limited(100)  # 100ms delay
        def test_func() -> None:
            call_times.append(time.monotonic())

        # Call three times
        test_func()
        test_func()
        test_func()

        # Verify delays
        assert len(call_times) == 3
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        # Should be at least 100ms between calls (with 10ms tolerance)
        assert delay1 >= 0.09
        assert delay2 >= 0.09

    def test_rate_limited_preserves_return_value(self) -> None:
        """Rate limited decorator preserves function return value."""

        @rate_limited(10)
        def add(a: int, b: int) -> int:
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_rate_limited_preserves_function_name(self) -> None:
        """Rate limited decorator preserves function metadata."""

        @rate_limited(10)
        def example_function() -> None:
            """Return nothing (test function)."""

        assert example_function.__name__ == "example_function"
        assert example_function.__doc__ == "Return nothing (test function)."


class TestCalculateAddressQuality:
    """Test address quality scoring."""

    def test_quality_rooftop_highest(self) -> None:
        """ROOFTOP location type gets highest quality score."""
        result = {
            "geometry": {"location_type": "ROOFTOP"},
            "address_components": [
                {"types": ["street_number"], "long_name": "123"},
                {"types": ["route"], "long_name": "Main St"},
                {"types": ["postal_code"], "long_name": "94102"},
            ],
        }

        quality = _calculate_address_quality(result)

        assert quality == pytest.approx(1.0)

    def test_quality_approximate_lowest(self) -> None:
        """APPROXIMATE location type gets lower quality score."""
        result = {
            "geometry": {"location_type": "APPROXIMATE"},
            "address_components": [],
        }

        quality = _calculate_address_quality(result)

        # Approximate with no components: 0.5 * 0.8 = 0.4
        assert quality == pytest.approx(0.4)

    def test_quality_partial_match_penalty(self) -> None:
        """Partial match applies 0.8x penalty."""
        result_full = {
            "geometry": {"location_type": "ROOFTOP"},
            "partial_match": False,
            "address_components": [],
        }

        result_partial = {
            "geometry": {"location_type": "ROOFTOP"},
            "partial_match": True,
            "address_components": [],
        }

        quality_full = _calculate_address_quality(result_full)
        quality_partial = _calculate_address_quality(result_partial)

        assert quality_partial == pytest.approx(quality_full * 0.8)

    def test_quality_completeness_bonus(self) -> None:
        """Complete address components increase quality score."""
        result_incomplete = {
            "geometry": {"location_type": "ROOFTOP"},
            "address_components": [],
        }

        result_complete = {
            "geometry": {"location_type": "ROOFTOP"},
            "address_components": [
                {"types": ["street_number"], "long_name": "123"},
                {"types": ["route"], "long_name": "Main St"},
                {"types": ["postal_code"], "long_name": "94102"},
            ],
        }

        quality_incomplete = _calculate_address_quality(result_incomplete)
        quality_complete = _calculate_address_quality(result_complete)

        # Complete should be 1.0, incomplete should be 0.8
        assert quality_complete == pytest.approx(1.0)
        assert quality_incomplete == pytest.approx(0.8)


class TestCalculateBusinessRelevance:
    """Test business relevance scoring."""

    def test_relevance_proximity_score(self, sample_coordinates: Coordinates) -> None:
        """Proximity score uses exponential decay."""
        # Business at same location
        business_data_same = {
            "geometry": {"location": {"lat": 37.8, "lng": -122.4}},
            "types": [],
        }

        relevance_same, distance_same = _calculate_business_relevance(
            business_data_same, sample_coordinates
        )

        # Business 50m away
        business_data_far = {
            "geometry": {"location": {"lat": 37.80045, "lng": -122.4}},
            "types": [],
        }

        relevance_far, distance_far = _calculate_business_relevance(
            business_data_far, sample_coordinates
        )

        # Same location should have higher relevance
        assert relevance_same > relevance_far
        assert distance_same == pytest.approx(0.0, abs=1.0)
        assert distance_far > 0

    def test_relevance_type_weighting(self, sample_coordinates: Coordinates) -> None:
        """High-emission business types get 1.5x multiplier."""
        # Regular business
        business_regular = {
            "geometry": {"location": {"lat": 37.8, "lng": -122.4}},
            "types": ["restaurant"],
        }

        # Stadium (high-emission type)
        business_stadium = {
            "geometry": {"location": {"lat": 37.8, "lng": -122.4}},
            "types": ["stadium"],
        }

        relevance_regular, _ = _calculate_business_relevance(
            business_regular, sample_coordinates
        )
        relevance_stadium, _ = _calculate_business_relevance(
            business_stadium, sample_coordinates
        )

        # Stadium should have 1.5x higher relevance
        assert relevance_stadium == pytest.approx(relevance_regular * 1.5)

    def test_relevance_rating_score(self, sample_coordinates: Coordinates) -> None:
        """Rating affects relevance score."""
        business_no_rating = {
            "geometry": {"location": {"lat": 37.8, "lng": -122.4}},
            "types": [],
        }

        business_high_rating = {
            "geometry": {"location": {"lat": 37.8, "lng": -122.4}},
            "types": [],
            "rating": 5.0,
        }

        relevance_no_rating, _ = _calculate_business_relevance(
            business_no_rating, sample_coordinates
        )
        relevance_high_rating, _ = _calculate_business_relevance(
            business_high_rating, sample_coordinates
        )

        # High rating should increase relevance
        assert relevance_high_rating > relevance_no_rating


class TestGeocodeCoordinates:
    """Test reverse geocoding with caching."""

    def test_geocode_coordinates_success(
        self, sample_coordinates: Coordinates, temp_cache_dir: Path
    ) -> None:
        """Geocode coordinates returns Address model."""
        mock_result = {
            "formatted_address": "123 Main St, San Francisco, CA 94102, USA",
            "place_id": "ChIJtest123",
            "geometry": {"location_type": "ROOFTOP"},
            "address_components": [
                {"types": ["street_number"], "long_name": "123"},
                {"types": ["route"], "long_name": "Main St"},
                {"types": ["locality"], "long_name": "San Francisco"},
                {"types": ["administrative_area_level_1"], "long_name": "California"},
                {"types": ["country"], "long_name": "United States"},
                {"types": ["postal_code"], "long_name": "94102"},
            ],
        }

        with patch(
            "sevenrad_ee.top_polluters.enrichment._geocode_coordinates_api"
        ) as mock_api:
            mock_api.return_value = mock_result

            address = geocode_coordinates(sample_coordinates)

            assert isinstance(address, Address)
            assert address.formatted == "123 Main St, San Francisco, CA 94102, USA"
            assert address.street == "Main St"
            assert address.city == "San Francisco"
            assert address.state == "California"
            assert address.country == "United States"
            assert address.postal_code == "94102"
            assert address.place_id == "ChIJtest123"
            assert 0.0 <= address.quality_score <= 1.0
            assert address.match_type == "ROOFTOP"

    def test_geocode_coordinates_uses_cache(
        self, sample_coordinates: Coordinates, temp_cache_dir: Path
    ) -> None:
        """Geocode coordinates uses cache on subsequent calls."""
        mock_result = {
            "formatted_address": "123 Main St, San Francisco, CA 94102, USA",
            "place_id": "ChIJtest123",
            "geometry": {"location_type": "ROOFTOP"},
            "address_components": [],
        }

        with patch(
            "sevenrad_ee.top_polluters.enrichment._geocode_coordinates_api"
        ) as mock_api:
            mock_api.return_value = mock_result

            # First call
            address1 = geocode_coordinates(sample_coordinates)
            call_count = mock_api.call_count

            # Second call - should use cache
            address2 = geocode_coordinates(sample_coordinates)

            assert mock_api.call_count == call_count  # No additional calls
            assert address1 is not None
            assert address2 is not None
            assert address1.formatted == address2.formatted

    def test_geocode_coordinates_no_api_key(
        self, sample_coordinates: Coordinates, temp_cache_dir: Path
    ) -> None:
        """Geocode coordinates returns None when API key missing."""
        with patch(
            "sevenrad_ee.top_polluters.enrichment.settings.google_maps_api_key", None
        ):
            address = geocode_coordinates(sample_coordinates)

            assert address is None


class TestFindNearbyBusinesses:
    """Test nearby business search with caching."""

    def test_find_nearby_businesses_success(
        self, sample_coordinates: Coordinates, temp_cache_dir: Path
    ) -> None:
        """Find nearby businesses returns sorted list of Business models."""
        mock_results = [
            {
                "name": "Restaurant A",
                "place_id": "ChIJA",
                "types": ["restaurant"],
                "vicinity": "123 Main St",
                "rating": 4.5,
                "geometry": {"location": {"lat": 37.8, "lng": -122.4}},
            },
            {
                "name": "Stadium B",
                "place_id": "ChIJB",
                "types": ["stadium"],
                "vicinity": "456 Stadium Way",
                "rating": 4.8,
                "geometry": {"location": {"lat": 37.80001, "lng": -122.40001}},
            },
        ]

        with patch(
            "sevenrad_ee.top_polluters.enrichment._find_nearby_businesses_api"
        ) as mock_api:
            mock_api.return_value = mock_results

            businesses = find_nearby_businesses(sample_coordinates, radius=100)

            assert len(businesses) == 2
            assert all(isinstance(b, Business) for b in businesses)

            # Verify sorted by relevance score descending
            assert businesses[0].relevance_score >= businesses[1].relevance_score

            # Stadium should be first (higher type weighting)
            assert businesses[0].name == "Stadium B"

    def test_find_nearby_businesses_uses_cache(
        self, sample_coordinates: Coordinates, temp_cache_dir: Path
    ) -> None:
        """Find nearby businesses uses cache on subsequent calls."""
        mock_results = [
            {
                "name": "Restaurant A",
                "place_id": "ChIJA",
                "types": ["restaurant"],
                "vicinity": "123 Main St",
                "geometry": {"location": {"lat": 37.8, "lng": -122.4}},
            }
        ]

        with patch(
            "sevenrad_ee.top_polluters.enrichment._find_nearby_businesses_api"
        ) as mock_api:
            mock_api.return_value = mock_results

            # First call
            businesses1 = find_nearby_businesses(sample_coordinates, radius=100)
            call_count = mock_api.call_count

            # Second call - should use cache
            businesses2 = find_nearby_businesses(sample_coordinates, radius=100)

            assert mock_api.call_count == call_count  # No additional calls
            assert len(businesses1) == len(businesses2)

    def test_find_nearby_businesses_default_radius(
        self, sample_coordinates: Coordinates, temp_cache_dir: Path
    ) -> None:
        """Find nearby businesses uses config default radius when None."""
        with (
            patch(
                "sevenrad_ee.top_polluters.enrichment._find_nearby_businesses_api"
            ) as mock_api,
            patch(
                "sevenrad_ee.top_polluters.enrichment.settings.business_search_radius_m",
                200,
            ),
        ):
            mock_api.return_value = []

            find_nearby_businesses(sample_coordinates)

            # Verify called with default radius from config
            mock_api.assert_called_once_with(sample_coordinates, 200)


class TestGetStreetViewImages:
    """Test Street View image download with caching."""

    def test_get_street_view_images_available(
        self, sample_coordinates: Coordinates, temp_cache_dir: Path
    ) -> None:
        """Get Street View images downloads and saves images."""
        with (
            patch(
                "sevenrad_ee.top_polluters.enrichment._check_streetview_availability"
            ) as mock_check,
            patch(
                "sevenrad_ee.top_polluters.enrichment.googlemaps.Client"
            ) as mock_client,
            patch(
                "sevenrad_ee.top_polluters.enrichment.settings.google_maps_api_key",
                "fake_key",
            ),
        ):
            mock_check.return_value = True
            mock_gmaps = Mock()
            mock_gmaps.streetview.return_value = b"fake_image_data"
            mock_client.return_value = mock_gmaps

            images = get_street_view_images(sample_coordinates, rank=1)

            assert isinstance(images, StreetViewImages)
            assert images.available is True
            assert images.north is not None
            assert images.south is not None
            assert images.east is not None
            assert images.west is not None

            # Verify images were saved
            assert images.north.exists() if images.north else False
            assert images.south.exists() if images.south else False
            assert images.east.exists() if images.east else False
            assert images.west.exists() if images.west else False

    def test_get_street_view_images_unavailable(
        self, sample_coordinates: Coordinates, temp_cache_dir: Path
    ) -> None:
        """Get Street View images returns unavailable when no imagery."""
        with patch(
            "sevenrad_ee.top_polluters.enrichment._check_streetview_availability"
        ) as mock_check:
            mock_check.return_value = False

            images = get_street_view_images(sample_coordinates, rank=1)

            assert isinstance(images, StreetViewImages)
            assert images.available is False
            assert images.north is None
            assert images.south is None
            assert images.east is None
            assert images.west is None

    def test_get_street_view_images_uses_cache(
        self, sample_coordinates: Coordinates, temp_cache_dir: Path
    ) -> None:
        """Get Street View images uses cache on subsequent calls."""
        with (
            patch(
                "sevenrad_ee.top_polluters.enrichment._check_streetview_availability"
            ) as mock_check,
        ):
            mock_check.return_value = False

            # First call
            images1 = get_street_view_images(sample_coordinates, rank=1)
            call_count = mock_check.call_count

            # Second call - should use cache
            images2 = get_street_view_images(sample_coordinates, rank=1)

            assert mock_check.call_count == call_count  # No additional calls
            assert images1.available == images2.available
