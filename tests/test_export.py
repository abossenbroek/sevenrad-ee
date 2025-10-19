"""
Tests for export module.

Tests YAML export functionality including structure validation,
Path serialization, and metadata inclusion.
"""

import datetime
from pathlib import Path

import pytest
import yaml
from sevenrad_ee.top_polluters.export import export_yaml
from sevenrad_ee.top_polluters.models import (
    Address,
    Business,
    Coordinates,
    StreetViewImages,
    TopEmitter,
)


class TestExportYAML:
    """Test YAML export functionality."""

    def test_export_yaml_creates_file(
        self, sample_top_emitters: list[TopEmitter], tmp_path: Path
    ) -> None:
        """Export YAML creates output file."""
        output_path = tmp_path / "test_output.yml"

        export_yaml(sample_top_emitters, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_export_yaml_structure(
        self, sample_top_emitters: list[TopEmitter], tmp_path: Path
    ) -> None:
        """Export YAML has expected structure with metadata and emitters."""
        output_path = tmp_path / "test_output.yml"

        export_yaml(sample_top_emitters, output_path)

        with output_path.open() as f:
            data = yaml.safe_load(f)

        # Verify top-level structure
        assert "metadata" in data
        assert "emitters" in data

        # Verify metadata structure
        assert "exported_at" in data["metadata"]
        assert "total_emitters" in data["metadata"]
        assert "source" in data["metadata"]

        # Verify metadata values
        assert data["metadata"]["total_emitters"] == len(sample_top_emitters)
        assert data["metadata"]["source"] == "NOAA VIIRS DNB Monthly"

        # Verify emitters list
        assert isinstance(data["emitters"], list)
        assert len(data["emitters"]) == len(sample_top_emitters)

    def test_export_yaml_emitter_fields(
        self, sample_top_emitters: list[TopEmitter], tmp_path: Path
    ) -> None:
        """Export YAML includes all emitter fields."""
        output_path = tmp_path / "test_output.yml"

        export_yaml(sample_top_emitters, output_path)

        with output_path.open() as f:
            data = yaml.safe_load(f)

        # Check first emitter
        emitter = data["emitters"][0]

        assert "rank" in emitter
        assert "coordinates" in emitter
        assert "avg_radiance" in emitter

        # Verify coordinates structure
        assert "lat" in emitter["coordinates"]
        assert "lon" in emitter["coordinates"]

        # Verify values match
        assert emitter["rank"] == sample_top_emitters[0].rank
        assert emitter["coordinates"]["lat"] == sample_top_emitters[0].coordinates.lat
        assert emitter["coordinates"]["lon"] == sample_top_emitters[0].coordinates.lon
        assert emitter["avg_radiance"] == sample_top_emitters[0].avg_radiance

    def test_export_yaml_with_address(self, tmp_path: Path) -> None:
        """Export YAML includes address data when present."""
        emitter = TopEmitter(
            rank=1,
            coordinates=Coordinates(lat=37.8, lon=-122.4),
            avg_radiance=50.0,
            address=Address(
                formatted="123 Main St, San Francisco, CA 94102, USA",
                street="Main St",
                city="San Francisco",
                state="California",
                country="United States",
                postal_code="94102",
                place_id="ChIJtest",
                quality_score=0.95,
                match_type="ROOFTOP",
            ),
        )

        output_path = tmp_path / "test_output.yml"
        export_yaml([emitter], output_path)

        with output_path.open() as f:
            data = yaml.safe_load(f)

        emitter_data = data["emitters"][0]

        assert "address" in emitter_data
        assert emitter_data["address"]["formatted"] == "123 Main St, San Francisco, CA 94102, USA"
        assert emitter_data["address"]["city"] == "San Francisco"
        assert emitter_data["address"]["quality_score"] == pytest.approx(0.95)

    def test_export_yaml_with_businesses(self, tmp_path: Path) -> None:
        """Export YAML includes business data when present."""
        emitter = TopEmitter(
            rank=1,
            coordinates=Coordinates(lat=37.8, lon=-122.4),
            avg_radiance=50.0,
            businesses=[
                Business(
                    name="Stadium A",
                    place_id="ChIJA",
                    types=["stadium"],
                    vicinity="123 Stadium Way",
                    distance_m=25.5,
                    rating=4.8,
                    relevance_score=0.92,
                ),
                Business(
                    name="Restaurant B",
                    place_id="ChIJB",
                    types=["restaurant"],
                    vicinity="456 Main St",
                    distance_m=50.2,
                    rating=4.2,
                    relevance_score=0.75,
                ),
            ],
        )

        output_path = tmp_path / "test_output.yml"
        export_yaml([emitter], output_path)

        with output_path.open() as f:
            data = yaml.safe_load(f)

        emitter_data = data["emitters"][0]

        assert "businesses" in emitter_data
        assert len(emitter_data["businesses"]) == 2  # noqa: PLR2004

        # Verify first business
        business = emitter_data["businesses"][0]
        assert business["name"] == "Stadium A"
        assert business["place_id"] == "ChIJA"
        assert "stadium" in business["types"]
        assert business["distance_m"] == pytest.approx(25.5)
        assert business["rating"] == pytest.approx(4.8)
        assert business["relevance_score"] == pytest.approx(0.92)

    def test_export_yaml_with_streetview(self, tmp_path: Path) -> None:
        """Export YAML includes Street View image paths when present."""
        # Create dummy image files
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        north_path = image_dir / "north.jpg"
        south_path = image_dir / "south.jpg"
        north_path.write_bytes(b"fake_image")
        south_path.write_bytes(b"fake_image")

        emitter = TopEmitter(
            rank=1,
            coordinates=Coordinates(lat=37.8, lon=-122.4),
            avg_radiance=50.0,
            streetview=StreetViewImages(
                available=True,
                north=north_path,
                south=south_path,
                east=None,
                west=None,
            ),
        )

        output_path = tmp_path / "test_output.yml"
        export_yaml([emitter], output_path)

        with output_path.open() as f:
            data = yaml.safe_load(f)

        emitter_data = data["emitters"][0]

        assert "streetview" in emitter_data
        assert emitter_data["streetview"]["available"] is True
        assert "north" in emitter_data["streetview"]
        assert "south" in emitter_data["streetview"]

        # Verify paths are serialized as strings
        assert isinstance(emitter_data["streetview"]["north"], str)
        assert isinstance(emitter_data["streetview"]["south"], str)

    def test_export_yaml_streetview_unavailable(self, tmp_path: Path) -> None:
        """Export YAML handles unavailable Street View correctly."""
        emitter = TopEmitter(
            rank=1,
            coordinates=Coordinates(lat=37.8, lon=-122.4),
            avg_radiance=50.0,
            streetview=StreetViewImages(available=False),
        )

        output_path = tmp_path / "test_output.yml"
        export_yaml([emitter], output_path)

        with output_path.open() as f:
            data = yaml.safe_load(f)

        emitter_data = data["emitters"][0]

        assert "streetview" in emitter_data
        assert emitter_data["streetview"]["available"] is False
        assert emitter_data["streetview"]["north"] is None
        assert emitter_data["streetview"]["south"] is None
        assert emitter_data["streetview"]["east"] is None
        assert emitter_data["streetview"]["west"] is None

    def test_export_yaml_empty_list(self, tmp_path: Path) -> None:
        """Export YAML handles empty emitter list."""
        output_path = tmp_path / "test_output.yml"

        export_yaml([], output_path)

        with output_path.open() as f:
            data = yaml.safe_load(f)

        assert data["metadata"]["total_emitters"] == 0
        assert data["emitters"] == []

    def test_export_yaml_encoding(self, tmp_path: Path) -> None:
        """Export YAML uses UTF-8 encoding."""
        emitter = TopEmitter(
            rank=1,
            coordinates=Coordinates(lat=37.8, lon=-122.4),
            avg_radiance=50.0,
            address=Address(
                formatted="Café España, 123 Zürich",  # Unicode characters
                place_id="ChIJtest",
                quality_score=0.9,
                match_type="ROOFTOP",
            ),
        )

        output_path = tmp_path / "test_output.yml"
        export_yaml([emitter], output_path)

        # Verify file is UTF-8 encoded and data roundtrips correctly
        with output_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
            assert data["emitters"][0]["address"]["formatted"] == "Café España, 123 Zürich"

    def test_export_yaml_preserves_order(
        self, sample_top_emitters: list[TopEmitter], tmp_path: Path
    ) -> None:
        """Export YAML preserves emitter order (sorted by rank)."""
        output_path = tmp_path / "test_output.yml"

        export_yaml(sample_top_emitters, output_path)

        with output_path.open() as f:
            data = yaml.safe_load(f)

        # Verify ranks are in order
        ranks = [emitter["rank"] for emitter in data["emitters"]]
        assert ranks == sorted(ranks)

    def test_export_yaml_timestamp_format(
        self, sample_top_emitters: list[TopEmitter], tmp_path: Path
    ) -> None:
        """Export YAML includes valid ISO 8601 timestamp."""
        output_path = tmp_path / "test_output.yml"

        export_yaml(sample_top_emitters, output_path)

        with output_path.open() as f:
            data = yaml.safe_load(f)

        timestamp_str = data["metadata"]["exported_at"]

        # Should be valid ISO 8601 format
        timestamp = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        assert timestamp.tzinfo is not None  # Should have timezone info

    def test_export_yaml_file_permissions(
        self, sample_top_emitters: list[TopEmitter], tmp_path: Path
    ) -> None:
        """Export YAML creates readable file."""
        output_path = tmp_path / "test_output.yml"

        export_yaml(sample_top_emitters, output_path)

        # Verify file is readable
        assert output_path.is_file()
        with output_path.open() as f:
            content = f.read()
            assert len(content) > 0

    def test_export_yaml_overwrites_existing(
        self, sample_top_emitters: list[TopEmitter], tmp_path: Path
    ) -> None:
        """Export YAML overwrites existing file."""
        output_path = tmp_path / "test_output.yml"

        # Create initial file
        output_path.write_text("old content")

        # Export should overwrite
        export_yaml(sample_top_emitters, output_path)

        with output_path.open() as f:
            content = f.read()
            assert "old content" not in content
            assert "metadata" in content
