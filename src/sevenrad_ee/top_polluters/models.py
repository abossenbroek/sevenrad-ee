"""
Pydantic models for the VIIRS Top Polluters project.

These models define the data structures used for representing coordinates,
emitters, and enriched data from various APIs.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Coordinates(BaseModel):
    """Represents a geographical coordinate."""

    lat: float = Field(..., description="Latitude, must be between -90 and 90.")
    lon: float = Field(..., description="Longitude, must be between -180 and 180.")

    @field_validator("lat")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        """Validate latitude is within valid range."""
        if not -90.0 <= v <= 90.0:  # noqa: PLR2004
            msg = "Latitude must be between -90 and 90"
            raise ValueError(msg)
        return v

    @field_validator("lon")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        """Validate longitude is within valid range."""
        if not -180.0 <= v <= 180.0:  # noqa: PLR2004
            msg = "Longitude must be between -180 and 180"
            raise ValueError(msg)
        return v


class Address(BaseModel):
    """Represents a geocoded address with a quality score."""

    formatted: str
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    postal_code: Optional[str] = None
    place_id: str
    quality_score: float = Field(
        ..., description="Geocoding quality score (0.0 to 1.0)."
    )
    match_type: str


class Business(BaseModel):
    """Represents a business found via Google Places API."""

    name: str
    place_id: str
    types: list[str]
    vicinity: Optional[str] = None
    distance_m: float = Field(..., description="Distance from source in meters.")
    rating: Optional[float] = None
    relevance_score: float = Field(
        ..., description="Calculated relevance score based on proximity, type, rating."
    )


class StreetViewImages(BaseModel):
    """Represents Street View images from four cardinal directions."""

    available: bool
    north: Optional[Path] = None
    south: Optional[Path] = None
    east: Optional[Path] = None
    west: Optional[Path] = None


class TopEmitter(BaseModel):
    """
    Main data model for a single VIIRS top emitter point.

    Includes coordinates, radiance value, and optional enriched data
    from geocoding, business lookup, and Street View imagery.
    """

    rank: int
    coordinates: Coordinates
    avg_radiance: float
    address: Optional[Address] = None
    businesses: list[Business] = Field(default_factory=list)
    streetview: Optional[StreetViewImages] = None
