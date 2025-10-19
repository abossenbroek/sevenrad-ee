"""
Application configuration management.

This module uses pydantic-settings to load and validate configuration
from environment variables and a .env file.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_maps_api_key: str | None = Field(
        default=None,
        description="Google Cloud API key for Geocoding, Places, and Street View APIs.",
    )
    ee_project_id: str | None = Field(
        default=None,
        description="Google Earth Engine project ID.",
    )
    cache_dir: Path = Field(
        default=Path("cache"),
        description="Directory for caching API responses and EE results.",
    )
    viirs_scale_m: int = Field(
        default=750,
        description=(
            "VIIRS DNB sensor resolution in meters (750m x 750m pixels at nadir)."
        ),
    )
    business_search_radius_m: int = Field(
        default=100,
        description="Search radius for nearby businesses in meters.",
    )
    perplexity_api_key: str | None = Field(
        default=None,
        description=(
            "Perplexity API key for greenhouse attribution "
            "(or set PERPLEXITY_API_KEY env var)."
        ),
    )


# Create a singleton instance of the settings
settings = Settings()
