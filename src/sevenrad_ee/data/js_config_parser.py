"""
Parse JavaScript configuration files to extract region and palette definitions.

This module provides utilities for parsing GEE JavaScript files that contain
region definitions and visualization parameters for VIIRS data processing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sevenrad_ee.data.palettes import WAVELENGTH_PALETTE_1025


@dataclass
class RegionConfig:
    """Configuration for a geographic region to process."""

    name: str
    west: float
    south: float
    east: float
    north: float

    def to_ee_rectangle(self) -> list[float]:
        """
        Convert to Earth Engine Rectangle coordinates.

        Returns:
            List of coordinates [west, south, east, north]

        """
        return [self.west, self.south, self.east, self.north]


@dataclass
class VisualizationConfig:
    """Visualization configuration for VIIRS data."""

    min_value: float
    max_value: float
    palette: list[str]


def parse_rectangle_region(js_content: str, var_name: str) -> RegionConfig | None:
    """
    Parse a single ee.Geometry.Rectangle region from JavaScript content.

    Args:
        js_content: JavaScript file content
        var_name: Variable name to extract (e.g., 'drc', 'us')

    Returns:
        RegionConfig if found, None otherwise

    """
    # Pattern to match: var name = ee.Geometry.Rectangle([west, south, east, north]);
    pattern = (
        rf"var\s+{var_name}\s*=\s*ee\.Geometry\.Rectangle\(\[\s*"
        r"([-\d.]+),\s*//.*\n\s*"
        r"([-\d.]+),\s*//.*\n\s*"
        r"([-\d.]+),\s*//.*\n\s*"
        r"([-\d.]+)\s*//.*\n\s*\]\)"
    )

    match = re.search(pattern, js_content)
    if match:
        west, south, east, north = map(float, match.groups())
        return RegionConfig(
            name=var_name, west=west, south=south, east=east, north=north
        )
    return None


def parse_all_regions(js_file: Path) -> dict[str, RegionConfig]:
    """
    Parse all region definitions from a JavaScript file.

    Args:
        js_file: Path to JavaScript file containing region definitions

    Returns:
        Dictionary mapping region names to RegionConfig objects

    """
    content = js_file.read_text()

    # Known region variable names from extract_geotiffs.js
    region_names = [
        "drc",
        "us",
        "china",
        "europe",
        "netherlands_full",
        "netherlands_regional",
        "netherlands_westland_moerkappele",
        "netherlands_moerkappele",
    ]

    regions: dict[str, RegionConfig] = {}
    for name in region_names:
        region = parse_rectangle_region(content, name)
        if region:
            regions[name] = region

    return regions


def parse_palette(js_file: Path) -> list[str]:
    """
    Parse color palette array from JavaScript file.

    Args:
        js_file: Path to JavaScript file containing palette definition

    Returns:
        List of hex color strings

    """
    content = js_file.read_text()

    # Pattern to match wavelengthPalette or similar array
    # Matches: var/wavelengthPalette = ['hex', 'hex', ...]
    palette_pattern = r"(?:var\s+)?wavelengthPalette\s*=\s*\[([\s\S]*?)\]"

    match = re.search(palette_pattern, content)
    if not match:
        return []

    palette_content = match.group(1)

    # Extract all hex color codes
    hex_colors = re.findall(r"'([0-9a-fA-F]{6})'", palette_content)

    return hex_colors


def parse_visualization_params(js_file: Path) -> VisualizationConfig | None:
    """
    Parse visualization parameters from JavaScript file.

    Uses built-in wavelength palette if not found in JavaScript files.

    Args:
        js_file: Path to JavaScript file

    Returns:
        VisualizationConfig if found, None otherwise

    """
    content = js_file.read_text()

    # Parse min value
    min_match = re.search(r"min:\s*([\d.]+)", content)
    # Parse max value
    max_match = re.search(r"max:\s*([\d.]+)", content)

    if not (min_match and max_match):
        return None

    min_value = float(min_match.group(1))
    max_value = float(max_match.group(1))

    # Try to get palette from same file or separate palette file
    palette = parse_palette(js_file)

    if not palette:
        # Try loading from separate palette file
        palette_file = js_file.parent / "wavelength_palette_1025.js"
        if palette_file.exists():
            palette = parse_palette(palette_file)

    # Use built-in palette as fallback
    if not palette:
        palette = WAVELENGTH_PALETTE_1025

    return VisualizationConfig(
        min_value=min_value, max_value=max_value, palette=palette
    )


def load_viirs_config(
    config_file: Path,
) -> tuple[dict[str, RegionConfig], VisualizationConfig | None]:
    """
    Load complete VIIRS configuration from JavaScript file.

    Args:
        config_file: Path to main JavaScript configuration file

    Returns:
        Tuple of (regions dict, visualization config)

    """
    regions = parse_all_regions(config_file)
    vis_config = parse_visualization_params(config_file)

    return regions, vis_config
