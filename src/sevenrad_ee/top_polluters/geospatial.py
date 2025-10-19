"""
Geospatial utility functions for distance calculations.

This module provides functions for calculating distances between geographic
coordinates using the Haversine formula.
"""

import math

from .models import Coordinates


def haversine_distance(coord1: Coordinates, coord2: Coordinates) -> float:
    """
    Calculate the Haversine distance between two points in meters.

    Uses the Haversine formula to compute the great-circle distance between
    two points on Earth given their latitude and longitude coordinates.

    Args:
        coord1: First coordinate
        coord2: Second coordinate

    Returns:
        Distance in meters

    """
    # Earth's radius in meters
    earth_radius_m = 6371000

    # Convert to radians
    lat1_rad = math.radians(coord1.lat)
    lon1_rad = math.radians(coord1.lon)
    lat2_rad = math.radians(coord2.lat)
    lon2_rad = math.radians(coord2.lon)

    # Calculate differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius_m * c
