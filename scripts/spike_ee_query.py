"""
Spike script to validate Earth Engine VIIRS DNB query approach.

This is throwaway code to de-risk the implementation by proving
the EE integration works before building abstractions.

Usage:
    python scripts/spike_ee_query.py
"""

import json
from datetime import date
from pathlib import Path

import ee

# Initialize Earth Engine
ee.Initialize()

# Hardcoded test values
REGION_PATH = Path("tests/fixtures/sf_region.geojson")
START_DATE = date(2024, 1, 1)
END_DATE = date(2024, 12, 31)
TOP_N = 20

# Load GeoJSON
with REGION_PATH.open() as f:
    geojson = json.load(f)

# Parse GeoJSON to ee.Geometry
if geojson["type"] == "FeatureCollection":
    geometry = ee.Geometry(geojson["features"][0]["geometry"])
elif geojson["type"] == "Feature":
    geometry = ee.Geometry(geojson["geometry"])
else:
    geometry = ee.Geometry(geojson)

# Query VIIRS DNB collection
viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
filtered = viirs.filterDate(START_DATE.isoformat(), END_DATE.isoformat()).filterBounds(
    geometry
)

# Temporal aggregation: mean radiance over the date range
mean_radiance = filtered.select("avg_rad").mean()

# Sample at 500m scale
samples = mean_radiance.sample(
    region=geometry,
    scale=500,
    numPixels=10000,  # Max samples to collect
    geometries=True,
)

# Get features and sort by radiance
features = samples.getInfo()["features"]
sorted_features = sorted(
    features, key=lambda f: f["properties"]["avg_rad"], reverse=True
)

# Take top N
top_features = sorted_features[:TOP_N]

# Print results
print(f"\nTop {TOP_N} VIIRS DNB emitters:")  # noqa: T201
print("=" * 60)  # noqa: T201

for i, feature in enumerate(top_features, 1):
    coords = feature["geometry"]["coordinates"]
    radiance = feature["properties"]["avg_rad"]
    print(  # noqa: T201
        f"{i:2d}. lat={coords[1]:.4f}, lon={coords[0]:.4f}, radiance={radiance:.2f}"
    )

print("\n✓ Spike successful! EE integration works.")  # noqa: T201
