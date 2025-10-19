# VIIRS Top Polluters

**Documentation Type**: How-to Guide

Identify and analyze top light-emitting locations using NOAA VIIRS DNB (Day/Night Band) satellite data with optional enrichment from Google Maps APIs.

## Overview

This tool queries Google Earth Engine for VIIRS DNB monthly nighttime light data to find the brightest 500m × 500m patches in a geographic region. Results can be enriched with:

- **Reverse geocoding**: Convert coordinates to addresses with quality scoring
- **Business lookup**: Find nearby businesses with relevance scoring
- **Street View imagery**: Download images from four cardinal directions

Output is exported to YAML format with comprehensive metadata.

## Prerequisites

### 1. Install Dependencies

```bash
# Install mise and uv
mise install

# Create virtual environment and install packages
uv venv
uv pip install -e ".[dev]"
```

### 2. Google Earth Engine Authentication

Authenticate with Google Earth Engine:

```bash
uv run earthengine authenticate
```

Follow the browser authentication flow and select your GCP project when prompted.

### 3. Google Maps API Key (Optional)

For geocoding, business lookup, and Street View features, create a Google Maps API key:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable these APIs:
   - Geocoding API
   - Places API
   - Street View Static API
3. Create an API key with appropriate restrictions
4. Set the environment variable:

```bash
export GOOGLE_MAPS_API_KEY="your_api_key_here"
```

Or create a `.env` file in the project root:

```bash
GOOGLE_MAPS_API_KEY=your_api_key_here
```

## Quick Start

### 1. Prepare a Region File

Create a GeoJSON file defining your area of interest. Supported formats:

- **FeatureCollection**: Uses first feature's geometry
- **Feature**: Uses the feature's geometry
- **Geometry**: Uses raw geometry directly

Example `my_region.geojson`:

```json
{
  "type": "Feature",
  "properties": {"name": "My Region"},
  "geometry": {
    "type": "Polygon",
    "coordinates": [[
      [-122.5, 37.7],
      [-122.5, 37.9],
      [-122.3, 37.9],
      [-122.3, 37.7],
      [-122.5, 37.7]
    ]]
  }
}
```

### 2. Run Basic Query

Query VIIRS DNB for top 20 emitters (default):

```bash
uv run viirs top-polluters \
  my_region.geojson \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

Output: `results.yml` with ranked emitters by average radiance.

### 3. Run with Enrichment

Add geocoding and business lookup:

```bash
uv run viirs top-polluters \
  my_region.geojson \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --geocode \
  --businesses \
  --output enriched_results.yml
```

## Configuration

### Environment Variables

Configure via `.env` file or environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_MAPS_API_KEY` | Google Maps API key for enrichment | None (optional) |
| `EE_PROJECT_ID` | Google Earth Engine project ID | None (optional) |
| `CACHE_DIR` | Directory for caching API responses | `cache` |
| `VIIRS_SCALE_M` | VIIRS sampling scale in meters | `500` |
| `BUSINESS_SEARCH_RADIUS_M` | Business search radius in meters | `100` |

Example `.env`:

```bash
GOOGLE_MAPS_API_KEY=AIza...
CACHE_DIR=.cache
VIIRS_SCALE_M=500
BUSINESS_SEARCH_RADIUS_M=100
```

### CLI Flags

```
REGION                 Path to GeoJSON file (required, positional argument)
--start-date TEXT      Start date in YYYY-MM-DD format (required)
--end-date TEXT        End date in YYYY-MM-DD format (required)
-n, --n INTEGER        Number of top emitters (1-30, default: 20)
--geocode/--no-geocode Enable/disable reverse geocoding (default: enabled)
--businesses           Enable business lookup
--streetview           Enable Street View imagery download
--output PATH          Output YAML file path (default: results.yml)
--clear-cache          Clear all caches before running
```

## Common Tasks

### Query Top 10 Emitters

```bash
uv run viirs top-polluters \
  san_francisco.geojson \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --n 10 \
  --no-geocode
```

### Full Enrichment with Street View

```bash
uv run viirs top-polluters \
  downtown.geojson \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --n 20 \
  --geocode \
  --businesses \
  --streetview \
  --output full_analysis.yml
```

### Clear Cache and Re-Query

```bash
uv run viirs top-polluters \
  my_region.geojson \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --clear-cache
```

### Query Specific Date Range

```bash
# Summer months only
uv run viirs top-polluters \
  city.geojson \
  --start-date 2024-06-01 \
  --end-date 2024-08-31
```

## Output Format

The tool generates YAML files with this structure:

```yaml
metadata:
  exported_at: "2025-10-18T12:00:00.000000+00:00"
  total_emitters: 20
  source: "NOAA VIIRS DNB Monthly"

emitters:
  - rank: 1
    coordinates:
      lat: 37.8042
      lon: -122.4101
    avg_radiance: 52.1
    address:
      formatted: "123 Main St, San Francisco, CA 94102, USA"
      street: "Main St"
      city: "San Francisco"
      state: "California"
      country: "United States"
      postal_code: "94102"
      place_id: "ChIJ..."
      quality_score: 0.95
      match_type: "ROOFTOP"
    businesses:
      - name: "AT&T Park"
        place_id: "ChIJ..."
        types: ["stadium", "point_of_interest"]
        vicinity: "24 Willie Mays Plaza"
        distance_m: 25.3
        rating: 4.5
        relevance_score: 0.92
    streetview:
      available: true
      north: "cache/streetview/1/north.jpg"
      south: "cache/streetview/1/south.jpg"
      east: "cache/streetview/1/east.jpg"
      west: "cache/streetview/1/west.jpg"
```

## Troubleshooting

### Authentication Errors

**Error**: `ee.ee_exception.EEException: Please authorize access to your Earth Engine account`

**Solution**:
```bash
uv run earthengine authenticate
```

### No Emitters Found

**Problem**: Query returns zero emitters

**Common causes**:
1. **Region too small**: VIIRS data is 500m resolution - use larger regions
2. **Date range invalid**: Check start/end dates are correct
3. **Region outside coverage**: VIIRS DNB covers global landmass

**Solution**: Try a larger region or different date range.

### Google Maps API Errors

**Error**: `Warning: Geocoding failed: ...`

**Common causes**:
1. Missing or invalid API key
2. API not enabled in Google Cloud Console
3. Quota exceeded
4. Network connectivity issues

**Solution**:
1. Check `GOOGLE_MAPS_API_KEY` is set correctly
2. Verify APIs are enabled in GCP Console
3. Check quota limits in GCP Console
4. Run with `--skip-geocode` to bypass geocoding

### Rate Limiting

**Problem**: Slow API responses or quota warnings

**Solution**: The tool includes automatic rate limiting (50ms delay for geocoding/places, 100ms for Street View). For large queries:

1. Use smaller regions
2. Reduce `-n` parameter
3. Run overnight for quota reset
4. Enable only required enrichment features

### Cache Issues

**Problem**: Stale or incorrect cached data

**Solution**:
```bash
# Clear all caches
uv run viirs top-polluters \
  my_region.geojson \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --clear-cache

# Or manually delete cache directory
rm -rf cache/
```

## Advanced Usage

### Custom Sampling Scale

Change VIIRS sampling resolution:

```bash
export VIIRS_SCALE_M=1000  # 1km resolution
```

### Larger Business Search Radius

Increase radius for business lookup:

```bash
export BUSINESS_SEARCH_RADIUS_M=200  # 200m radius
```

### Processing Multiple Regions

Use a shell script to process multiple regions:

```bash
#!/bin/bash
for region in regions/*.geojson; do
  name=$(basename "$region" .geojson)
  uv run viirs top-polluters \
    "$region" \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --output "results_${name}.yml"
done
```

### Programmatic Usage

Import and use the Python API directly:

```python
from datetime import date
from pathlib import Path
from sevenrad_ee.top_polluters.earth_engine import get_top_emitters
from sevenrad_ee.top_polluters.enrichment import geocode_coordinates
from sevenrad_ee.top_polluters.export import export_yaml

# Query VIIRS DNB
region_path = Path("my_region.geojson")
emitters = get_top_emitters(
    region_path=region_path,
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
    n=20,
)

# Enrich with geocoding
for emitter in emitters:
    emitter.address = geocode_coordinates(emitter.coordinates)

# Export to YAML
export_yaml(emitters, Path("results.yml"))
```

## Understanding the Data

### Quality Scoring

**Address Quality** (0.0 to 1.0):
- 1.0 = Rooftop-level precision with complete address
- 0.9 = Range-interpolated address
- 0.75 = Geometric center of area
- 0.5 = Approximate location

**Business Relevance** (calculated from):
- Proximity (exponential decay: closer = higher score)
- Business type (stadiums, malls, etc. get 1.5x multiplier)
- User rating (5-star = 1.2x, 1-star = 0.6x)

### Radiance Values

VIIRS DNB measures nighttime light in nanoWatts/cm²/sr. Typical values:
- **<5**: Rural/dark areas
- **5-20**: Residential neighborhoods
- **20-50**: Commercial districts
- **>50**: Stadiums, industrial facilities, major infrastructure

### Caching Behavior

The tool caches:
- **Earth Engine queries**: By region + date range + N
- **Geocoding results**: By coordinates
- **Business searches**: By coordinates + radius
- **Street View availability**: By coordinates

Cache keys use SHA256 hashing for deterministic lookups.

## Related Documentation

- [Earth Engine VIIRS DNB Dataset](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG)
- [Google Geocoding API](https://developers.google.com/maps/documentation/geocoding)
- [Google Places API](https://developers.google.com/maps/documentation/places/web-service)
- [Google Street View Static API](https://developers.google.com/maps/documentation/streetview)

## Support

For issues or questions:
- GitHub Issues: https://github.com/abossenbroek/sevenrad-ee/issues
- Project Documentation: See root README.md
