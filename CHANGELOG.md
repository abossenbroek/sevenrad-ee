# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Instagram Discovery**: Perplexity AI now identifies Instagram accounts for greenhouse businesses
  - Added `instagram_handle` field to `PerplexityAnalysis` model
  - Updated AI prompt to request Instagram account discovery
  - Display Instagram handle in attribution summary output
- **Street View Links**: Automatic Google Street View URL generation for all businesses
  - Added `streetview_link` field to Business model
  - Links use `map_action=pano` format for direct Street View access
  - Included in YAML export for easy access
- **Configurable Image Directory**: New `--images-dir` CLI option
  - Customize location for Street View image downloads
  - Default: `images/` (was hardcoded to `cache/streetview/`)
  - Updated `get_street_view_images()` to accept custom base directory

### Security
- **Prompt Injection Protection**: Sanitized user input in Perplexity prompts
  - Added `_sanitize_location_context()` function
  - Removes special characters that could manipulate AI prompts
  - Limits input length to 100 characters
  - Prevents malicious instructions via GeoJSON filenames

### Improved
- **Code Quality Enhancements**:
  - Created `geospatial.py` module for shared distance calculations
  - Eliminated code duplication (haversine function was in 2 places)
  - Replaced magic numbers with named constants in confidence scoring
  - Added comprehensive unit tests for attribution logic (12 tests)
  - Clarified confidence score clamping behavior with documentation

### Fixed
- **CRITICAL BUG**: Fixed VIIRS resolution from 500m to 750m (correct sensor specification)
  - Reference: https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VJ102DNB
  - Added spatial filtering to ensure minimum 750m separation between emitters
  - Prevents returning adjacent/overlapping pixels as distinct polluters
  - New `NotEnoughEmittersError` exception when insufficient spatially distinct emitters found
  - Provides clear guidance: expand region or reduce `-n` parameter

### Changed
- **BREAKING**: Migrated from Typer to Click for CLI framework
  - Unified CLI entry point: `viirs` command with subcommands
  - `viirs-top-polluters` → `viirs top-polluters REGION` (region is now positional argument)
  - `--skip-geocode` → `--no-geocode` (Click's boolean flag pattern)
  - All subcommands now accessible via `viirs --help`
- Updated all documentation to reflect new CLI structure
- Removed Typer dependency, added Click ~=8.1.0
- Removed obsolete script entry `viirs-top-polluters`

### Fixed
- Resolved Typer subcommand parameter collision issues
- Fixed `app()` reference in top_polluters CLI (now `top_polluters()`)
- Cleaned up unused noqa directives

### Technical Details
- Click version: 8.1.0
- Unified CLI pattern using `@click.group()`
- All tests updated and passing (52 tests, 76% coverage)
- Code quality checks passing (ruff, mypy, pytest)

### Migration Guide

If you were using the old CLI, update your commands as follows:

**Old syntax:**
```bash
uv run viirs-top-polluters \
  --region my_region.geojson \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --skip-geocode
```

**New syntax:**
```bash
uv run viirs top-polluters \
  my_region.geojson \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --no-geocode
```

**Key changes:**
1. Command is now `viirs top-polluters` (with space)
2. Region is a positional argument (no `--region` flag)
3. `--skip-geocode` is now `--no-geocode`
4. All other options remain the same

## [0.1.0] - 2025-10-18

### Added
- Initial project setup with Python 3.12+
- VIIRS Top Polluters feature for identifying nighttime light emitters
- VIIRS Maps generation for composite nighttime light maps
- Google Earth Engine integration
- Google Maps API enrichment (geocoding, businesses, Street View)
- Comprehensive test suite with VCR.py cassettes
- Pre-commit hooks for code quality
- GitHub Actions CI/CD workflows
- Documentation following Diátaxis framework
