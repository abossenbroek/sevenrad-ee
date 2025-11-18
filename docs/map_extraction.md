# How to Extract VIIRS Maps

**Documentation Type:** How-to Guide (Task-oriented)

This guide shows you how to generate VIIRS nighttime light composite maps for specific geographic regions using the map extraction CLI.

## Prerequisites

- Earth Engine authentication configured (`uv run earthengine authenticate`)
- JavaScript configuration file with region definitions (e.g., `extract_geotiffs.js`)
- Color palette file (e.g., `wavelength_palette_1025.js`)

## Quick Start

### 1. List Available Map Regions

```bash
uv run python -m sevenrad_ee.operations.generate_viirs_maps --list-maps
```

This displays all available regions with descriptions and their availability status in your config file.

### 2. Authenticate with Google Earth Engine

```bash
uv run earthengine authenticate
```

Follow the prompts to authenticate with your Google account.

### 3. Generate All Maps for a Year

```bash
uv run python -m sevenrad_ee.operations.generate_viirs_maps \
    --start-date 2020-01-01 \
    --end-date 2020-12-31 \
    --maps all
```

This will process all regions defined in your `extract_geotiffs.js` configuration file.

### 4. Generate Specific Maps

Use the region codes from `--list-maps`:

```bash
uv run python -m sevenrad_ee.operations.generate_viirs_maps \
    --start-date 2020-01-01 \
    --end-date 2020-12-31 \
    --maps us,europe,china
```

This processes only the US, Europe, and China regions.

## Configuration

### JavaScript Configuration File

The CLI reads region definitions from a JavaScript configuration file. By default, it looks for `extract_geotiffs.js` in the current directory.

**Required Format:**

```javascript
// Define regions as ee.Geometry.Rectangle
var region_name = ee.Geometry.Rectangle([
  west,    // west longitude
  south,   // south latitude
  east,    // east longitude
  north    // north latitude
]);

// Define visualization parameters
var nighttimeLogVis = {
  min: 0,
  max: 4.27,
  palette: wavelengthPalette
};

// Define color palette
var wavelengthPalette = [
  '000000', 'hex_color', ...
];
```

**Example Regions:**

```javascript
// United States
var us = ee.Geometry.Rectangle([
  -90.5,    // west (Moline, Illinois)
  26.65,    // south
  -66.9,    // east
  45.5      // north (Montreal)
]);

// Europe
var europe = ee.Geometry.Rectangle([
  -11.6,    // west
  42.0,     // south
  18.8,     // east
  60.85     // north
]);

// Netherlands - Full coverage
var netherlands_full = ee.Geometry.Rectangle([
  -10.5,    // west (extended to Ireland)
  50.75,    // south (Limburg)
  7.2,      // east (German border)
  53.5      // north (Wadden Islands)
]);
```

### Palette File

The color palette can be defined in a separate file (e.g., `wavelength_palette_1025.js`):

```javascript
wavelengthPalette = [
  '000000', '000001', '000002', // ... 1025 colors
];
```

The parser will automatically load this file if it exists in the same directory as the main config.

## Command-Line Options

### Required Arguments

- `--start-date YYYY-MM-DD` - Start date for data collection
- `--end-date YYYY-MM-DD` - End date for data collection
- `--maps MAPS` - Maps to generate (see below)

### Optional Arguments

- `--list-maps` - Show available map regions with descriptions and exit
- `--config PATH` - Path to JavaScript config file (default: `./extract_geotiffs.js`)
- `--output-dir PATH` - Output directory for results (future use)

### Maps Selection

The `--maps` argument accepts:

1. **All maps:** `--maps all`
   - Processes every region defined in the config file

2. **Specific maps:** `--maps region1,region2,region3`
   - Comma-separated list of region codes
   - Use `--list-maps` to see all available regions
   - Examples: `--maps drc,us` or `--maps netherlands_moerkappele`

3. **Single map:** `--maps netherlands_full`
   - Process just one region

**Known Region Codes:**
- `drc` - Democratic Republic of Congo
- `us` - United States
- `china` - China
- `europe` - Europe
- `netherlands_full` - Netherlands (Full - Ireland to Germany)
- `netherlands_regional` - Netherlands (Regional)
- `netherlands_westland_moerkappele` - Netherlands (Westland + Moerkappele)
- `netherlands_moerkappele` - Netherlands (Moerkappele only)

## Common Tasks

### Process All Netherlands Regions

```bash
uv run python -m sevenrad_ee.operations.generate_viirs_maps \
    --start-date 2020-01-01 \
    --end-date 2020-12-31 \
    --maps netherlands_full,netherlands_regional,netherlands_westland_moerkappele,netherlands_moerkappele
```

### Process a Specific Time Period

```bash
# Summer months only
uv run python -m sevenrad_ee.operations.generate_viirs_maps \
    --start-date 2020-06-01 \
    --end-date 2020-08-31 \
    --maps europe
```

### Use a Custom Configuration File

```bash
uv run python -m sevenrad_ee.operations.generate_viirs_maps \
    --start-date 2020-01-01 \
    --end-date 2020-12-31 \
    --maps all \
    --config /path/to/my_custom_regions.js
```

### List Available Maps

View all available map regions with descriptions:

```bash
uv run python -m sevenrad_ee.operations.generate_viirs_maps --list-maps
```

This displays a formatted table showing:
- Region codes (to use with --maps)
- Full descriptions
- Availability status in your config file

Example output:
```
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Region Code          ┃ Description                 ┃ Status      ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ drc                  │ Democratic Republic of Congo│ ✓ Available │
│ us                   │ United States               │ ✓ Available │
│ netherlands_moerkappele│ Netherlands (Moerkappele only)│ ✓ Available │
└──────────────────────┴─────────────────────────────┴─────────────┘
```

## Understanding the Output

### Console Output

The CLI provides colorful, formatted output:

1. **Header:** Banner with tool name
2. **Summary Table:** Shows processing parameters
   - Start/End dates
   - Maps to process
   - Config file location
3. **Progress Indicators:** For each map:
   - Creating region geometry
   - Loading VIIRS DNB collection
   - Filtering by region
   - Creating composite image
4. **Success Message:** Confirms completion
5. **Visualization Info:** Min/max values and palette size

### Example Output

```
╭───────────────────────────────────────────────────╮
│ VIIRS Nighttime Light Map Generator               │
│ Generate composite maps using Google Earth Engine │
╰───────────────────────────────────────────────────╯

┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Parameter       ┃ Value                     ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Start Date      │ 2020-01-01                │
│ End Date        │ 2020-12-31                │
│ Maps            │ us, europe, china         │
│ Config File     │ ./extract_geotiffs.js     │
│ Output Directory│ Not saving                │
└─────────────────┴───────────────────────────┘

Loading configuration from extract_geotiffs.js...
✓ Found 8 region(s) in config
Initializing Earth Engine...
✓ Earth Engine initialized successfully

Processing 3 map(s)...

Processing 'us' from 2020-01-01 to 2020-12-31
⠋ Creating region geometry...
⠋ Loading VIIRS DNB collection...
⠋ Filtering by region...
⠋ Creating composite image...
✓ Composite for 'us' generated

Success! Generated 3 VIIRS composite(s).
```

## Data Details

### VIIRS Dataset

- **Collection:** `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`
- **Band:** `avg_rad` (average radiance)
- **Temporal Resolution:** Monthly composites
- **Processing:** Median composite across time range

### Spatial Resolution

- **Scale:** 500 meters per pixel (from JavaScript config)
- **CRS:** EPSG:4326 (WGS84 geographic coordinates)

### Visualization

- **Transform:** Logarithmic (log10) scale
- **Min Value:** 0 (log10(1) for original value 0)
- **Max Value:** 4.27 (log10(18581) for original value 18580)
- **Palette:** 1025-color wavelength-based spectrum

## Troubleshooting

### "Config file not found"

**Problem:** The default `extract_geotiffs.js` is not in the current directory.

**Solution:** Specify the config file location:
```bash
--config /full/path/to/extract_geotiffs.js
```

### "No regions found in config file"

**Problem:** The parser couldn't extract region definitions.

**Solution:** Verify your JavaScript config file uses this exact format:
```javascript
var region_name = ee.Geometry.Rectangle([
  west,    // comment
  south,   // comment
  east,    // comment
  north    // comment
]);
```

### "Invalid map name(s)"

**Problem:** Requested map name doesn't exist in the config.

**Solution:** Check available maps by looking at the error message or reviewing your JavaScript config. Map names must exactly match JavaScript variable names (case-sensitive).

### Earth Engine Authentication Errors

**Problem:** `Failed to initialize Earth Engine`

**Solution:**
1. Run: `uv run earthengine authenticate`
2. Follow authentication prompts
3. Retry the map extraction

## Advanced Usage

### Creating Custom Region Configurations

1. **Copy the template:**
   ```javascript
   var my_region = ee.Geometry.Rectangle([
     west_longitude,
     south_latitude,
     east_longitude,
     north_latitude
   ]);
   ```

2. **Add to your config file**

3. **Process the new region:**
   ```bash
   uv run python -m sevenrad_ee.operations.generate_viirs_maps \
       --start-date 2020-01-01 \
       --end-date 2020-12-31 \
       --maps my_region
   ```

### Processing Multiple Years

Process each year separately for better memory management:

```bash
for year in 2018 2019 2020 2021; do
  uv run python -m sevenrad_ee.operations.generate_viirs_maps \
      --start-date ${year}-01-01 \
      --end-date ${year}-12-31 \
      --maps all
done
```

### Using Different VIIRS Products

The current implementation uses `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`. To use different products, you'll need to modify `src/sevenrad_ee/operations/generate_viirs_maps.py:116`:

```python
viirs_collection = ee.ImageCollection(
    "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"  # Change this
).filterDate(start_date, end_date)
```

## Related Documentation

- [Earth Engine VIIRS Documentation](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG)
- [VIIRS DNB Product Guide](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/viirs/)
- Project CLAUDE.md for development guidelines

## Notes

- Processing time depends on region size and time range
- Earth Engine quotas apply to all processing
- Composite generation happens server-side (no local download yet)
- Export functionality is planned for future releases
