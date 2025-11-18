# How to Use AI-Powered Greenhouse Attribution

**Documentation Type: How-to Guide**

This guide explains how to use the AI-powered greenhouse attribution feature to identify specific greenhouse operations responsible for VIIRS nighttime light pollution.

## What This Feature Does

The greenhouse attribution feature combines:
1. **Geospatial Business Search** - Find businesses near light pollution pixels using Google Places API
2. **AI Analysis** - Use Perplexity AI to research greenhouse lighting practices
3. **Confidence Scoring** - Multi-factor algorithm to rank attribution likelihood

This is useful for identifying which specific greenhouse operations are contributing to measured nighttime light pollution from satellite data.

## Prerequisites

### Required Setup

1. **Google Maps API Key** - For business search
   ```bash
   export GOOGLE_MAPS_API_KEY="your_google_maps_api_key"
   ```

2. **Perplexity API Key** - For AI-powered analysis
   ```bash
   export PERPLEXITY_API_KEY="your_perplexity_api_key"
   ```

   Get your key from: https://www.perplexity.ai/settings/api

3. **Earth Engine Authentication**
   ```bash
   uv run earthengine authenticate
   ```

4. **Region GeoJSON File** - Defines the geographic area to analyze

### Environment Variables

Add these to your `.env` file (copy from `.env.example`):

```bash
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
EE_PROJECT_ID=your_ee_project_id_here
```

## Quick Start

### 1. Basic Attribution Analysis

Run attribution analysis on top light emitters in a region:

```bash
uv run viirs top-polluters moerkapelle.geojson \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --n 5 \
  --businesses \
  --attribute-greenhouses \
  --output moerkapelle_with_attribution.yml
```

This will:
- Find the top 5 light emitters in the region
- Search for nearby businesses around each pixel
- Use AI to analyze which businesses are likely greenhouses with night lighting
- Output results with confidence scores

### 2. Conservative Mode (Minimize API Usage)

To conserve Perplexity API quota:

```bash
uv run viirs top-polluters region.geojson \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --n 3 \
  --attribute-greenhouses \
  --output results.yml
```

The attribution feature automatically limits analysis to:
- **Top 5 emitters maximum** (conserves API quota)
- **3 businesses per pixel** (reduces redundant queries)
- **85% confidence threshold** (stops when high-confidence match found)

## Understanding the Output

### Console Output

During execution, you'll see:

```
════════════════════════════════════════════════════════
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Greenhouse Attribution Scanner                      ┃
┃ AI-Powered Light Pollution Source Identification    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
Pixel Center: 52.0360°N, 4.5983°E
Radiance: 9841.77 nW/cm²/sr
════════════════════════════════════════════════════════

Phase 1: Searching nearby businesses (1km radius)...
✓ Found 12 businesses

Analyzing: Westlandse Groenteteelt BV (235m)
  ✓ Analysis complete
  Confidence: 0.92
  Night lighting: CONFIRMED
  Crops: tomato, pepper

✓ HIGH CONFIDENCE MATCH FOUND! Stopping scan.
```

### Attribution Results Table

```
ATTRIBUTION RESULTS

Rank  Business                     Distance  Confidence  Status
─────────────────────────────────────────────────────────────────
1     Westlandse Groenteteelt BV   235m      0.92        ✓ Confirmed
2     Tuinbouw Moerkapelle         450m      0.78        ✓ Confirmed
3     Agribusiness Solutions       680m      0.45        ? Unknown
```

### Confidence Score Breakdown

Confidence scores range from 0.0 to 1.0 and are calculated from:

- **Distance Score (max 0.30)** - Closer businesses score higher
- **Business Type Score (max 0.20)** - Greenhouses/nurseries score higher
- **Night Lighting (base 0.50)** - Confirmed night lighting (veto if absent)
- **Multipliers** (applied to base):
  - Energy-intensive crops (+15%): tomato, pepper, cucumber, flower
  - Large operation (+10%): >5 hectares
  - Year-round operations (+5%)

**Veto Conditions** (result in 0.0 confidence):
- Perplexity analysis failed
- Confirmed NO night lighting
- No lighting data available

**Interpretation**:
- **≥0.85**: High confidence - likely culprit
- **0.70-0.84**: Medium-high confidence - probable contributor
- **0.50-0.69**: Medium confidence - possible contributor
- **<0.50**: Low confidence - unlikely or insufficient data

## Common Tasks

### Task 1: Analyze Specific High-Radiance Pixels

Focus on only the brightest emitters:

```bash
uv run viirs top-polluters region.geojson \
  --start-date 2024-06-01 \
  --end-date 2024-08-31 \
  --n 3 \
  --attribute-greenhouses
```

### Task 2: Combine with Other Enrichment Features

Get comprehensive data including addresses, businesses, and attribution:

```bash
uv run viirs top-polluters region.geojson \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --n 10 \
  --geocode \
  --businesses \
  --attribute-greenhouses \
  --output comprehensive_analysis.yml
```

### Task 3: Clear Cache Before Re-Running

Attribution results are cached to save API quota. To force fresh analysis:

```bash
uv run viirs top-polluters region.geojson \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --clear-cache \
  --attribute-greenhouses
```

## Troubleshooting

### Error: "PERPLEXITY_API_KEY not set"

**Cause**: Missing or incorrect Perplexity API key

**Solution**:
```bash
# Set environment variable
export PERPLEXITY_API_KEY="your_key_here"

# Or add to .env file
echo "PERPLEXITY_API_KEY=your_key_here" >> .env
```

### Error: "No businesses found within 1km"

**Cause**: Pixel is in remote area with no registered businesses nearby

**Solution**:
- This is expected for rural or industrial areas
- Attribution cannot proceed without nearby businesses
- Consider using Street View imagery instead (`--streetview`)

### Error: "Rate limit hit"

**Cause**: Exceeded Perplexity API rate limits

**Solution**:
- The tool automatically retries with exponential backoff
- Reduce `--n` parameter to analyze fewer emitters
- Wait before retrying (rate limits usually reset within minutes)

### Warning: "Analysis failed for business X"

**Cause**: Perplexity couldn't find reliable data about the business

**Solution**:
- This is normal for small operations without web presence
- The tool continues analyzing other businesses
- Results will show lower confidence scores

### Low Confidence Scores (<0.50)

**Possible Causes**:
- Business doesn't use night lighting
- Insufficient public data available
- Business is far from pixel center (>500m)

**What to Do**:
- Check other nearby businesses in the results
- Review Street View imagery if available
- Consider manual verification for high-radiance pixels

## Advanced Usage

### Understanding the Algorithm

The attribution system uses a three-phase approach:

**Phase 1: Geospatial Search**
- Single 1km radius search around pixel center
- Finds all businesses via Google Places API
- Sorts by distance (closest first)

**Phase 2: AI Analysis**
- Queries Perplexity AI for each business
- Structured JSON prompt asks for:
  - Night lighting usage (boolean)
  - Primary crops (list)
  - Operation size (hectares)
  - Year-round operations (boolean)
  - Verifiable sources
- Robust JSON parsing with retry logic

**Phase 3: Confidence Scoring**
- Continuous distance decay (not step functions)
- Night lighting treated as veto condition
- Business type and operation characteristics as multipliers
- Returns ranked list of likely contributors

### API Quota Management

The feature is designed to minimize API costs:

1. **Cached Results** - Previous analyses are cached by business ID
2. **Early Stopping** - Stops at 85% confidence threshold
3. **Limited Scope** - Maximum 5 emitters × 3 businesses = 15 API calls
4. **Single-Call Search** - One Google Places call per pixel (not incremental radii)

**Estimated Costs** (per run):
- Google Maps Places: ~$0.032 per search × 5 pixels = $0.16
- Perplexity API: ~$0.005 per query × 15 businesses = $0.075
- **Total**: ~$0.24 per analysis run

### Limitations and Caveats

**Known Limitations**:
1. **Hallucination Risk** - AI may provide inaccurate data for businesses without web presence
2. **Source Verification Required** - Always check `sources` field in results
3. **Small Operations** - May not have detectable online footprint
4. **Temporal Accuracy** - AI provides current data, not historical
5. **1km Search Limit** - Businesses beyond 1km are not considered

**Best Practices**:
- Use for initial attribution, verify manually
- Check Perplexity sources for credibility
- Combine with Street View imagery (`--streetview`)
- Review all results, not just highest confidence
- Document attribution rationale in reports

## Related Documentation

- **Tutorial**: "Getting Started with VIIRS Top Polluters" (for beginners)
- **Reference**: API documentation for attribution module (src/sevenrad_ee/top_polluters/attribution.py)
- **Explanation**: "Why We Use ~= Version Constraints" (CLAUDE.md)
- **How-to**: "How to Process VIIRS Nighttime Light Data" (for general VIIRS usage)

## Technical Details

For developers implementing or extending this feature:

- **Module**: `src/sevenrad_ee/top_polluters/attribution.py`
- **Models**: Pydantic models in `src/sevenrad_ee/top_polluters/models.py`
- **Configuration**: Settings in `src/sevenrad_ee/top_polluters/config.py`
- **CLI Integration**: `src/sevenrad_ee/top_polluters/cli.py`

## Support

For issues or questions:
- Check troubleshooting section above
- Review source code comments in attribution.py
- Report bugs at: https://github.com/abossenbroek/sevenrad-ee/issues
