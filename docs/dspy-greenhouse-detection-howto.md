# How to Detect Greenhouse Artificial Lighting Using DSPy

**Documentation Type: How-to Guide**

This guide shows you how to use the DSPy-based greenhouse detection system to determine if a company is a greenhouse using artificial lighting, with structured Pydantic-parseable output.

## What You'll Learn

- How to configure DSPy with Perplexity for greenhouse detection
- How to run structured greenhouse analysis with Dutch source priorities
- How to parse and use the structured Pydantic output
- How to interpret confidence scores and evidence

## Prerequisites

1. **Perplexity API Key**: Sign up at [https://www.perplexity.ai/](https://www.perplexity.ai/) and get an API key
2. **Python 3.12+**: Project requires Python 3.12 or higher
3. **Dependencies installed**: Run `uv pip install -e ".[dev]"`
4. **Environment setup**: Add your API key to `.env` file:
   ```bash
   PERPLEXITY_API_KEY=pplx-your-api-key-here
   ```

## Quick Start

### 1. Configure DSPy with Perplexity

```python
import os
import dspy

# Load API key from environment
api_key = os.getenv("PERPLEXITY_API_KEY")

# Configure Perplexity as the language model
lm = dspy.LM("perplexity/sonar", api_key=api_key)
dspy.configure(lm=lm)
```

### 2. Analyze a Greenhouse

```python
from sevenrad_ee.ai.dspy_greenhouse import analyze_greenhouse

# Analyze a known greenhouse
result = analyze_greenhouse(
    company_name="Royal Van Zanten",
    location="Rijsenhout, Nederland",
    additional_context="Gerbera and rose grower"
)

# Access structured results
print(f"Uses lighting: {result.uses_artificial_lighting}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Lighting type: {result.lighting_type}")
print(f"Crops: {result.primary_crops}")
print(f"Sources: {result.sources}")
```

### 3. Understand the Output

The `analyze_greenhouse()` function returns a `GreenhouseLightingAnalysis` Pydantic model with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `uses_artificial_lighting` | bool | Whether greenhouse uses artificial lighting |
| `confidence` | float | Confidence score (0.0-1.0) |
| `lighting_type` | str\|None | Type of lighting (LED, SON-T, HPS, etc.) |
| `evidence` | list[str] | Evidence excerpts from sources |
| `primary_crops` | list[str] | Main crops grown |
| `size_hectares` | float\|None | Greenhouse size in hectares |
| `sources` | list[str] | Source URLs (prioritizes .nl domains) |
| `reasoning` | str\|None | Explanation of classification |

## Dutch Source Prioritization

The system automatically enhances queries to prioritize Dutch and German sources:

### Priority Sources
- royalvanzanten.com
- floraldaily.com
- vakbladvoordebloemisterij.nl
- kasmagazine.nl
- onderglas.nl

### Dutch Terminology Used
- assimilatieverlichting (assimilation lighting)
- groeilicht (grow light)
- kunstlicht (artificial light)
- kas (greenhouse)
- teelt (cultivation)
- LED-verlichting (LED lighting)
- SON-T (high-pressure sodium)
- lichtspectrum (light spectrum)

## Common Use Cases

### Use Case 1: Batch Analysis of Multiple Companies

```python
import dspy
from sevenrad_ee.ai.dspy_greenhouse import analyze_greenhouse

# Configure once
lm = dspy.LM("perplexity/sonar", api_key="your-key")
dspy.configure(lm=lm)

# Analyze multiple companies
companies = [
    {"name": "Royal Van Zanten", "location": "Rijsenhout, NL"},
    {"name": "Duijvestijn Tomaten", "location": "Pijnacker, NL"},
    {"name": "Prominent", "location": "Poeldijk, NL"},
]

results = []
for company in companies:
    result = analyze_greenhouse(
        company_name=company["name"],
        location=company["location"]
    )
    results.append(result)

# Filter for high-confidence positives
high_confidence = [
    r for r in results
    if r.uses_artificial_lighting and r.confidence > 0.7
]

print(f"Found {len(high_confidence)} greenhouses with artificial lighting")
```

### Use Case 2: Integration with Existing Attribution Pipeline

```python
from sevenrad_ee.ai.dspy_greenhouse import GreenhouseDetector
import dspy

# Initialize detector once
lm = dspy.LM("perplexity/sonar", api_key="your-key")
dspy.configure(lm=lm)

detector = GreenhouseDetector()

# Run analysis
prediction = detector(
    company_name="Your Company",
    location="Your Location",
    additional_context="Optional context"
)

# Convert to Pydantic for validation
pydantic_result = detector.to_pydantic(prediction)

# Integrate with existing confidence scoring
if pydantic_result.uses_artificial_lighting:
    combined_confidence = calculate_combined_score(
        pydantic_result.confidence,
        distance_score,
        other_factors
    )
```

### Use Case 3: Export Results to JSON/CSV

```python
import json
from sevenrad_ee.ai.dspy_greenhouse import analyze_greenhouse

result = analyze_greenhouse(
    company_name="Royal Van Zanten",
    location="Rijsenhout, Nederland"
)

# Export as JSON (Pydantic models have .model_dump())
json_output = result.model_dump_json(indent=2)
print(json_output)

# Save to file
with open("greenhouse_analysis.json", "w") as f:
    f.write(json_output)

# Or as dict for CSV/DataFrame
dict_output = result.model_dump()
# Use dict_output with pandas, csv module, etc.
```

## Configuration Options

### Perplexity Model Selection

Different Perplexity models have different trade-offs:

```python
# Fast, cheaper (recommended for batch processing)
lm = dspy.LM("perplexity/sonar", api_key=api_key)

# More accurate, slower (recommended for high-value queries)
lm = dspy.LM("perplexity/sonar-pro", api_key=api_key)

# Legacy model name (if others don't work)
lm = dspy.LM("perplexity/llama-3.1-sonar-small-128k-online", api_key=api_key)
```

### Custom Context Enhancement

Add domain-specific context to improve accuracy:

```python
result = analyze_greenhouse(
    company_name="Your Company",
    location="Your Location",
    additional_context="""
        This is a large-scale operation growing tomatoes year-round.
        Known for using LED supplemental lighting.
        Part of a cooperative with multiple locations.
    """
)
```

## Troubleshooting

### Issue: "Access denied" API error

**Problem**: Perplexity API returns 403 or access denied

**Solutions**:
1. Verify API key is correct: `echo $PERPLEXITY_API_KEY`
2. Check API key has not expired
3. Verify you have API credits remaining
4. Try different model name (see Configuration Options)

### Issue: Low confidence scores

**Problem**: All results have confidence < 0.5

**Solutions**:
1. Add more specific `additional_context`:
   ```python
   additional_context="Greenhouse growing roses with year-round production"
   ```
2. Verify company name spelling is correct
3. Try more specific location (include country)
4. Check if company has online presence (Perplexity needs sources)

### Issue: No Dutch sources in results

**Problem**: All sources are English or other languages

**Solutions**:
1. Verify company is actually Dutch/European
2. Add ".nl" to company name if it has a Dutch website
3. Use Dutch company names (e.g., "Koninklijke Van Zanten" instead of "Royal Van Zanten")
4. Check the evidence field - Dutch terms might still be detected

### Issue: ValidationError when creating results

**Problem**: Pydantic raises validation errors

**Solutions**:
1. Check that confidence is between 0.0 and 1.0
2. Verify size_hectares is non-negative
3. Ensure required fields are present
4. Example fix:
   ```python
   # Bad
   GreenhouseLightingAnalysis(confidence=1.5)  # ERROR

   # Good
   GreenhouseLightingAnalysis(
       uses_artificial_lighting=True,
       confidence=0.95
   )
   ```

## Advanced Usage

### Optimizing with DSPy Teleprompters

DSPy can optimize prompts using demonstrations:

```python
import dspy
from dspy.teleprompt import BootstrapFewShot
from sevenrad_ee.ai.dspy_greenhouse import GreenhouseDetector

# Create training examples (demonstrations)
trainset = [
    dspy.Example(
        company_name="Royal Van Zanten",
        location="Rijsenhout, Nederland",
        uses_artificial_lighting=True,
        confidence=0.95
    ).with_inputs("company_name", "location"),
    # Add more examples...
]

# Optimize the detector
detector = GreenhouseDetector()
teleprompter = BootstrapFewShot(metric=your_metric_function)
optimized_detector = teleprompter.compile(detector, trainset=trainset)

# Use optimized detector
prediction = optimized_detector(
    company_name="New Company",
    location="Location"
)
```

### Custom Validation Logic

Add custom validation on top of Pydantic:

```python
def validate_greenhouse_result(result: GreenhouseLightingAnalysis) -> bool:
    """Custom validation for greenhouse analysis results."""
    # Require high confidence for positive classification
    if result.uses_artificial_lighting and result.confidence < 0.7:
        return False

    # Require at least one Dutch source for NL companies
    if "nederland" in result.sources[0].lower():
        if not any(".nl" in s for s in result.sources):
            return False

    # Require evidence if classified as using lighting
    if result.uses_artificial_lighting and not result.evidence:
        return False

    return True

# Use in pipeline
result = analyze_greenhouse(...)
if validate_greenhouse_result(result):
    # Proceed with high-quality result
    process_greenhouse(result)
```

## Performance Tips

1. **Batch processing**: Configure LM once, reuse for multiple queries
2. **Caching**: DSPy supports caching - enable for repeated queries
3. **Async processing**: Use DSPy's async capabilities for parallel queries
4. **Rate limiting**: Respect Perplexity API rate limits (add delays if needed)

```python
import time

for company in companies:
    result = analyze_greenhouse(company["name"], company["location"])
    results.append(result)
    time.sleep(1)  # Avoid rate limits
```

## Related Documentation

- **Tutorial**: [Getting Started with DSPy Greenhouse Detection](./dspy-greenhouse-tutorial.md)
- **Reference**: [GreenhouseLightingAnalysis API](../src/sevenrad_ee/ai/dspy_greenhouse.py)
- **Explanation**: [Why DSPy for Structured LLM Output](./dspy-explanation.md)

## Running the Example Script

A complete example script is provided:

```bash
# Set API key
export PERPLEXITY_API_KEY='pplx-your-key-here'

# Run example
uv run python scripts/example_dspy_greenhouse_detection.py
```

The script demonstrates:
- Basic configuration
- Running analysis
- Parsing structured output
- Handling Dutch sources
- Error handling

## Summary

The DSPy greenhouse detector provides:
- ✅ **Structured output**: Pydantic-validated results
- ✅ **Dutch source priority**: Optimized for Netherlands/Germany
- ✅ **Domain terminology**: Industry-specific search terms
- ✅ **High accuracy**: Confidence scoring with evidence
- ✅ **Easy integration**: Simple API with comprehensive documentation

For questions or issues, see the troubleshooting section or check the test file for more examples.
