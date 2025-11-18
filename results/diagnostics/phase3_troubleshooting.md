# Phase 3: GEPA Optimization Troubleshooting Guide

**Date**: 2025-11-19
**Purpose**: Document errors encountered during Phase 3 preparation and their solutions
**Status**: 2 errors fixed, 1 blocker requiring decision

---

## Executive Summary

During Phase 3 GEPA optimization preparation, we encountered three critical errors:

1. **GEPA API Parameter Conflict** ✅ FIXED
2. **Dataset Field Name Mismatch** ✅ FIXED
3. **Perplexity API Incompatibility** ❌ BLOCKER (requires model selection decision)

This document provides detailed error analysis, fixes applied, and proposed solutions for the current blocker.

---

## Error 1: GEPA API Parameter Conflict

### Symptom

```
TypeError: GEPA.__init__() got an unexpected keyword argument 'max_bootstrapped_demos'
```

Later evolved to:

```
AssertionError: Exactly one of max_metric_calls, max_full_evals, auto must be set.
You set max_metric_calls=20, max_full_evals=None, auto=light.
```

### Root Cause

The GEPA optimizer requires **exactly one** of the following parameters:
- `auto` (preset configuration: 'light', 'medium', 'heavy')
- `max_metric_calls` (explicit evaluation budget)
- `max_full_evals` (explicit full evaluation budget)

Our scripts were providing **both** `auto` and `max_metric_calls`, causing a conflict.

### Affected Files

- `scripts/dry_run_cost_estimate.py` (line 249)
- `notebooks/optimize_with_cached_data.py` (lines 253-258)

### Fix Applied

**Before:**
```python
optimizer = GEPA(
    metric=gepa_compatible_metric,
    auto='medium',
    max_metric_calls=500,  # ❌ Conflicts with auto parameter
    reflection_lm=teacher_lm,
    seed=SEED,
)
```

**After:**
```python
optimizer = GEPA(
    metric=gepa_compatible_metric,
    auto='medium',  # ✅ Single budget control parameter
    reflection_lm=teacher_lm,
    seed=SEED,
)
```

### Additional Changes

Removed all references to `max_metric_calls` from:
- Optimizer configuration dictionaries
- Config display tables (Rich table output)
- Markdown report templates
- Results dictionaries

### Verification

```bash
# Dry run successfully initializes GEPA optimizer
uv run python scripts/dry_run_cost_estimate.py \
  --cache-dir data/research \
  --output results/cached_optimization/dry_run_report.json
```

**Result**: GEPA initialization successful ✅

---

## Error 2: Dataset Field Name Mismatch

### Symptom

```
TypeError: GreenhouseDetector.forward() got an unexpected keyword argument 'company_name'
```

From error logs:
```
ERROR dspy.utils.parallelizer: Error for Example({'company_name': 'Alex Warmerdam',
'location': 'Noordwijkerhout', ...}): GreenhouseDetector.forward() got an
unexpected keyword argument 'company_name'.
```

### Root Cause

Mismatch between dataset Example field names and GreenhouseDetector method signature:

**Dataset provided:**
- `company_name`
- `location`

**GreenhouseDetector.forward() expects:**
- `location_name`
- `location_area`

### Method Signature (from `src/sevenrad_ee/ai/dspy_greenhouse.py:87`)

```python
def forward(
    self,
    location_name: str,
    location_area: str,
) -> dspy.Prediction:
    """
    Detect if a location is a greenhouse using RAG with Perplexity.

    Args:
        location_name: Name of the location/company
        location_area: Geographic area (city, region, country)
    """
```

### Affected Files

- `scripts/dry_run_cost_estimate.py` (lines 174-180)
- `notebooks/optimize_with_cached_data.py` (lines 101-107, 318-321)

### Fix Applied

**Before:**
```python
# Create dspy.Example
example = dspy.Example(
    company_name=company_name,  # ❌ Wrong field name
    location=location,          # ❌ Wrong field name
    is_greenhouse=is_greenhouse,
    uses_growlight=uses_growlight,
).with_inputs("company_name", "location")  # ❌ Wrong input fields
```

**After:**
```python
# Create dspy.Example (using parameter names that match GreenhouseDetector.forward())
example = dspy.Example(
    location_name=company_name,     # ✅ Correct field name
    location_area=location,         # ✅ Correct field name
    is_greenhouse=is_greenhouse,
    uses_growlight=uses_growlight,
).with_inputs("location_name", "location_area")  # ✅ Correct input fields
```

Also updated prediction calls:

**Before:**
```python
pred = optimized_detector(
    company_name=example.company_name,  # ❌ Wrong parameter
    location=example.location,          # ❌ Wrong parameter
)
```

**After:**
```python
pred = optimized_detector(
    location_name=example.location_name,  # ✅ Correct parameter
    location_area=example.location_area,  # ✅ Correct parameter
)
```

### Verification

```bash
# Dry run successfully processes examples
uv run python scripts/dry_run_cost_estimate.py \
  --cache-dir data/research \
  --samples 5
```

**Result**: Examples processed without TypeError ✅

---

## Error 3: Perplexity API Incompatibility (BLOCKER)

### Symptom

```
litellm.exceptions.BadRequestError: PerplexityException -
["At body -> response_format -> ResponseFormatText -> type: Input should be 'text'",
 "At body -> response_format -> ResponseFormatJSONSchema -> type: Input should be 'json_schema'",
 "At body -> response_format -> ResponseFormatJSONSchema -> json_schema: Field required",
 "At body -> response_format -> ResponseFormatRegex -> type: Input should be 'regex'",
 "At body -> response_format -> ResponseFormatRegex -> regex: Field required"]
```

### Root Cause

The `perplexity/sonar` model (selected in Phase 2.5) **does not support** the `response_format` parameter that DSPy uses for structured JSON outputs.

**Technical Details:**
- DSPy attempts to enforce output schema via `response_format` parameter
- Perplexity API rejects this parameter with validation error
- This is a fundamental model limitation, not a configuration issue

### Why This Matters

The Phase 2.5 model selection chose `perplexity/sonar` based on:
- No statistically significant difference vs Sonar Reasoning Pro (p=0.9978)
- Cheaper and faster than Sonar Reasoning Pro
- Baseline F1: 59.95%

However, this model cannot be used with DSPy's current structured output requirements.

### Impact

**Cannot proceed with Phase 3 GEPA optimization** until student model is changed or DSPy configuration is modified.

### Proposed Solutions

#### Option 1: Switch to OpenAI GPT-4o-mini (RECOMMENDED)

**Pros:**
- ✅ Supports structured outputs natively
- ✅ Fast and cost-effective ($0.15/1M input, $0.60/1M output)
- ✅ Well-tested with DSPy framework
- ✅ Minimal code changes required

**Cons:**
- ❌ Deviates from Phase 2.5 model selection
- ❌ Requires re-establishing baseline (optional)

**Command:**
```bash
uv run python notebooks/optimize_with_cached_data.py \
  --cache-dir data/research \
  --output-dir results/cached_optimization \
  --student-model openai/gpt-4o-mini \
  --teacher-model gemini/gemini-2.5-pro \
  --baseline-f1 0.5995
```

**Cost Estimate:**
- Similar to Perplexity Sonar
- Expected Phase 3 cost: $15-25

#### Option 2: Disable Structured Outputs in DSPy

**Pros:**
- ✅ Keeps original `perplexity/sonar` model selection
- ✅ Honors Phase 2.5 statistical analysis

**Cons:**
- ❌ Requires DSPy configuration changes
- ❌ May reduce optimization quality (less reliable output parsing)
- ❌ More complex to implement and test
- ❌ Potential for parsing errors during optimization

**Implementation Required:**
- Modify DSPy LM configuration to disable structured outputs
- Add robust output parsing fallbacks
- Test optimization pipeline with free-form outputs

**Risk:** Lower optimization quality, potential parsing failures

#### Option 3: Switch to Gemini 1.5 Flash for Student

**Pros:**
- ✅ Supports structured outputs
- ✅ Consistent ecosystem (both student and teacher from Google)
- ✅ Fast and cost-effective

**Cons:**
- ❌ Both models from same family (less diverse teacher-student relationship)
- ❌ Higher cost than OpenAI ($0.075/1M input, $0.30/1M output)

**Command:**
```bash
uv run python notebooks/optimize_with_cached_data.py \
  --cache-dir data/research \
  --output-dir results/cached_optimization \
  --student-model gemini/gemini-1.5-flash \
  --teacher-model gemini/gemini-2.5-pro \
  --baseline-f1 0.5995
```

**Cost Estimate:**
- Expected Phase 3 cost: $20-35

### Decision Required

**Next Action:** Choose one of the three options above to proceed with Phase 3.

**Recommendation:** Option 1 (OpenAI GPT-4o-mini) for best balance of compatibility, cost, and reliability.

---

## Lessons Learned

### 1. GEPA API Design

The GEPA optimizer enforces **exactly one budget control mechanism**:
- Use `auto` presets for budget-controlled optimization
- Use explicit budgets (`max_metric_calls`, `max_full_evals`) for fine control
- **Never mix both approaches**

### 2. Dataset Schema Alignment

Always verify dataset field names match the forward method signature:

```python
# Check method signature FIRST
def forward(self, location_name: str, location_area: str) -> dspy.Prediction:
    ...

# Then create Examples with matching field names
example = dspy.Example(
    location_name=company,
    location_area=location,
    ...
).with_inputs("location_name", "location_area")
```

### 3. Model Compatibility Research

Before selecting a student model, verify it supports required DSPy features:
- ✅ Check for structured output support (`response_format` parameter)
- ✅ Test a simple DSPy program before committing to optimization
- ✅ Consider running a quick compatibility check script

### 4. Phase 2.5 Enhancement Opportunity

Future model comparisons should include a **compatibility check** phase:

```python
def check_model_compatibility(model_name: str) -> dict[str, bool]:
    """Check if model supports DSPy features."""
    return {
        "structured_outputs": test_structured_outputs(model_name),
        "function_calling": test_function_calling(model_name),
        "streaming": test_streaming(model_name),
    }
```

This would have caught the Perplexity API limitation before Phase 3.

---

## Related Documentation

- `OPTIMIZATION_STRATEGY.md` - Complete 4-phase optimization strategy
- `results/diagnostics/model_comparison_report_repeated_cv.md` - Phase 2.5 results
- `scripts/dry_run_cost_estimate.py` - Cost estimation script
- `notebooks/optimize_with_cached_data.py` - Main Phase 3 optimization script
- `src/sevenrad_ee/ai/dspy_greenhouse.py` - GreenhouseDetector implementation

---

**Status**: Awaiting decision on Perplexity API blocker (Option 1, 2, or 3)
**Last Updated**: 2025-11-19
**Next Step**: Select student model approach and proceed with Phase 3 optimization
