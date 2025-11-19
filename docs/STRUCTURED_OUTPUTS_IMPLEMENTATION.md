# Perplexity Structured Outputs Implementation

## Overview

This document describes the implementation of Perplexity's native structured output API for the DSPy RAG pipeline, addressing critical gaps identified through Gemini's analysis.

## Problem Statement (From Gemini Analysis)

### Critical Gaps Identified

1. **Missing Native API**: Implementation completely omitted `response_format` parameter
2. **Anti-Pattern**: Manual JSON parsing when API provides native enforcement
3. **Type Safety Gap**: String-based DSPy `OutputField` with no enforcement
4. **Optimization Impact**: GEPA optimizer wasting cycles on formatting vs. reasoning failures
5. **Reliability Loss**: Exposed to non-deterministic JSON parsing failures

### Impact

- **Brittleness**: JSON parsing errors pollute GEPA optimization signal
- **Reduced Accuracy**: Model focuses on syntax formatting instead of reasoning
- **Unnecessary Complexity**: Boilerplate error handling for solved problem
- **Wasted API Costs**: Failed predictions due to formatting, not logic

## Implementation Status

### ✅ Phase 1: PerplexityClient Refactor (COMPLETED)

**File**: `src/sevenrad_ee/ai/perplexity_client.py`

**Changes**:
- Added `response_model` parameter to `query()` method
- Implemented type-safe overloads for typed vs. untyped queries
- Added `response_format` parameter injection when Pydantic model provided
- Schema-aware cache key generation
- Increased timeout to 60s for first schema compilation
- Helper method `_parse_structured_response()` for JSON → Pydantic parsing

**Key Features**:
```python
# Type-safe query with structured output
result = client.query(
    query="Analyze Marjoland greenhouse",
    response_model=SimplifiedGreenhouseAnalysis,  # Pydantic model
)
# result is typed as SimplifiedGreenhouseAnalysis, not PerplexityResponse!
print(result.uses_growlight)  # Literal["YES", "NO", "UNKNOWN"] - guaranteed!
```

**Backward Compatibility**:
```python
# Old behavior still works
response = client.query("Search query")  # Returns PerplexityResponse
```

**Type Safety**: Full mypy compliance with overloaded signatures

### ✅ Phase 1.5: StructuredPredictor Module (CREATED)

**File**: `src/sevenrad_ee/ai/dspy_structured.py`

**Purpose**: Bridge DSPy and Perplexity structured outputs

**Status**: Basic structure created, full integration pending

### ✅ Phase 1.6: Test Script (CREATED)

**File**: `scripts/test_structured_outputs.py`

**Purpose**: Validate PerplexityClient structured output functionality

**Test Coverage**:
- Pydantic model validation
- API response_format parameter
- Cache key schema hashing
- Type safety checks

**Usage**:
```bash
uv run python scripts/test_structured_outputs.py
```

## Remaining Work

### 🔄 Phase 2: DSPy Integration (IN PROGRESS)

**Tasks**:
1. Complete StructuredPredictor implementation
2. Integrate with existing retrievers (CachedRetriever, PerplexityRetriever)
3. Create custom DSPy signatures for structured outputs
4. Test with GEPA optimizer

**Design Decision Needed**:
- Should DSPy use native LM structured outputs or wrapper pattern?
- How to handle DSPy's prompt optimization with fixed schemas?

### 📋 Phase 3: Update GreenhouseDetector (PENDING)

**File**: `src/sevenrad_ee/ai/dspy_greenhouse.py`

**Changes**:
- Replace `dspy.ChainOfThought` with `StructuredPredictor`
- Pass `SimplifiedGreenhouseAnalysis` as response_model
- Remove post-hoc string parsing in `to_pydantic()` method
- Update forward() to use structured predictions

**Expected Impact**:
- Eliminate all JSON parsing errors during GEPA optimization
- Improve F1 score through cleaner optimization signal
- Reduce API costs from failed formatting attempts

### 📋 Phase 4: Testing & Validation (PENDING)

**Test Scenarios**:
1. Test structured outputs with known examples from `data/research/*.json`
2. Run GEPA optimization with CachedRetriever (frozen evidence)
3. Compare optimization metrics: before vs. after
4. Validate against known negatives (0% false positive rate)

**Success Criteria**:
- ✅ No JSON parsing errors during optimization
- ✅ F1 score maintained or improved
- ✅ GEPA converges faster (fewer wasted generations)
- ✅ All Pydantic validations pass

### 📋 Phase 5: Documentation & Notebooks (PENDING)

**Updates Needed**:
- `notebooks/dspy_optimization_pipeline.py`: Add structured output examples
- `notebooks/optimize_with_cached_data.py`: Update with new approach
- Module docstrings: Document structured output usage
- Examples: Show Pydantic model → DSPy integration

## Technical Details

### Schema Compilation

**First Request**: 10-30 seconds (Perplexity compiles schema)
**Subsequent Requests**: No delay (schema cached server-side)

**Implementation**:
- Timeout increased to 60s in `_execute_request()`
- User notified of initial delay in console output

### Cache Strategy

**Cache Key**: Includes schema hash to ensure different schemas get different caches

```python
if response_model is not None:
    schema = response_model.model_json_schema()
    config_dict["__schema_hash__"] = str(hash(str(schema)))
```

**Cached Response**: Stored as `PerplexityResponse.content` (JSON string)
**Retrieval**: Parsed into Pydantic model on cache hit

### API Payload

**Without Structured Output**:
```json
{
  "model": "sonar-pro",
  "temperature": 0.0,
  "messages": [{"role": "user", "content": "..."}]
}
```

**With Structured Output**:
```json
{
  "model": "sonar-pro",
  "temperature": 0.0,
  "messages": [{"role": "user", "content": "..."}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "SimplifiedGreenhouseAnalysis",
      "schema": {...}  # Full Pydantic JSON schema
    }
  }
}
```

## Gemini's Recommendations

### ✅ Implemented

1. **Target the Predictor**: Focused refactor on classification phase (not retrieval)
2. **PerplexityClient Enhancement**: Added optional Pydantic model support
3. **Backward Compatibility**: Maintained existing code behavior
4. **Type Safety**: Full mypy compliance with overloaded signatures

### 🔄 In Progress

5. **StructuredPredictor Module**: Basic structure created, full integration pending

### 📋 Remaining

6. **Update Signatures**: Create DSPy signatures referencing Pydantic models
7. **End-to-End Testing**: Validate with GEPA optimizer on cached data

## Benefits Realized (Once Complete)

### Reliability
- **Before**: JSON parsing failures pollute optimization runs
- **After**: API guarantees valid JSON conforming to schema

### Type Safety
- **Before**: `uses_growlight: str` (could be "MAYBE", "I don't know", etc.)
- **After**: `uses_growlight: Literal["YES", "NO", "UNKNOWN"]` (enforced at API level)

### Optimization
- **Before**: GEPA wastes generations fixing JSON syntax
- **After**: GEPA focuses purely on reasoning quality

### Maintainability
- **Before**: Complex error handling for JSON parsing failures
- **After**: Entire failure class eliminated

## Next Steps

1. **Run Test Script**:
   ```bash
   uv run python scripts/test_structured_outputs.py
   ```

2. **Complete StructuredPredictor**: Integrate with DSPy LM layer

3. **Update GreenhouseDetector**: Replace ChainOfThought with StructuredPredictor

4. **GEPA Optimization Test**: Compare F1 scores before/after

5. **Update Notebooks**: Document structured output approach

## References

- **Perplexity Docs**: https://docs.perplexity.ai/guides/structured-outputs
- **Gemini Analysis**: `continuation_id: 60fbd6c9-be38-4ea6-b05c-6df42a24cdb2`
- **Implementation PR**: TBD

---

**Status**: Phase 1 Complete | Phase 2-5 In Progress
**Last Updated**: 2025-11-19
**Author**: Claude Code with Gemini validation
