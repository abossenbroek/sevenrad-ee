# Current State: GEPA Optimization for Greenhouse Attribution

**Documentation Type**: Technical Reference

**Last Updated**: 2025-12-20

---

## Overall Goal

Attribute high VIIRS DNB (nighttime satellite light) values to actual sources - specifically Dutch greenhouses using grow lights (assimilatiebelichting) for roses, tomatoes, etc.

**The Problem**: VIIRS has 750m x 750m swaths, making it difficult to pinpoint which company is the culprit.

**The Solution Pipeline**:
1. Get VIIRS DNB map → identify bright spots
2. Use Google Maps to find companies in those areas
3. Use Perplexity as grounded RAG search to compute a probability score (0-1) for each company

---

## Current Architecture

### Two-Stage Pipeline

```
Stage 1: RETRIEVE
├─ CachedRetriever: Loads pre-researched JSON evidence (for deterministic optimization)
└─ PerplexityRetriever: Live 3-query web search (for production)
    ├─ Query 1: Positive signals (assimilatiebelichting, groeilicht, LED)
    ├─ Query 2: Supplier associations (Signify, Hortilux, Philips, Gavita)
    └─ Query 3: Negative signals (onbelichte teelt, daglichtkas)

Stage 2: CLASSIFY
└─ GreenhouseDetector (DSPy ChainOfThought)
    ├─ is_greenhouse: bool
    ├─ uses_growlight: YES/NO/UNKNOWN/NOT_APPLICABLE
    ├─ confidence: float (0.0-1.0)
    └─ reasoning: str (with evidence citations)
```

### Key Innovation: LiteLLM Bypass

**Problem**: DSPy uses LiteLLM internally, but LiteLLM's `response_format` doesn't work with Perplexity's structured outputs.

**Solution**: Custom `PerplexityLM` wrapper that BYPASSES LiteLLM entirely:
```
DSPy Predictor → PerplexityLM → PerplexityClient → Perplexity API
                 (bypasses LiteLLM for structured outputs)
```

**Implementation Details**:
- `src/sevenrad_ee/ai/perplexity_client.py` - Direct API client with SHA-256 caching
- `src/sevenrad_ee/ai/dspy_perplexity.py` - DSPy-compatible LM wrapper
- `src/sevenrad_ee/ai/dspy_greenhouse.py` - GreenhouseDetector with hierarchical logic
- `src/sevenrad_ee/ai/dspy_evaluation.py` - Metrics with feedback for GEPA

See `docs/BYPASS_BASELM_FOR_PERPLEXITY_STUDENT.md` for technical details on the bypass implementation.

See `docs/PERPLEXITY_STRUCTURED_OUTPUTS_FIX.md` for structured outputs fix details.

---

## GEPA Optimization Setup

**GEPA** = Generative Error-driven Program-Aligning optimizer

### Configuration

- **Student Model**: PerplexityLM (sonar-pro) - makes predictions
- **Teacher Model**: Gemini 2.5 Pro - provides reflection feedback
- **Metric**: `dutch_aware_hierarchical_f1`
  - 70% classification accuracy (is_greenhouse + uses_growlight)
  - 15% Dutch terminology detection
  - 15% Evidence quality (tier-2 sources)

### Data Split

- Training: 41 examples (63%)
- Validation: 11 examples (17%)
- Test: 13 examples (20%)
- Total: 65 companies with research data

### Optimization Parameters

The GEPA optimizer uses the following configuration:
- Student model generates predictions with reasoning
- Teacher model provides reflection feedback on errors
- Metrics guide prompt improvements through multiple iterations
- Cached retrieval ensures deterministic optimization

---

## Current State: GEPA Run FAILED

### Phase 3 Optimization Results

**Summary** (`results/phase3_optimization/optimization_summary.json`):
```json
{
  "optimization_complete": true,
  "avg_test_score": NaN,   // ← FAILED
  "std_test_score": NaN,   // ← FAILED
  "num_train": 41,
  "num_val": 11,
  "num_test": 13
}
```

**Test Results** (`results/phase3_optimization/test_results.json`):
```json
{
  "test_scores": [],     // ← EMPTY
  "predictions": []      // ← EMPTY
}
```

### Root Cause Analysis: TWO Distinct Bugs

**Bug 1: API Key Bug (401 Unauthorized)**
- Error at 09:23: `401 Client Error: Unauthorized`
- **Root Cause**: Commit 4372a6c fixed sending REDACTED API key instead of full key
- **Status**: FIXED on Nov 22, but optimization may have run BEFORE the fix

**Bug 2: Invalid Model Name (400 Bad Request)**
- Error at 09:27: `Invalid model 'llama-3.1-sonar-large-128k-chat'`
- **Root Cause**: Old deprecated model name being used somewhere
- **Current Code**: Uses `sonar-pro` everywhere, but old name still appearing in errors
- **Status**: NEEDS INVESTIGATION - where is old model name coming from?

### Error Dump Evidence

26 error dumps in `results/phase3_optimization/`:
- ~13 with 401 Unauthorized (API key bug)
- ~13 with 400 Invalid model (deprecated model name)

---

## Data Quality Assessment

### Training Data

- 65 companies in `data/research/*.json`
- Each has 3 queries with Perplexity responses
- Evidence categorized: positive, negative, ambiguous
- Dutch terms extracted

### Labeled Examples

Source: `src/sevenrad_ee/ai/dspy_training_data.py`

- **TRAINING_SET_HARD_YES**: 17 confirmed greenhouse+growlight examples
- **NEGATIVE_SET**: 6 hard NO examples (non-greenhouses)
- **VALIDATION_SET**: 4 hold-out examples
- **Total labeled**: ~27 examples (may be insufficient for GEPA)

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/sevenrad_ee/ai/dspy_perplexity.py` | PerplexityLM wrapper (bypasses LiteLLM) |
| `src/sevenrad_ee/ai/dspy_greenhouse.py` | GreenhouseDetector with hierarchical logic |
| `src/sevenrad_ee/ai/dspy_evaluation.py` | Metrics with feedback for GEPA |
| `src/sevenrad_ee/ai/perplexity_client.py` | Direct API client with SHA-256 caching |
| `src/sevenrad_ee/ai/retrievers.py` | CachedRetriever + PerplexityRetriever |
| `scripts/run_phase3_gepa_optimization.py` | Full GEPA optimization script |
| `scripts/dry_run_cost_estimate.py` | Cost estimation before full run |
| `data/research/*.json` | Pre-researched company evidence |
| `results/phase3_optimization/` | GEPA optimization outputs and error dumps |

---

## Recent Commits Timeline

```
4372a6c Nov 22 - fix: use full API key (not redacted) ← CRITICAL FIX
3fac971 Nov 22 - Add comprehensive logging for GEPA debugging
55a0ca8 Nov 20 - Fix PerplexityLM parameter handling
c571b73 Nov 20 - Update GEPA dry run to use PerplexityLM
9e89e6d Nov 20 - Complete Phase 3 GEPA with PerplexityLM integration
```

---

## Next Steps

**Status**: GEPA optimization technically "completed" but produced NO valid predictions due to:
1. API key bug (fixed, but run may predate fix)
2. Deprecated model name leak (needs investigation)

**Action Required**:
1. Fix deprecated model names in test files
2. Verify API key and model name work correctly
3. Re-run GEPA optimization
4. Evaluate if prompt improvements help accuracy

**Alternative Approaches**:
- Consider MIPROv2 optimizer as alternative to GEPA
- Expand labeled training dataset beyond 27 examples
- Validate metrics align with business goals
