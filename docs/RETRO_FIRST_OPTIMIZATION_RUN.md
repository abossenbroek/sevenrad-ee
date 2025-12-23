# MIPROv2 Optimization Retrospective - First Run

**Date:** 2025-12-20
**Result:** 62.08% avg score (target: 85%+)
**Status:** Improvement from 48.5% baseline

---

## Executive Summary

The first MIPROv2 optimization run improved from 48.5% to 62.08%, but fell short of the 85% target. The main gap is **uses_growlight accuracy (53.8%)** - the model defaults to UNKNOWN when it should apply causal reasoning from crop types.

---

## What Went Well

### Architecture Improvements (Phase 3)
- **Native Perplexity RAG works:** Single API call for search + reason + cite
- **Eliminated over-engineering:** Removed separate retriever → classifier → verifier pipeline
- **PerplexityLM integration:** Custom DSPy wrapper bypasses LiteLLM, enables native structured outputs
- **All 13 test predictions returned valid responses** (no NaN scores)

### Infrastructure Fixes
| Commit | Fix |
|--------|-----|
| 4372a6c | Full API key in Authorization header |
| 9dea115 | sonar-pro instead of deprecated pplx-* models |
| 0dcbbf6 | 20 candidates, GrowlightUsage enum |
| 3fac971 | Comprehensive logging for debugging |

### Successful Classifications (7/13 = 54%)
| Company | Score | Correct On |
|---------|-------|------------|
| Vereijken Kwekerijen | 84.1% | is_greenhouse + uses_growlight |
| Vereijken 's-Gravenzande | 83.5% | Both |
| Vreugdenhil Bulbs & Plants | 82.4% | Both |
| Bernhard Optimum | 80.7% | Both |
| Batist Westmade | 80.7% | Both |
| Tomatenkwekerij A. de Bruijn | 79.8% | Both |
| Kwekerij De Opstal | 79.6% | Both |

---

## What Went Wrong

### uses_growlight Classification Failures (6/13 = 46%)
| Company | Predicted | Expected | Score |
|---------|-----------|----------|-------|
| Van den Bos Premium Ardisia's | UNKNOWN | YES | 42.5% |
| Nunhems Netherlands BV | false (wrong is_greenhouse too!) | YES | 10.5% |
| N.L. van Geest bv | UNKNOWN | YES | 47.2% |
| WPK - Westlandse Plantenkwekerij | UNKNOWN | YES | 51.9% |
| J.H.A. Stolk Kwekerij | UNKNOWN | YES | 44.7% |
| Kwekerij Jongland B.V. | UNKNOWN | YES | 39.7% |

**Pattern:** Model defaults to UNKNOWN when insufficient Dutch terminology is found

### Root Causes

1. **Dutch Terminology Not Found**
   - 11/13 predictions: "NO Dutch terminology found - search strategy may be ineffective"
   - Perplexity's web search doesn't naturally surface Dutch horticultural terms

2. **Causal Reasoning Not Applied**
   - Signature has crop-to-lighting rules, but model doesn't apply them
   - Van den Bos grows Ardisia's (often lit) → Should infer YES, got UNKNOWN

3. **Nunhems Misclassification**
   - Model thought "seed company = not greenhouse" (wrong!)
   - Nunhems HAS 2.5 hectare veredelingskassen (breeding greenhouses)
   - Source: onderglas.nl confirms Bayer building greenhouses in Nunhem

4. **Optimization Interrupted**
   - KeyboardInterrupt during bootstrapping set 4/20
   - Results are from a previous successful run

---

## Technical Improvements Needed

### Priority 1: Nunhems Fix (Easy)
**Problem:** Model conflates business type with facility type

**Solution:** Add to signature instructions:
```
IMPORTANT: Business type and facility type are DIFFERENT:
- Seed company WITH greenhouses → is_greenhouse = TRUE
- Breeding facility WITH greenhouses → is_greenhouse = TRUE
- Research center WITH greenhouses → is_greenhouse = TRUE

is_greenhouse = TRUE if they HAVE greenhouse structures,
regardless of whether production, breeding, or research is primary business
```

### Priority 2: Multi-Stage Causal Reasoning (Challenged by GPT-5.2 & O3)

**Original Proposal:** 4 separate stages (crops → seasonality → academic lighting lookup → causal reasoning)

**Consensus Challenge Result:**
- GPT-5.2 (FOR): 4-stage provides intermediate supervision for MIPROv2
- O3 (AGAINST): Error propagation (0.9)³ ≈ 73%, 4× cost, Stage 3 academic RAG is brittle

**Revised Plan: 2-Stage Hybrid** (compromise)

```
STAGE 1: Structured Extraction (ONE Perplexity call)
├── Find crops grown (Dutch + English bilingual queries)
├── Determine seasonality (summer-only vs year-round)
├── Output: {crops: [], seasonality: "year-round"|"summer"|"unknown"}
└── Metrics: Crop F1 >80%, Seasonality accuracy >85%, Dutch coverage >50%

STAGE 2: Rule-Augmented Reasoning (ONE call + curated lookup + RAG fallback)
├── CURATED LOOKUP TABLE (extended):
│
│   ALMOST_ALWAYS_LIT (>90% of commercial growers use lighting):
│   ├── Cut flowers: roses, gerbera, alstroemeria, freesia, lisianthus, gypsophila
│   ├── Vegetables: tomatoes (year-round), bell peppers (year-round), cucumbers (year-round)
│   ├── Orchids: phalaenopsis, dendrobium, oncidium
│   └── Pot plants: kalanchoe, begonia (winter), poinsettia
│
│   OFTEN_LIT (50-90% use lighting, depends on season/market):
│   ├── Vegetables: eggplant, strawberries (winter), herbs (year-round)
│   ├── Cut flowers: carnations, chrysanthemums (timing), lilies (forcing)
│   ├── Pot plants: cyclamen, primula, hydrangea
│   └── Berries: raspberries (winter), blueberries (extended season)
│
│   SOMETIMES_LIT (20-50% use lighting, often specialty/premium):
│   ├── Bulbs: tulips (forcing), hyacinths, daffodils
│   ├── Orchids: cymbidium (some growers)
│   └── Cut flowers: ranunculus, anemone
│
│   RARELY_LIT (<20% use lighting):
│   ├── Tropical: anthurium (most daylight), bromeliads
│   ├── Succulents: cacti, echeveria
│   ├── Outdoor/seasonal: bedding plants, garden plants
│   └── Seeds/propagation: seed production (usually daylight)
│
├── RULE ENGINE:
│   IF crops ∈ ALMOST_ALWAYS_LIT AND seasonality == year-round
│   THEN uses_growlight = YES (confidence 0.90)
│
│   IF crops ∈ OFTEN_LIT AND seasonality == year-round
│   THEN uses_growlight = YES (confidence 0.75)
│
│   IF crops ∈ SOMETIMES_LIT
│   THEN uses_growlight = UNKNOWN → trigger RAG fallback
│
│   IF crops ∈ RARELY_LIT AND no explicit lighting evidence
│   THEN uses_growlight = NO (confidence 0.70)
│
│   OVERRIDE: "onbelichte teelt" or "daglichtkas" found → NO
│   OVERRIDE: "assimilatiebelichting" or "SON-T" or "LED" found → YES
│
├── RAG FALLBACK (for species NOT in lookup table):
│   1. Search: "{species} assimilatiebelichting" OR "{species} grow light requirements"
│   2. Search: "{species} DLI requirements" OR "{species} photoperiod"
│   3. Search Dutch extension: "WUR {species} belichting" OR "kasgroenten {species}"
│   4. If no evidence found → uses_growlight = UNKNOWN (with low confidence)
│
└── Metrics: uses_growlight accuracy >85%, UNKNOWN rate <15%, rule hit rate >70%
```

**Why 2-Stage Instead of 4-Stage:**
- Lower error propagation: (0.9)² = 81% vs (0.9)³ = 73%
- Half the API cost: 2 calls vs 4 calls per grower
- Curated knowledge more reliable than academic RAG (<40% recall)
- Still provides intermediate supervision for MIPROv2
- Matches industry practice (Climate FieldView, Xarvio use extraction + rules overlay)

**Extensibility:** New species can be added to lookup table as discovered. RAG fallback ensures coverage for rare/new crops.

### Priority 3: Dutch Search Strategy
**Problem:** Perplexity searches English-first, misses Dutch terms

**Solution:** Multi-language search instructions:
```
SEARCH STRATEGY:
1. First search: "{company} site:.nl" (Dutch sources)
2. Then search: "{company} assimilatiebelichting OR belichte teelt OR kunstlicht"
3. Then search: "{company} glastuinbouw kwekerij"
4. Finally: "{company}" (general info)

Also search Dutch trade media: kasmagazine.nl, onderglas.nl, agf.nl
```

### Priority 4: Checkpointing
**Problem:** Long optimization runs can be interrupted

**Solution:** Add checkpoint/resume support:
```python
import signal

def save_checkpoint(bootstrap_set_idx, demos, scores):
    checkpoint_path = f"results/checkpoints/bootstrap_{bootstrap_set_idx}.json"
    save_json(checkpoint_path, {"demos": demos, "scores": scores})

def handle_interrupt(sig, frame):
    logger.info("Interrupt received, saving progress...")
    save_checkpoint(current_set, current_demos, current_scores)
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt)
```

---

## Data Science Improvements (If Score < 80%)

### Training Data Diversity
- **Collect 10-15 negative examples:** Auction houses, transport, seed cos WITHOUT greenhouses
- **Collect 5-10 NO-growlight examples:** Cymbidium, tulips, daglichtkas operations
- **Add edge cases:** Seed/breeding companies WITH greenhouses (Nunhems-style)

### Budget Expansion ($30-50)
- **Increase num_candidates to 30:** More prompt variations
- **Expand validation set:** From 11 to 15-20 examples
- **Consider two-stage optimization:** Search instructions first, then causal reasoning

---

## Implementation Order

```
Phase F (Nunhems Fix) ──────────────────┐  ← Easy fix first
                                        │
Phase A (Causal Reasoning) ─────────────┼──► Run Optimization ──► Evaluate
                                        │
Phase B (Dutch Search) ─────────────────┤
                                        │
Phase D (Checkpointing) ────────────────┘

                    ↓ (if score < 80%)

Phase C (Training Data) ────────────────┐
                                        ├──► Run Optimization ──► Evaluate
Phase E (Budget Expansion) ─────────────┘
```

**Order:** F → A → B → D → Run → (if <80%: C → E → Run)

---

## Files to Modify

| File | Phase | Changes |
|------|-------|---------|
| `src/sevenrad_ee/ai/dspy_greenhouse.py` | F, A, B | Business/facility distinction, causal reasoning, search instructions |
| `scripts/run_miprov2_optimization.py` | D | Checkpoint/resume, SIGINT handler |
| `src/sevenrad_ee/ai/dspy_training_data.py` | C | Negative examples, NO-growlight cases |

---

## Key Metrics to Track

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| **Avg Test Score** | 62.08% | 85%+ | Primary optimization target |
| **Std Test Score** | 22.93% | <10% | Indicates consistency |
| **is_greenhouse Accuracy** | 92.3% (12/13) | >95% | Mostly solved |
| **uses_growlight Accuracy** | 53.8% (7/13) | >85% | Main improvement area |
| **Dutch Terms Found** | ~2/13 predictions | >80% | Search strategy issue |

---

## User Priorities (Confirmed)

1. **Improve uses_growlight accuracy** - 46% failure rate is main gap
2. **Fix Dutch search strategy** - ensure Perplexity searches Dutch terminology
3. **Add training data diversity** - negative examples and NO-growlight cases
4. **Add robustness (checkpointing)** - prevent losing progress
5. **Improve causal reasoning** - deduce from crop types, not direct search
6. **Increase search breadth** - leverage more sources

**Budget:** Increased to $30-50 (allows 30+ candidates)

**Nunhems Confirmed:** IS a greenhouse (Bayer veredelingskassen)
- Source: onderglas.nl - "2,5 hectare moderne veredelingskassen in Nunhem"

---

## Phase 4: Validated Improvements (2025-12-22)

### Critical Domain Insight

> **GEEN EEN TELER ZAL ZEGGEN DAT ZE ASSIMILATIEVERLICHTING GEBRUIKEN**
> (No grower will say they use assimilation lighting)

Growers do NOT advertise their lighting usage. Searching for terms like "assimilatiebelichting" or "SON-T" is **fundamentally flawed** - these terms don't appear in company information.

### Validated Solution: Full Dutch + Causal Reasoning

**Key insight:** Instead of searching for lighting terms, identify the **crop** and infer lighting from domain knowledge.

**Approach:**
1. Write entire DSPy signature in Dutch → forces Perplexity to search Dutch sources
2. Use 3-step causal reasoning: Find crop → Determine seasonality → Infer lighting
3. Embed crop→lighting lookup table directly in signature

**New Signature:** `PerplexityKasClassificatie`
```python
class PerplexityKasClassificatie(dspy.Signature):
    """Classificeer of een locatie een commerciële kas is.

    Stap 1: Zoek informatie over dit bedrijf - wat telen ze?
    Stap 2: Als het een kas is, bepaal het hoofdgewas.
    Stap 3: Leid af of belichting waarschijnlijk is op basis van het gewas.
    """

    # Output includes: is_kas, hoofdgewas, seizoen, gebruikt_groeilicht, redenering
```

### Red-Team Validated Mitigations

1. **Expand crop table** - Original list was incomplete
   - Added: Ardisia, lisianthus, gypsophila, kalanchoe, cyclamen, begonia

2. **Fallback for unknown crops** - Prevent excessive ONBEKEND
   - "Als het gewas niet in lijst staat, zoek aanvullende informatie"

3. **Business ≠ Facility distinction** - Fix Nunhems-style cases
   - "Zaadveredelingsbedrijf MET kassen → is_kas = True"

4. **Test gate before full run** - Prevent wasted optimization
   - Run 3-5 examples first, verify Dutch COT working

### Success Criteria

| Metric | Baseline | Target |
|--------|----------|--------|
| uses_growlight accuracy | 53.8% | >70% |
| Crop identification rate | N/A | >80% |
| Dutch in responses | ~15% | >50% |
| ONBEKEND rate | High | <30% |

### Deferred to Future Work

- Issue 2: 2-stage causal reasoning (if single-stage insufficient)
- Issue 4: Checkpointing (nice-to-have)
