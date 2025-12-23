# MIPROv2 Optimization Retrospective

**Date:** 2025-12-22
**Sprint Period:** 2025-11-11 to 2025-12-22
**Status:** Completed
**Result:** 67.9% average test score (up from 48.5%)

---

## Executive Summary

The MIPROv2 optimization with the Dutch PerplexityKasDetector completed successfully after resolving critical infrastructure issues. The test score improved from 48.5% to 67.9%, but remains below the 85%+ target. Analysis reveals specific gaps in domain knowledge coverage and model reasoning that can be addressed in future iterations.

---

## Key Takeaways

### 1. Review Misclassified Examples

Several examples were misclassified, requiring different corrective actions:

| Company | Score | Issue Type | Corrective Action |
|---------|-------|------------|-------------------|
| Nunhems Netherlands BV | 0.0 | Model searched for "kweker" but Nunhems is "veredelaar" | Extend search to "veredelaar of andere kaseigenaar" |
| J.H.A. Stolk Kwekerij | 0.25 | Buxus nursery → **model was CORRECT** (no growlights) | **Update ground truth label** to uses_growlight=NEE |
| Van den Bos Premium Ardisia's | 0.55 | Ardisia → **model said NEE but should be JA** | Growlight very likely for Ardisia (LED winterverkoop) |
| Vreugdenhil Bulbs & Plants | 0.55 | Bulb crops (Amaryllis) missing from domain knowledge | Add bulb crops to domain knowledge |

**Actions:**
1. **Nunhems:** Change search strategy from "kweker" to "kaseigenaar" (includes veredelaar, onderzoek, etc.)
2. **J.H.A. Stolk:** Update ground truth - buxus is outdoor, no growlights needed
3. **Ardisia:** Model failure - Ardisia IS in domain list, model didn't apply it correctly

### 2. Add More Robust Search Strategy

The current approach only looks for direct lighting evidence. A more sophisticated approach should reason from horticultural science and light spectrum optimization.

**Key Improvements:**

1. **Always indicate commercial cultivation** - distinguish from hobby/research
2. **Ask for "kaseigenaar" instead of "kweker"** - allows longer reasoning, avoids dead-ends like Nunhems
3. **Include light spectrum reasoning** - which colors are recommended by trade literature
4. **Reference manufacturers/suppliers** - Atophort, Signify, Philips for credibility

**New Search Strategy (Dutch):**

> **CONTEXT:** Dit betreft COMMERCIËLE glastuinbouw, niet hobby of onderzoek.
>
> **STAP 1: Identificeer de kaseigenaar**
> Zoek naar informatie over de kaseigenaar (niet alleen kweker, maar ook veredelaar, vermeerderaar, of ander type glastuinbouwbedrijf).
>
> **STAP 2: Plausibiliteitsredenering voor belichting**
> Als de kaseigenaar niet een traditionele kweker is maar bijvoorbeeld een veredelaar of ander type:
> - Stuurt deze op groei-optimalisatie door micro-mol sturing met LED-topverlichting of SON-T?
> - Is er vakliteratuur die aangeeft dat dit gewas beter groeit bij meer fotonen?
>
> **STAP 3: Lichtspectrum analyse**
> Welke kleuren in het lichtspectrum worden door vakliteratuur aanbevolen? Beantwoord verschillende scenarios:
>
> | Lichtspectrum | Doel |
> |---------------|------|
> | Meer rood, relatief minder blauw, gecontroleerde far-red | Maximaliseren bloem/vrucht productie |
> | Lagere R:FR-ratio | Bloemsteel, blad en productie in balans houden |
> | Meer blauw | Compactere groei, sterkere stengels |
> | Far-red pulsen | Bloeisturing, vervroeging |
>
> **STAP 4: Vakliteratuur en fabrikanten**
> Zoek naar redenen voor specifieke lichtspectra volgens:
> - Vakliteratuur (Wageningen, TNO, proefstations)
> - Fabrikanten: Atophort, Signify (Philips), Gavita, Fluence
> - Leveranciers en installateurs
>
> **STAP 5: Plausibiliteitsconclusie**
> - Welke fotosynthese heeft gedocumenteerd het meeste invloed op gewenste karakteristieken?
>   * Grotere bloem
>   * Sterkere bloem
>   * Betere houdbaarheid
>   * Intensere kleur
> - Is het aannemelijk dat de kaseigenaar LED en/of SON-T gebruikt?
> - Weeg kosten/baten af voor dit specifieke gewas en bedrijfstype

**Implementation:** Update `PerplexityKasClassificatie` signature with this reasoning chain.

---

## Git Commits in Scope

### Phase 3: Dutch Native RAG (2025-12-20 to 2025-12-22)
| Date | Commit | Description |
|------|--------|-------------|
| 2025-12-22 | `e2b031f` | Fix: update error dump to use Dutch field names |
| 2025-12-22 | `d9d71f2` | Update optimization script to use Dutch PerplexityKasDetector |
| 2025-12-22 | `444b9b9` | Add Dutch PerplexityKasClassificatie signature for improved classification |
| 2025-12-20 | `0dcbbf6` | Fix MIPROv2 optimizer parameters and add GrowlightUsage enum |
| 2025-12-20 | `9dea115` | Fix deprecated model names and add MIPROv2 optimizer |

### Phase 2.5-3: Perplexity Integration (2025-11-18 to 2025-11-22)
| Date | Commit | Description |
|------|--------|-------------|
| 2025-11-22 | `4372a6c` | fix: use full API key in Authorization header, not redacted version |
| 2025-11-22 | `3fac971` | Add comprehensive logging for GEPA optimization debugging |
| 2025-11-20 | `55a0ca8` | Fix PerplexityLM to properly handle model parameters |
| 2025-11-20 | `c571b73` | Update GEPA dry run script to use PerplexityLM for student model |
| 2025-11-20 | `9e89e6d` | Complete Phase 3 GEPA optimization with PerplexityLM integration |
| 2025-11-20 | `03adc7a` | Add comprehensive test suite for Perplexity DSPy integration |
| 2025-11-19 | `0ccf116` | fix: removes old troubleshooting docs |
| 2025-11-19 | `6fa7624` | Add Perplexity native structured outputs support for DSPy RAG |
| 2025-11-19 | `d5b0692` | Complete Phase 2.5 and fix Phase 3 GEPA optimization issues |

### Phase 1-2: Foundation (2025-11-11 to 2025-11-18)
| Date | Commit | Description |
|------|--------|-------------|
| 2025-11-18 | `fa319c7` | Complete Phase 2.5: Extended model comparison with DSPy fix |
| 2025-11-18 | `25fe83e` | Complete Phase 2: Baseline & Model Selection |
| 2025-11-18 | `327e287` | Add Phase 1: Architecture Foundation for DSPy Optimization |
| 2025-11-18 | `d9bb07c` | Add complete iterative refinement infrastructure for dataset expansion |
| 2025-11-11 | `a847ef7` | Add comprehensive plan for 65-example GEPA optimization |
| 2025-11-11 | `9e788a6` | Add manual classification update script |

---

## Timeline of Issues and Fixes

### Issue 1: KeyboardInterrupt (2025-12-20)

**Symptom:** Process interrupted during socket read
**Root Cause:** Manual/timeout interruption of a working process
**Resolution:** N/A - was not a bug, process was functioning correctly

### Issue 2: HTTP 401 Authorization Required (2025-12-22)

**Symptom:** Immediate failure with Cloudflare 401 response
**Root Cause:** Perplexity API quota exhausted
**Resolution:** User added API quota credits

**Root Cause Chain:**
```
Exit code 1
    └── PerplexityAPIError: "401 Unauthorized"
        └── Cloudflare rejected Authorization header
            └── API quota/credits exhausted
                └── .env unchanged since Oct 25 (no key rotation needed, just credits)
```

**Lesson Learned:** Add pre-flight API validation check before starting long-running optimizations.

---

## Test Results Analysis

### Score Distribution

| Score | Count | Percentage |
|-------|-------|------------|
| 1.0 (Perfect) | 6 | 46% |
| 0.55 (Partial) | 4 | 31% |
| 0.375 | 1 | 8% |
| 0.25 | 1 | 8% |
| 0.0 (Failure) | 1 | 8% |

### Perfect Scores (1.0) - What Worked

| Company | Crop | Why It Worked |
|---------|------|---------------|
| Tomatenkwekerij A. de Bruijn | tomaten | Common crop, clear domain knowledge match |
| Kwekerij De Opstal | trosrozen | Roses explicitly listed as "BIJNA ALTIJD BELICHT" |
| Vereijken 's-Gravenzande | trostomaten | Common crop, clear match |
| Bernhard Optimum | Rozen/Phalaenopsis | Both crops in domain knowledge |
| Vereijken Kwekerijen | trostomaten | Common crop |
| Batist Westmade | gerbera | Gerbera explicitly listed |

**Pattern:** Common greenhouse crops (tomatoes, roses, gerbera) that appear explicitly in the domain knowledge are classified correctly.

### Partial Scores (0.55) - Close But Wrong

| Company | Crop Found | Predicted | Expected | Issue |
|---------|------------|-----------|----------|-------|
| Van den Bos Premium Ardisia's | Ardisia crenata | NEE | JA | **Ardisia IS in domain list** - model failed to apply |
| Vreugdenhil Bulbs & Plants | Amaryllis, Zantedeschia | ONBEKEND | JA | Bulb crops not in domain knowledge |
| N.L. van Geest bv | Amaryllis (Hippeastrum) | ONBEKEND | JA | Bulb crops not in domain knowledge |
| WPK - Westlandse Plantenkwekerij | opkweek tomaten/paprika | ONBEKEND | JA | Model didn't infer from "opkweek" context |

### Failures - Root Cause Analysis

#### Nunhems Netherlands BV (Score: 0.0)
- **Predicted:** is_kas=false, groeilicht=ONBEKEND
- **Expected:** is_kas=true, groeilicht=JA
- **Crop Found:** None identified
- **Confidence:** 0.95 (HIGH!)
- **Issue:** Seed breeding company → model assumed no greenhouse, but Nunhems HAS greenhouses for breeding
- **Domain Knowledge Gap:** Signature says "Zaadveredelingsbedrijf MET kassen = true" but model didn't find greenhouse evidence

#### J.H.A. Stolk Kwekerij (Score: 0.25)
- **Predicted:** is_kas=false, groeilicht=ONBEKEND
- **Expected:** is_kas=true, groeilicht=JA
- **Crop Found:** Buxus (buxusplanten)
- **Issue:** Buxus is outdoor/container nursery, not glasshouse. Model correctly identified buxus but ground truth may be incorrect?
- **Question:** Is this a ground truth labeling issue?

#### Kwekerij Jongland B.V. (Score: 0.375)
- **Predicted:** is_kas=true, groeilicht=ONBEKEND
- **Expected:** is_kas=true, groeilicht=JA
- **Crop Found:** None
- **Issue:** Failed to identify crop despite company being a known nursery

---

## Gap Analysis: Domain Knowledge vs Failures

### Missing from Domain Knowledge

| Crop | Examples Affected | Recommendation |
|------|-------------------|----------------|
| Amaryllis/Hippeastrum | 2 (Vreugdenhil, N.L. van Geest) | Add to "VAAK BELICHT" list with LED/SON-T |
| Zantedeschia (Calla) | 1 (Vreugdenhil) | Add to "VAAK BELICHT" list |
| Buxus | 1 (J.H.A. Stolk) | Add to "ZELDEN BELICHT" or clarify as outdoor |

### Present But Not Applied

| Crop | In Domain List? | Model Applied? | Issue |
|------|-----------------|----------------|-------|
| Ardisia | Yes ("LED voor winterverkoop") | No (predicted NEE) | Model failed to match "Ardisia crenata" to "Ardisia" |

---

## Architecture Changes Summary

### Before (Pre-November 2025)
```
Perplexity Search → Cache → Text → DSPy Predictor → Parse (3+ LLM calls)
```

### After (December 2025)
```
Perplexity RAG (search + reason + cite in ONE call) (1 LLM call)
```

### Key Improvements
| Change | Commit | Impact |
|--------|--------|--------|
| Native RAG architecture | `6fa7624` | Reduced API calls by 66% |
| Dutch signature (PerplexityKasClassificatie) | `444b9b9` | Better Dutch source discovery |
| Comprehensive logging | `3fac971` | Enabled root cause analysis |
| GrowlightUsage enum | `0dcbbf6` | Type-safe classifications |
| API key fix | `4372a6c` | Fixed authorization header |

---

## Recommendations for Next Sprint

### Priority 1: Review Misclassified Examples

Review all examples with score < 1.0:
- Verify ground truth labels are correct
- Identify if model was actually right and ground truth wrong
- Update training data accordingly

### Priority 2: Implement Robust Search Strategy

Add photosynthesis-based plausibility reasoning to signature:

```python
# Add to PerplexityKasClassificatie docstring:

STAP 6: Plausibiliteitsredenering voor belichting

Als niet kweker maar veredelaar:
- Stuurt deze op groei-optimalisatie door micro-mol sturing?
- Is LED-topverlichting of SON-T gebruikelijk voor dit teeltdoel?

Indien opkweek en/of bloei:
- Is er vakliteratuur die aangeeft dat dit gewas beter groeit bij meer fotonen?
- Welke fotosynthese heeft gedocumenteerd het meeste invloed op:
  * Grotere bloem
  * Sterkere bloem
  * Betere houdbaarheid
  * Intensere kleur

Redeneer vanuit plausibiliteit:
- Is het aannemelijk dat er LED en/of SON-T gebruikt wordt?
- Weeg kosten/baten af voor dit specifieke gewas
```

### Priority 3: Add Pre-flight Validation

```python
def validate_api_key() -> bool:
    """Test API key before starting optimization."""
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": "test"}],
            model="sonar-pro"
        )
        return response.status_code == 200
    except Exception:
        return False
```

### Priority 4: Expand Domain Knowledge

Add missing bulb/flower crops:
```python
GEWASSEN DIE VAAK BELICHT WORDEN (zeg JA bij jaarrond):
# ADD:
- Amaryllis (Hippeastrum): SON-T/LED voor geforceerde bloei
- Zantedeschia (Calla): LED, vaak belichting voor jaarrond
- Hyacint (forcering): SON-T in trek
```

---

## Metrics Summary

| Metric | Before | After | Target | Gap |
|--------|--------|-------|--------|-----|
| Average Score | 48.5% | 67.9% | 85%+ | -17% |
| Perfect Scores | ? | 46% | 70%+ | -24% |
| Complete Failures | ? | 8% | 0% | -8% |
| Duration | N/A | 25.5 min | <60 min | ✓ |
| API Success | 0% | 100% | 100% | ✓ |

---

## Conclusion

The optimization pipeline is now functional and improved by +19.4 percentage points. The remaining gap to 85%+ is primarily due to:

1. **Shallow reasoning** - Model looks for direct evidence instead of plausibility reasoning
2. **Domain knowledge gaps** (bulb crops) - Easy fix
3. **Fuzzy crop matching** (Ardisia crenata → Ardisia) - Medium fix
4. **Edge cases** (seed breeders with greenhouses) - Needs investigation
5. **Possible ground truth issues** (Buxus nursery) - Needs review

The architecture is sound. Focus next sprint on:
1. **Reviewing misclassified examples** for ground truth accuracy
2. **Implementing robust plausibility-based search strategy** using photosynthesis science

---

## Appendix: Full Test Results

| Company | Crop | Score | Predicted | Expected | Feedback |
|---------|------|-------|-----------|----------|----------|
| Van den Bos Premium Ardisia's | Ardisia crenata | 0.55 | NEE | JA | Domain knowledge not applied |
| Tomatenkwekerij A. de Bruijn | tomaten | 1.0 | JA | JA | ✓ |
| Kwekerij De Opstal | trosrozen | 1.0 | JA | JA | ✓ |
| Vereijken 's-Gravenzande | trostomaten | 1.0 | JA | JA | ✓ |
| Nunhems Netherlands BV | (none) | 0.0 | false/ONBEKEND | true/JA | Seed breeder misclassified |
| Vreugdenhil Bulbs & Plants | Amaryllis, Zantedeschia | 0.55 | ONBEKEND | JA | Bulb crops missing |
| N.L. van Geest bv | Amaryllis | 0.55 | ONBEKEND | JA | Bulb crops missing |
| Bernhard Optimum | Rozen, Phalaenopsis | 1.0 | JA | JA | ✓ |
| WPK - Westlandse Plantenkwekerij | opkweek tomaten/paprika | 0.55 | ONBEKEND | JA | Opkweek context not inferred |
| Vereijken Kwekerijen | trostomaten | 1.0 | JA | JA | ✓ |
| J.H.A. Stolk Kwekerij | Buxus | 0.25 | false/ONBEKEND | true/JA | Ground truth issue? |
| Batist Westmade | gerbera | 1.0 | JA | JA | ✓ |
| Kwekerij Jongland B.V. | (none) | 0.375 | true/ONBEKEND | true/JA | Crop not identified |
