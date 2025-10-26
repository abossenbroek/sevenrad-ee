# DSPy Greenhouse Detection F1 Score Improvement Plan
**Target: 95-98% F1 Score | Web-Only Architecture | Real Data Expansion | GEPA/Advanced Optimization**

**Generated:** 2025-10-25
**Analysis by:** GPT-5-Pro + Gemini-2.5-Pro (Deep Analysis)

---

## Executive Summary

Your current DSPy greenhouse detection system suffers from **severe overfitting** (89% train vs 62% validation F1) and a **broken DSPy 3.0.3 installation**. Deep analysis by two frontier models (GPT-5-Pro and Gemini-2.5-Pro) identified the path to 95-98% F1 score through:

1. **Fix DSPy installation** - PyPI version 3.0.3 is broken; install from GitHub main
2. **Data expansion** - Add 40+ real Dutch greenhouse examples (currently 23 → target 60-65)
3. **Dutch-aware architecture** - Hierarchical classification with explicit Dutch terminology guidance
4. **GEPA optimizer** - Genetic Pareto optimization with reflection (or MIPRO fallback)
5. **Evidence-first validation** - Source-tiered web search with Dutch-language priority
6. **Robust cross-validation** - Repeated nested CV to eliminate overfitting

**Current Performance:**
- Training F1: 89.13%
- Validation F1: 62.50%
- **Overfitting Gap: 26.63 percentage points** ❌

**Target Performance:**
- Cross-validated F1: 95-98%
- Standard Deviation: <3%
- Overfitting Gap: <10% ✅

---

## Implementation Progress 🚀

**Last Updated:** 2025-10-25

### ✅ Completed

1. **Phase 1 Planning (Day 0)**
   - ✅ Consulted Gemini-2.5-Pro for detailed Phase 1 implementation strategy
   - ✅ Defined sequential, risk-minimizing approach with clear milestones
   - ✅ Added marimo~=0.9.0 to `data-augmentation` dependency group in pyproject.toml
   - ✅ Verified GEPA optimizer available (gepa==0.0.18) in DSPy GitHub installation

2. **Phase 1 Step 1.1: DSPy Installation (Day 0)**
   - ✅ Verified DSPy 3.0.4b2 installed from GitHub (working version)
   - ✅ Removed broken dspy-ai 3.0.3 from PyPI
   - ✅ Confirmed all optimizers importable (BootstrapFewShot, MIPROv2, GEPA)

3. **Phase 1 Step 1.2: SHA-256 Caching Infrastructure (Day 0)**
   - ✅ Created `src/sevenrad_ee/ai/perplexity_cache.py` with CacheManager and Pydantic v2 models
   - ✅ Created `src/sevenrad_ee/ai/perplexity_client.py` with rate limiting and error handling
   - ✅ Created `src/sevenrad_ee/operations/test_perplexity_cache.py` validation CLI tool
   - ✅ Implemented SHA-256 deterministic cache keys (query + config)
   - ✅ Added .env support for PERPLEXITY_API_KEY
   - ✅ All quality checks pass (ruff format, ruff check, mypy)
   - ✅ Ready for HITL validation with 3 sample Dutch greenhouse queries

4. **Phase 1 Step 1.3: Company Research Models & 3-Query Strategy (Day 0)**
   - ✅ Created `src/sevenrad_ee/ai/company_research_models.py` with comprehensive Pydantic v2 models
   - ✅ Created `src/sevenrad_ee/ai/company_researcher.py` implementing 3-query Perplexity strategy
   - ✅ Created `src/sevenrad_ee/operations/test_company_research.py` comprehensive validation CLI
   - ✅ Implemented evidence categorization (positive/negative/ambiguous)
   - ✅ Implemented source tier classification (company websites, supplier case studies, trade media)
   - ✅ Implemented Dutch terminology detection and confidence scoring
   - ✅ All quality checks pass (ruff format, ruff check, mypy)
   - ✅ **TESTED SUCCESSFULLY with 3 sample companies:**
     * **Porta Nova**: POSITIVE, 100% confidence, 51 evidence pieces, 63% tier-2
     * **Kwekerij Overgaag**: POSITIVE, 100% confidence, 34 evidence pieces, 59% tier-2
     * **BM Roses**: POSITIVE, 100% confidence, 32 evidence pieces, 47% tier-2

5. **Phase 2: Hierarchical Signature with Dutch Guidance (Day 0)**
   - ✅ Created EvidenceSource Pydantic v2 model with 5 tier types
   - ✅ Created GreenhouseDetectionOutput model with field validators
   - ✅ Implemented hierarchical gating validation (not greenhouse → growlight = UNKNOWN)
   - ✅ Implemented confidence-evidence alignment validation (high confidence requires tier-2)
   - ✅ Enhanced GreenhouseClassification DSPy signature with comprehensive Dutch guidance
   - ✅ Added Dutch terminology tracking and source tier classification
   - ✅ Created GreenhouseClassificationValidator with 4 validation rules
   - ✅ Comprehensive test coverage (34 tests, all passing)
   - ✅ All code quality checks pass (mypy clean, ruff clean)
   - ✅ Ready for Phase 3 metric integration

6. **Phase 3: Dutch-Aware Hierarchical F1 Metric (Day 0)**
   - ✅ Created `dutch_aware_hierarchical_f1()` metric returning (score, feedback) tuple
   - ✅ Implemented 3-component weighted scoring:
     * 70%: Hierarchical classification (is_greenhouse + uses_growlight)
     * 15%: Dutch terminology detection (16 Dutch terms)
     * 15%: Evidence quality (tier-based scoring)
   - ✅ Comprehensive feedback generation for GEPA reflection:
     * Misclassification details with expected vs. predicted
     * Missing Dutch terminology with search suggestions
     * Evidence quality warnings
     * Confidence-evidence alignment checks
   - ✅ Created `dutch_aware_f1_score_only()` wrapper for BootstrapFewShot/MIPROv2
   - ✅ Added constants: DUTCH_TERMS set, SOURCE_TIER_SCORES, thresholds
   - ✅ Comprehensive test coverage (27 tests, all passing)
   - ✅ All code quality checks pass (ruff format, ruff check, mypy clean)
   - ✅ Zero regressions (existing tests still pass)
   - ✅ Ready for Phase 4 optimizer integration

7. **Phase 4: GEPA/MIPROv2 Optimizer Configuration (Day 0)**
   - ✅ Added optimizer import hierarchy with fallback (GEPA → MIPROv2 → BootstrapFewShot)
   - ✅ Updated metric imports to use Phase 3 Dutch-aware metrics
   - ✅ Created `configure_gemini_teacher()` function for Gemini 2.5 Pro teacher model
   - ✅ Created `configure_optimizer()` function with recommended parameters:
     * GEPA: 15 generations, population 8, mutation 0.5, 5-fold CV
     * MIPROv2: auto="medium", 10 candidates, 4+6 demos
     * BootstrapFewShot: 5+10 demos (fallback)
   - ✅ Updated `phase2_optimization()` to use teacher-student model configuration
   - ✅ Enhanced console output showing optimizer type, teacher model, student model
   - ✅ Updated markdown report to include optimizer configuration details
   - ✅ All code quality checks pass (ruff format, ruff check, mypy clean)
   - ✅ Verified GEPA imports successfully from DSPy GitHub installation
   - ✅ **Active Optimizer: GEPA** (confirmed via import test)
   - ✅ Ready for Phase 1 data expansion (Steps 2-4)

### 🔄 In Progress

*Nothing currently in progress*

### 📋 Pending

1. **Phase 1: Fix DSPy + Data Expansion (Days 1-3)**
   - ⏳ Step 2: Manual prototyping and HITL validation with 10-15 companies
   - ⏳ Step 3: Build CLI tool for batch research
   - ⏳ Step 4: Scaled collection + HITL validation (60-65 examples)

2. **Phase 5-6: Cross-Validation & Deployment (Days 7-11)**
   - ⏳ Phase 5: Repeated nested cross-validation
   - ⏳ Phase 6: Production deployment

### 📊 Success Metrics Tracking

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Training F1 | 89.13% | N/A | ⚠️ Overfitting |
| Validation F1 | 62.50% | 95-98% | ❌ Below target |
| Overfitting Gap | 26.63% | <10% | ❌ Severe |
| Dataset Size | 23 examples | 60-65 | ❌ Insufficient |
| Dutch Sources | Unknown | ≥70% | ⏳ Pending |
| Tier-2 Evidence | Unknown | ≥80% | ⏳ Pending |

---

## Critical Finding: DSPy 3.0.3 is Broken 🚨

### Problem Identified

The DSPy 3.0.3 package from PyPI is **completely non-functional**:

```python
# ALL of these fail:
from dspy.teleprompt import BootstrapFewShot  # ImportError
from dspy.teleprompt import MIPROv2           # ImportError
from dspy.teleprompt import GEPA               # ImportError

import dspy
print(dir(dspy))  # Returns: [] (empty!)
```

**Root Cause:** Missing `__init__.py` files throughout the package, broken import paths, and incomplete module structure.

### Solution

```bash
# Remove broken PyPI version
uv pip uninstall dspy-ai

# Install working version from GitHub
uv pip install "git+https://github.com/stanfordnlp/dspy.git"
```

**Status:** This is a BLOCKER - must be fixed before any optimization work.

---

## Current State Analysis

### Existing Architecture

**Current System:**
- **Signature:** `GreenhouseDetector` with Perplexity Sonar web search
- **Optimizer:** BootstrapFewShot (but broken - can't import!)
- **Dataset:** 23 training examples, 4 validation examples
- **Metrics:** `greenhouse_f1_metric`, `growlight_accuracy_metric`, `combined_f1_metric`

**Problems Identified by Expert Analysis:**

1. **Severe Overfitting (26.6% gap)**
   - Model memorizes 23 training examples
   - Fails to generalize to new companies
   - 4-example validation set too small for reliable metrics

2. **Generic Signature**
   - No guidance for Dutch terminology
   - Doesn't prioritize Dutch sources (Groenten&Fruit, Floraldaily NL)
   - Missing hierarchical classification logic

3. **Data Quality Issues**
   - **schenkeveld** - Location precision error
   - **ubink** - Company identity/name change
   - **HilverdaFlorist** - Incorrect growlight classification

4. **Missing Identity Resolution**
   - No handling of company aliases/trade names
   - No temporal tracking (acquisitions, closures)
   - Exact location matching too rigid

5. **Weak Validation Strategy**
   - Single 4-example holdout → high variance
   - No cross-validation → unreliable metrics
   - Train/val split not stratified

---

## The 6-Phase Optimization Roadmap

### Phase 1: Fix DSPy + Data Expansion (Days 1-3)

#### 1.1 Install Working DSPy

```bash
uv pip uninstall dspy-ai
uv pip install "git+https://github.com/stanfordnlp/dspy.git"

# Verify installation
uv run python -c "from dspy.teleprompt import BootstrapFewShot, MIPROv2; print('✓ DSPy working')"
```

#### 1.2 Expand Dataset with Real Examples

**Current:** 23 total examples (19 train + 4 val)
**Target:** 60-65 real examples

**Positive Examples to Add:**
- Kwekerij Overgaag, Maasland ✓
- Porta Nova, Waddinxveen ✓
- BM Roses, Maasland ✓
- Diamond Flowers, Zuilichem ✓
- Batist Westmade, Made ✓
- Vereijken Kwekerijen, 's Gravenzande ✓
- Pannekoek Orchideeën, Berkel en Rodenrijs ✓
- Marjoland, Waddinxveen ✓
- Zentoo - MG GRAND B.V., Monster ✓
- Vreugdenhil Bulbs & Plants, Naaldwijk ✓
- Kwekerij De Opstal V.O.F., Naaldwijk ✓
- Kwekerij Vicini Herenwerf, Maasland ✓
- Tomatenkwekerij A. de Bruijn en Zn., Maasland ✓

**Negative Examples to Add:**
- P.J.J.M. Verbeek en P.H.M. Verbeek, Maasland ✓
- Kwekerij Ted Vijverberg B.V., De Lier ✓
- Kwekerij Figaro B.V., Naaldwijk ✓
- Fachjan Project Plants, Honselersdijk ✓
- Van Onselen Aubergines B.V., 's Gravenzande ✓
- Van der Sar Plants, 's Gravenzande ✓

**Additional Sources for Real Examples:**
- **Glastuinbouw Nederland** directory
- **Kas als Energiebron** participants
- Companies explicitly using "**onbelichte teelt**" (unlit cultivation)
- **Summer-only crop growers** (no winter lighting)
- **Dutch trade media** mentions (Groenten&Fruit, Floraldaily NL)
- **Supplier case studies** (Signify/Philips, Hortilux, Priva)

**Target:** Add 35-40 more real examples to reach 60-65 total

#### 1.3 Fix Data Quality Issues

**Specific Fixes:**

```python
# data/flagged_issues.json
{
  "schenkeveld": {
    "issue": "location_precision_error",
    "fix": "Verify exact address and geocode",
    "status": "pending"
  },
  "ubink": {
    "issue": "company_identity_change",
    "fix": "Check for acquisitions/name changes since labeling",
    "status": "pending"
  },
  "HilverdaFlorist": {
    "issue": "incorrect_growlight_classification",
    "fix": "Re-verify with current web sources",
    "status": "pending"
  }
}
```

#### 1.4 Company Identity Resolution

**New File:** `src/sevenrad_ee/ai/company_identity.py`

```python
"""Company identity resolution and canonicalization."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CompanyIdentity:
    """Canonical company identity."""

    canonical_name: str
    aliases: list[str]
    trade_names: list[str]
    locations: list[dict[str, str]]  # [{"address": "...", "geocode": "..."}]
    active_since: Optional[str] = None
    acquired_by: Optional[str] = None
    notes: str = ""


class IdentityResolver:
    """Resolve company names to canonical identities."""

    def __init__(self, identity_db_path: str):
        """Load identity database."""
        self.identities: dict[str, CompanyIdentity] = {}
        # Load from JSON database

    def resolve(self, company_name: str, location: str) -> Optional[CompanyIdentity]:
        """
        Resolve company name and location to canonical identity.

        Handles:
        - Aliases and trade names
        - Historical name changes
        - Multiple facility locations
        - Acquisitions and mergers
        """
        # Implementation
        pass
```

**Data File:** `data/company_identities.json`

```json
{
  "porta_nova": {
    "canonical_name": "Porta Nova",
    "aliases": ["Porta Nova B.V.", "PortaNova"],
    "trade_names": ["Porta Nova Roses"],
    "locations": [
      {"address": "Waddinxveen, Zuid-Holland", "geocode": "52.04,4.65"}
    ],
    "active_since": "1987",
    "notes": "Premium rose grower, uses LED assimilation lighting"
  }
}
```

---

## Phase 1 Deep-Dive: Perplexity Sonar-Pro Data Collection Strategy

**Expert Analysis by:** Gemini-2.5-Pro
**Purpose:** Systematic, high-quality data expansion from 23 → 60-65 real examples
**Timeline:** 2-3 days for 40+ examples with production-ready automation

### Overview

Data scarcity (23 examples) is the **root cause** of the 26.6% overfitting gap. This deep-dive provides a production-ready strategy for using Perplexity Sonar-Pro API to efficiently collect high-quality Dutch greenhouse examples with verifiable evidence of lighting usage.

**Key Principles:**
1. **Quality over quantity** - Tier-2 evidence (company websites, supplier case studies) only
2. **Dutch-first** - Prioritize Dutch terminology and sources
3. **Caching for efficiency** - SHA-256 cached queries prevent redundant API calls
4. **Systematic discovery** - Multi-query strategy per company
5. **Human-in-the-loop** - Automated categorization with manual validation

---

### A) API Configuration for Maximum Quality

#### Model Selection

**Primary Model:** `pplx-70b-online` (Sonar Pro)
- Optimized for factual, search-grounded responses
- Fast and cost-effective for batch processing
- Deterministic output with temperature=0

**Fallback Model:** `sonar-reasoning`
- Reserve for ambiguous cases only
- Slower and more expensive
- Use when first-pass results are unclear

#### Critical Parameters

```python
PERPLEXITY_API_CONFIG = {
    "model": "pplx-70b-online",
    "temperature": 0.0,  # CRITICAL: Deterministic, factual responses
    "return_citations": True,  # NON-NEGOTIABLE: Need source URLs

    # Domain filtering - curated Dutch sources
    "search_domain_filter": [
        # Tier 1: Dutch Trade Media (highest signal)
        "groentennieuws.nl",
        "agf.nl",
        "floraldaily.nl",
        "nieuweoogst.nl",
        "gfactueel.nl",

        # Tier 2: Knowledge Hubs & Suppliers
        "wur.nl",  # Wageningen University (research authority)
        "signify.com/nl-nl",  # LED supplier case studies
        "hortilux.com",  # Lighting supplier
        "priva.com",  # Climate control supplier
    ],

    # Recency filter - LED tech changes rapidly
    "search_recency_filter": "last_3_years",

    # Response constraints
    "max_tokens": 1000,
    "top_p": 1.0,
}
```

**Why These Domains:**
- **Trade media** (groentennieuws.nl, agf.nl): Industry news, installations, investments
- **WUR** (wur.nl): Research publications, technical validation
- **Suppliers** (signify, hortilux, priva): Case studies naming specific customers

---

### B) Multi-Query Strategy

#### For Known Companies (3-Query Sequence)

**Query 1: Broad Positive Search**
```
"[Company Name]" AND (assimilatiebelichting OR groeilicht OR "belichte teelt")
```
*Goal:* Direct evidence of lighting usage

**Query 2: Supplier-Association Search**
```
"[Company Name]" AND (Signify OR Hortilux OR "Philips LED" OR Gavita)
```
*Goal:* Indirect evidence through supplier mentions (case studies, installations)

**Query 3: Negative Signal Search**
```
"[Company Name]" AND ("onbelichte teelt" OR "biologische teelt zonder kunstlicht" OR "daglichtkas")
```
*Goal:* Explicit evidence of NOT using lights (strong negative signal)

**Example for "Porta Nova, Waddinxveen":**

```python
queries = [
    '"Porta Nova" Waddinxveen AND (assimilatiebelichting OR groeilicht)',
    '"Porta Nova" AND (Signify OR Hortilux OR "Philips LED")',
    '"Porta Nova" AND ("onbelichte teelt" OR "daglichtkas")',
]
```

#### For Discovery (Finding New Companies)

**Strategy 1: Region + Crop Queries**
```
lijst van tomatenkwekerijen met LED-belichting in Westland
nieuwe kassen met SON-T assimilatiebelichting voor rozen
belichte teelt orchideeën Nederland
```
*Goal:* Discover clusters by geography + crop type

**Strategy 2: Supplier-Driven Discovery**
```
case study "nieuwe LED installatie" kwekerij site:signify.com/nl-nl
klantenlijst glastuinbouw site:priva.com
Hortilux projecten Nederland rozen
```
*Goal:* Leverage supplier marketing to find customers

**Strategy 3: Trade News Monitoring**
```
"investeert in belichting" OR "schakelt over op LED" site:groentennieuws.nl
"nieuwe assimilatiebelichting" site:agf.nl
LED-belichting kwekerij 2024 site:floraldaily.nl
```
*Goal:* Find news announcements of new installations

---

### C) Expanded Dutch Terminology Reference

#### Primary Positive Terms (Core Evidence)
- `assimilatiebelichting` - Assimilation lighting (primary term)
- `assimilatieverlichting` - Alternative spelling
- `groeilicht` - Grow light
- `belichte teelt` - Lighted cultivation
- `kunstlicht` - Artificial light

#### Secondary Positive Terms (Tech-Specific)
- `LED-belichting` - LED lighting (modern)
- `SON-T` - High-pressure sodium lamps (legacy)
- `hybride belichting` - **NEW:** Hybrid (LED + SON-T mix)
- `vertical farming` - **NEW:** Always implies artificial light
- `assimilatielampen` - Assimilation lamps

#### Negative Indicator Terms
- `onbelichte teelt` - **PRIMARY:** Unlit cultivation (explicit negative)
- `zonder kunstlicht` - Without artificial light
- `daglichtkas` - Daylight greenhouse only
- `biologische teelt` - ⚠️ **AMBIGUOUS:** Some organic growers DO use lights

#### Key Supplier Names (High-Signal)
- `Signify` (formerly Philips Lighting)
- `Hortilux` / `Philips Hortilux`
- `Fluence` - **NEW:** LED specialist
- `Gavita` - **NEW:** HPS/LED supplier
- `Oreon` - **NEW:** Dutch LED supplier
- `Priva` - Climate control (indirect signal)
- `Hoogendoorn` - Automation (indirect signal)

#### Job Posting Terms (Goldmine for Evidence)
```
vacature "specialist belichting" [Company Name]
vacature "medewerker gewasverzorging belichte teelt"
functie "teeltspecialist" assimilatiebelichting
```
*Why valuable:* Job postings prove active lighting operations

---

### D) Automated Workflow with Caching

#### Script Architecture

```python
"""
Perplexity-based greenhouse data collector with SHA-256 caching.

Purpose: Systematically search for 40+ real Dutch greenhouse examples
Timeline: 2-3 days with manual validation
Cost: <$50 in API calls (with caching)
"""

import hashlib
import json
from pathlib import Path
from typing import Literal
import requests
import time

# Configuration
CACHE_DIR = Path("cache/perplexity_searches")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("data/research")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

# API configuration (from Section A)
API_CONFIG = {
    "model": "pplx-70b-online",
    "temperature": 0.0,
    "return_citations": True,
    "search_domain_filter": [
        "groentennieuws.nl", "agf.nl", "floraldaily.nl",
        "nieuweoogst.nl", "gfactueel.nl", "wur.nl",
        "signify.com/nl-nl", "hortilux.com", "priva.com"
    ],
    "search_recency_filter": "last_3_years",
    "max_tokens": 1000,
}


def query_with_cache(query: str, config: dict = None) -> dict:
    """
    Execute Perplexity query with local SHA-256 caching.

    Caching prevents redundant API calls and saves significant cost
    when iterating on data collection strategy.

    Args:
        query: Search query string
        config: API configuration (default: API_CONFIG)

    Returns:
        Perplexity API response dict
    """
    if config is None:
        config = API_CONFIG

    # Generate cache key: SHA-256(query + config)
    cache_input = query + json.dumps(config, sort_keys=True)
    cache_key = hashlib.sha256(cache_input.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"

    # Check cache
    if cache_file.exists():
        print(f"  [CACHE HIT] {query[:60]}...")
        return json.loads(cache_file.read_text())

    # Execute API call
    print(f"  [API CALL] {query[:60]}...")

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        **config,
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    }

    try:
        response = requests.post(
            PERPLEXITY_API_URL,
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        # Save to cache
        cache_file.write_text(json.dumps(result, indent=2))

        # Rate limiting courtesy delay
        time.sleep(1)

        return result

    except Exception as e:
        print(f"  [ERROR] {e}")
        return {"error": str(e)}


def research_company(company_name: str, location: str) -> dict:
    """
    Research a single company using 3-query strategy.

    Returns structured results for human review.
    """
    results = {
        "company": company_name,
        "location": location,
        "timestamp": datetime.now().isoformat(),
        "queries": [],
        "evidence": {
            "positive": [],
            "negative": [],
            "ambiguous": [],
        },
        "classification_suggestion": None,
    }

    # Query 1: Positive signals
    q1 = f'"{company_name}" {location} AND (assimilatiebelichting OR groeilicht OR "belichte teelt")'
    r1 = query_with_cache(q1)
    results["queries"].append({"query": q1, "response": r1})

    # Parse citations and categorize
    if "citations" in r1:
        for citation in r1["citations"]:
            # Check for positive terms
            if any(term in citation.get("text", "").lower()
                   for term in ["assimilatie", "groeilicht", "led-belichting"]):
                results["evidence"]["positive"].append({
                    "url": citation["url"],
                    "snippet": citation["text"],
                    "source": "query_1_positive"
                })

    # Query 2: Supplier associations
    q2 = f'"{company_name}" AND (Signify OR Hortilux OR "Philips LED" OR Gavita)'
    r2 = query_with_cache(q2)
    results["queries"].append({"query": q2, "response": r2})

    if "citations" in r2:
        for citation in r2["citations"]:
            results["evidence"]["positive"].append({
                "url": citation["url"],
                "snippet": citation["text"],
                "source": "query_2_supplier"
            })

    # Query 3: Negative signals
    q3 = f'"{company_name}" AND ("onbelichte teelt" OR "daglichtkas")'
    r3 = query_with_cache(q3)
    results["queries"].append({"query": q3, "response": r3})

    if "citations" in r3:
        for citation in r3["citations"]:
            if any(term in citation.get("text", "").lower()
                   for term in ["onbelicht", "daglichtkas", "zonder kunstlicht"]):
                results["evidence"]["negative"].append({
                    "url": citation["url"],
                    "snippet": citation["text"],
                    "source": "query_3_negative"
                })

    # Suggest classification for human review
    pos_count = len(results["evidence"]["positive"])
    neg_count = len(results["evidence"]["negative"])

    if pos_count >= 2 and neg_count == 0:
        results["classification_suggestion"] = "POSITIVE (uses growlights)"
    elif neg_count >= 1 and pos_count == 0:
        results["classification_suggestion"] = "NEGATIVE (no growlights)"
    else:
        results["classification_suggestion"] = "NEEDS_MANUAL_REVIEW"

    return results


# Main execution
if __name__ == "__main__":
    companies_to_research = [
        ("Kwekerij Overgaag", "Maasland"),
        ("Porta Nova", "Waddinxveen"),
        ("BM Roses", "Maasland"),
        # ... add all 19 provided companies
    ]

    for company_name, location in companies_to_research:
        print(f"\n[RESEARCHING] {company_name}, {location}")

        results = research_company(company_name, location)

        # Save results
        filename = f"{company_name.replace(' ', '_')}_{location}.json"
        output_path = RESULTS_DIR / filename
        output_path.write_text(json.dumps(results, indent=2))

        print(f"  [SAVED] {output_path}")
        print(f"  [SUGGESTION] {results['classification_suggestion']}")
```

#### Expected Output Structure

```json
{
  "company": "Porta Nova",
  "location": "Waddinxveen",
  "timestamp": "2025-10-25T14:30:00",
  "queries": [
    {
      "query": "\"Porta Nova\" Waddinxveen AND assimilatiebelichting",
      "response": { "..." }
    }
  ],
  "evidence": {
    "positive": [
      {
        "url": "https://signify.com/nl-nl/case-study/porta-nova",
        "snippet": "Porta Nova installeert 5000 Philips GreenPower LED modules",
        "source": "query_2_supplier"
      }
    ],
    "negative": [],
    "ambiguous": []
  },
  "classification_suggestion": "POSITIVE (uses growlights)"
}
```

---

### E) Evidence Quality Validation Rubric

#### Tier 2: HIGH CONFIDENCE (Accept for Training Data)

**Company Website Evidence**
- ✅ Direct statement on official company domain
- ✅ Example: "Onze kwekerij gebruikt LED-assimilatiebelichting van Signify"
- ✅ Verify: Check domain ownership, not subdomain/blog

**Supplier Case Studies**
- ✅ Formal case study from lighting manufacturer
- ✅ Example: Signify, Hortilux, Gavita case study page
- ✅ Must name specific company and installation details

**Job Postings**
- ✅ Vacancy for lighting specialist/technician
- ✅ Example: "Vacature: Specialist Assimilatiebelichting bij [Company]"
- ✅ Sources: Indeed.nl, LinkedIn, company careers page

**This is the gold standard for training data.**

#### Tier 1: MEDIUM-HIGH CONFIDENCE (Review Carefully)

**Trade Media Articles**
- ⚠️ News article in reputable trade journal
- ⚠️ Must explicitly state lighting usage (not just "modern facility")
- ⚠️ Sources: Groenten&Fruit, Floraldaily NL, Nieuweoogst, AGF
- ⚠️ **Verify publication date** (within 3 years for LED claims)

**Example acceptable article:**
> "Kwekerij X investeert €2 miljoen in nieuwe LED-assimilatiebelichting"
> — Groenten & Fruit, 15 maart 2024

**Example unacceptable article:**
> "Kwekerij X opent ultramoderne kas" (too vague, no lighting specifics)

#### REJECT: UNRELIABLE (Do Not Use)

**Forum Posts & Comments**
- ❌ User-generated content
- ❌ Unverified claims
- ❌ Example: Tuinbouwforum discussions

**Outdated Information**
- ❌ Articles older than 5 years
- ❌ LED technology landscape changed significantly
- ❌ Company may have switched technologies

**Vague Marketing Language**
- ❌ "State-of-the-art facility"
- ❌ "Modern technology"
- ❌ "Sustainable operations"
- ❌ Must explicitly mention lighting technology

**Snippet-Only Evidence**
- ❌ Perplexity snippet without clicking through to source
- ❌ **ALWAYS verify by reading the full source article**

---

### F) Manual Validation Process

#### Step 1: Automated Categorization

Script outputs three categories:
1. **POSITIVE** - ≥2 positive evidence pieces, 0 negative
2. **NEGATIVE** - ≥1 negative evidence, 0 positive
3. **NEEDS_MANUAL_REVIEW** - Conflicting or insufficient evidence

#### Step 2: Human Review Protocol

For each company marked POSITIVE or NEGATIVE:

1. **Click through to ALL citation URLs**
   - Do not trust snippet alone
   - Read the full article/page
   - Verify context is correct

2. **Apply Evidence Quality Rubric**
   - Does this meet Tier 1 or Tier 2 standards?
   - Is the source authoritative?
   - Is the information current (within 3 years)?

3. **Document Decision**
   ```json
   {
     "company": "Porta Nova",
     "classification": "POSITIVE",
     "confidence": "HIGH",
     "evidence_tier": 2,
     "sources": [
       {
         "url": "https://signify.com/nl-nl/case-study/porta-nova",
         "tier": "supplier_case_study",
         "quote": "Porta Nova installeert 5000 Philips GreenPower LED modules voor 12 hectare rozen",
         "verified": true,
         "date": "2023-08-15"
       }
     ],
     "notes": "Supplier case study with specific installation details. High confidence."
   }
   ```

4. **Handle Edge Cases**
   - **Company name ambiguity:** Verify it's the correct company
   - **Temporal changes:** Use most recent evidence
   - **Mixed evidence:** Prefer Tier-2 over Tier-1 sources

#### Step 3: Quality Threshold

**Minimum Standards for Inclusion:**
- At least ONE Tier-2 source OR
- TWO Tier-1 sources with consistent information

**If Below Threshold:**
- Mark as NEEDS_MORE_RESEARCH
- Add to follow-up list
- Do not include in training data yet

---

### G) Expected Timeline & Milestones

#### Day 1: Setup & Initial Research (8 hours)

**Morning (4 hours):**
- Install dependencies: `uv pip install requests`
- Set up directory structure (`cache/`, `data/research/`)
- Configure `PERPLEXITY_API_KEY`
- Test API with 3 known companies
- Verify caching works correctly

**Afternoon (4 hours):**
- Run automated search for 19 provided companies
- Expected: ~60 API calls (3 queries × 19 companies + discovery)
- Cost: ~$10-15 (with caching, subsequent runs free)
- Review automated categorizations

**Output:** 19 company research files with initial classifications

#### Day 2: Discovery & Expansion (8 hours)

**Morning (4 hours):**
- Run discovery queries (regions, suppliers, trade news)
- Identify 30-40 new candidate companies
- Research new candidates with 3-query strategy
- Expected: ~100 API calls (new queries only)

**Afternoon (4 hours):**
- Manual review of all positive classifications
- Click through to verify Tier-1/Tier-2 evidence
- Document decisions in structured format
- Identify companies needing additional research

**Output:** 40-50 research files total, 25-30 with solid classification

#### Day 3: Validation & Finalization (6 hours)

**Morning (3 hours):**
- Manual review of negative classifications
- Research NEEDS_MANUAL_REVIEW cases
- Fill gaps to reach 60-65 total examples
- Cross-check for duplicates (aliases, multiple locations)

**Afternoon (3 hours):**
- Format final dataset: `data/dutch_companies_expanded.json`
- Create evidence summary report
- Verify class balance (aim for 40+ positive, 20+ negative)
- Document data quality issues for Phase 1.3

**Output:** `dutch_companies_expanded.json` with 60-65 high-quality, verified examples

---

### H) Cost & Efficiency Analysis

#### API Cost Estimates

**With Caching (Recommended):**
- Initial run: ~160 API calls (19 known + 40 new × 3 queries + 10 discovery)
- Cost: $20-30 (depending on response length)
- Subsequent iterations: **FREE** (cached)

**Without Caching:**
- Each iteration: ~160 API calls
- Cost per iteration: $20-30
- **Total waste over 3 days: $60-90**

**Caching ROI:** Saves 66-75% on API costs during iteration

#### Time Efficiency

| Task | Manual (No Tool) | With Script | Savings |
|------|-----------------|-------------|---------|
| Search 1 company | 15 min | 2 min | 87% |
| Search 19 companies | 4.75 hours | 40 min | 86% |
| Search 60 companies | 15 hours | 2 hours | 87% |

**Total Time Savings:** ~13 hours over 3 days

---

### I) Success Metrics for Phase 1

#### Quantitative Targets

✅ **Dataset Size: 60-65 real examples**
- 40+ positive (uses growlights)
- 20+ negative (no growlights or explicitly unlit)

✅ **Evidence Quality: ≥80% Tier-2 sources**
- Company websites, supplier case studies, job postings
- Tier-1 acceptable only with corroborating evidence

✅ **Dutch Source Coverage: ≥70%**
- Majority of evidence from Dutch-language sources
- groentennieuws.nl, agf.nl, floraldaily.nl, signify.com/nl-nl

✅ **Class Balance: 60-70% positive**
- Mirrors real-world distribution
- Prevents minority class issues in training

#### Qualitative Checks

✅ **No Duplicates**
- Companies deduplicated by canonical name
- Aliases/trade names documented in company_identities.json

✅ **Geographic Diversity**
- Not all from Westland region
- Include companies from multiple provinces

✅ **Crop Diversity**
- Roses, tomatoes, orchids, gerbera (common lighting users)
- Cucumbers, peppers, lettuce (sometimes use lighting)

✅ **Temporal Currency**
- All evidence from last 3 years
- LED technology context (not outdated SON-T only)

---

### J) Integration with Phases 2-6

#### How This Data Feeds Downstream

**Phase 2 (Hierarchical Signature):**
- Evidence structure informs signature design
- Dutch terms list validates terminology coverage
- Source tiers guide evidence_sources field

**Phase 3 (Dutch-Aware Metric):**
- Tier-2 threshold (80%) becomes metric component
- Dutch term frequency validates search strategy
- Evidence quality scores calibration

**Phase 4 (GEPA Optimizer):**
- High-quality data prevents garbage-in-garbage-out
- Diverse examples enable better generalization
- Evidence sources allow reflection model to learn

**Phase 5 (Cross-Validation):**
- 60-65 examples enable 5-fold CV with 12-13 per fold
- Stratified splitting maintains class balance
- Sufficient data to detect overfitting reliably

**Phase 6 (Production):**
- Search strategy becomes operational template
- Evidence rubric guides quality control
- Caching infrastructure scales to production

---

### K) Troubleshooting Common Issues

#### Issue 1: Too Few Results

**Symptoms:**
- Query returns 0-1 citations
- Perplexity says "no results found"

**Solutions:**
1. Broaden search: Remove restrictive domain filter temporarily
2. Try alternative company name spellings
3. Search for location + crop type instead of company name
4. Use discovery queries to find related companies first

#### Issue 2: Low-Quality Results

**Symptoms:**
- Only general web sources
- Old articles (>5 years)
- No Dutch-language results

**Solutions:**
1. Add `site:` operators for specific domains
2. Strengthen Dutch terminology in query
3. Adjust `search_recency_filter` to `last_year`
4. Manual search on groentennieuws.nl directly

#### Issue 3: Conflicting Evidence

**Symptoms:**
- Company appears in both positive and negative results
- Temporal changes (switched from SON-T to LED)
- Multiple facilities with different technologies

**Solutions:**
1. Prioritize most recent evidence (within 1 year)
2. Mark as NEEDS_MANUAL_REVIEW
3. Check if evidence refers to different facilities
4. Use date-specific queries: `"[Company]" LED 2024`

#### Issue 4: Rate Limiting

**Symptoms:**
- HTTP 429 errors
- Slow response times

**Solutions:**
1. Add `time.sleep(2)` between requests
2. Batch processing with checkpoints
3. Use caching to avoid re-running queries
4. Spread research over multiple days

---

### L) Key Takeaways

**Critical Success Factors:**
1. **Caching is mandatory** - SHA-256 prevents redundant API calls
2. **Human validation required** - Script categorizes, humans decide
3. **Tier-2 evidence only** - Company sites, supplier case studies, job postings
4. **Dutch-first queries** - Terminology and domain filtering critical
5. **3-query strategy** - Positive + supplier + negative for completeness

**What Makes This Approach Production-Ready:**
- ✅ Systematic, repeatable process
- ✅ Cost-efficient with caching (~$30 vs $90)
- ✅ Quality-focused with evidence rubric
- ✅ Time-efficient (2-3 days vs 2 weeks manual)
- ✅ Scales to 100+ companies if needed

**Expected Outcome:**
- 60-65 high-quality, verified Dutch greenhouse examples
- 80%+ with Tier-2 evidence (company sites, case studies)
- 70%+ with Dutch-language sources
- Ready for Phase 2 hierarchical signature implementation

---

**Extended by:** Gemini-2.5-Pro expert analysis
**Attribution:** Production-ready implementation guidance via Zen MCP

---

### Phase 2: Hierarchical Signature with Dutch Guidance (Days 4-5)

#### 2.1 Enhanced Signature Design

**File:** `src/sevenrad_ee/ai/dspy_greenhouse.py` (updated)

```python
"""DSPy greenhouse detection with hierarchical classification."""

import dspy
from typing import Literal
from pydantic import BaseModel, Field


class GreenhouseClassification(dspy.Signature):
    """
    Classify Dutch greenhouse companies using Perplexity Sonar web search.

    CRITICAL: This is a web-search-only classifier. Use Perplexity Sonar to find
    evidence from Dutch horticultural sources.

    SEARCH STRATEGY (Dutch-First):
    1. Start with Dutch company websites and trade media
    2. Look for Dutch terminology:
       - Greenhouse: "kwekerij", "glastuinbouw", "teler", "kassen"
       - Lighting: "assimilatiebelichting", "assimilatieverlichting", "kunstlicht",
                  "belichte teelt", "LED-belichting", "SON-T lampen"
       - Suppliers: "Signify", "Philips Hortilux", "Priva", "Hoogendoorn"
    3. Prioritize these source types:
       - Tier 2 (highest): Company websites, supplier case studies, job postings
       - Tier 1: Dutch trade media (Groenten&Fruit, Floraldaily NL, KAS Magazine)
       - Tier 0.5: General web sources
    4. Check for negative indicators:
       - "onbelichte teelt" (unlit cultivation)
       - "daglichtkas" (daylight greenhouse only)
       - Summer-only crops with no winter operation

    HIERARCHICAL LOGIC:
    - If is_greenhouse = NO → uses_growlight MUST be UNKNOWN
    - If is_greenhouse = YES → determine uses_growlight based on evidence
    - If evidence insufficient → uses_growlight = UNKNOWN (prefer precision)
    - Species provides validation (roses/tomatoes often use lighting)
    """

    # Inputs
    location_name: str = dspy.InputField(desc="Company name and location to classify")
    location_area: str = dspy.InputField(desc="Geographic area (city, region)")

    # Hierarchical outputs with gating
    is_greenhouse: Literal["YES", "NO"] = dspy.OutputField(
        desc="Is this a greenhouse/kwekerij/nursery? Use web search to verify."
    )

    uses_growlight: Literal["YES", "NO", "UNKNOWN"] = dspy.OutputField(
        desc=(
            "Does this greenhouse use artificial lighting (assimilatiebelichting)? "
            "ONLY answer YES if greenhouse=YES AND you have evidence. "
            "Answer UNKNOWN if insufficient evidence. "
            "Answer NO only with explicit negative evidence (e.g., 'onbelichte teelt')."
        )
    )

    species_grown: list[str] = dspy.OutputField(
        desc=(
            "List of species/crops grown (e.g., ['roses', 'tomatoes']). "
            "Empty list if unknown. Species helps validate lighting usage: "
            "roses/orchids/gerbera often require lighting; lettuce/herbs sometimes; "
            "cucumbers/peppers less common."
        )
    )

    # Evidence capture (critical for validation)
    dutch_terms_found: list[str] = dspy.OutputField(
        desc=(
            "Dutch horticultural terms found in sources. "
            "Examples: 'assimilatiebelichting', 'kwekerij', 'belichte teelt', "
            "'SON-T', 'LED-belichting'. More Dutch terms = higher confidence."
        )
    )

    evidence_sources: list[dict] = dspy.OutputField(
        desc=(
            "List of source URLs with quotes and tier classification. "
            "Format: [{'url': '...', 'quote': '...', 'tier': 'company_website'}, ...]. "
            "Tiers: company_website, supplier_case_study, job_posting (tier 2); "
            "trade_media_nl (tier 1); general_web (tier 0.5)."
        )
    )

    confidence: float = dspy.OutputField(
        desc=(
            "Classification confidence 0.0-1.0. "
            "High confidence (>0.8) requires: Dutch sources + Dutch terminology + tier 2 evidence. "
            "Medium (0.5-0.8): Some Dutch evidence or tier 1 sources. "
            "Low (<0.5): Only general web sources or ambiguous evidence."
        )
    )

    rationale: str = dspy.OutputField(
        desc=(
            "Brief explanation of classification decision. "
            "Mention: key evidence found, source quality, Dutch terminology, "
            "and any uncertainty factors."
        )
    )


# Validation rules
class GreenhouseClassificationValidator:
    """Validate hierarchical classification logic."""

    @staticmethod
    def validate(prediction: "GreenhouseClassification") -> tuple[bool, str]:
        """
        Validate prediction follows hierarchical rules.

        Returns:
            (is_valid, error_message)
        """
        # Rule 1: If not greenhouse, growlight must be UNKNOWN
        if prediction.is_greenhouse == "NO" and prediction.uses_growlight != "UNKNOWN":
            return False, "uses_growlight must be UNKNOWN when is_greenhouse=NO"

        # Rule 2: If uses_growlight=YES, must have evidence
        if prediction.uses_growlight == "YES" and not prediction.evidence_sources:
            return False, "uses_growlight=YES requires evidence_sources"

        # Rule 3: Confidence must match evidence quality
        tier2_count = sum(
            1 for s in prediction.evidence_sources
            if s.get("tier") in ["company_website", "supplier_case_study", "job_posting"]
        )
        if prediction.confidence > 0.8 and tier2_count == 0:
            return False, "High confidence (>0.8) requires at least one tier-2 source"

        # Rule 4: Dutch terms should be present for high confidence
        if prediction.confidence > 0.8 and len(prediction.dutch_terms_found) == 0:
            return False, "High confidence requires Dutch terminology evidence"

        return True, ""
```

#### 2.2 Pydantic Output Schema

```python
from pydantic import BaseModel, Field, field_validator


class EvidenceSource(BaseModel):
    """Evidence source with URL, quote, and quality tier."""

    url: str = Field(..., description="Source URL")
    quote: str = Field(..., description="Relevant quote from source")
    tier: Literal["company_website", "supplier_case_study", "job_posting",
                  "trade_media_nl", "general_web"] = Field(..., description="Source quality tier")


class GreenhouseDetectionOutput(BaseModel):
    """Validated greenhouse detection output."""

    is_greenhouse: Literal["YES", "NO"]
    uses_growlight: Literal["YES", "NO", "UNKNOWN"]
    species_grown: list[str] = Field(default_factory=list)
    dutch_terms_found: list[str] = Field(default_factory=list)
    evidence_sources: list[EvidenceSource] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str

    @field_validator("uses_growlight")
    @classmethod
    def validate_hierarchical_logic(cls, v, info):
        """Enforce hierarchical gating."""
        if info.data.get("is_greenhouse") == "NO" and v != "UNKNOWN":
            raise ValueError("uses_growlight must be UNKNOWN when is_greenhouse=NO")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence_matches_evidence(cls, v, info):
        """Ensure confidence aligns with evidence quality."""
        sources = info.data.get("evidence_sources", [])
        tier2_count = sum(1 for s in sources if s.tier in [
            "company_website", "supplier_case_study", "job_posting"
        ])

        if v > 0.8 and tier2_count == 0:
            raise ValueError("High confidence (>0.8) requires tier-2 evidence")

        return v
```

---

### Phase 3: Dutch-Aware Hierarchical F1 Metric (Day 6)

**File:** `src/sevenrad_ee/ai/dspy_evaluation.py` (updated)

```python
"""DSPy evaluation metrics with Dutch-aware feedback for GEPA optimization."""

from typing import Optional
import dspy
from sklearn.metrics import f1_score


def dutch_aware_hierarchical_f1(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Optional[str] = None
) -> tuple[float, str]:
    """
    Hierarchical F1 metric with Dutch terminology weighting and textual feedback.

    This metric is designed for GEPA optimizer's reflection mechanism. It returns:
    1. A score (0.0-1.0) for optimization
    2. Textual feedback explaining successes/failures for reflection model

    Scoring Components:
    - 70%: Hierarchical classification accuracy (is_greenhouse + uses_growlight)
    - 15%: Dutch terminology detection (rewards finding Dutch terms)
    - 15%: Evidence quality (rewards tier-2 Dutch sources)

    Args:
        example: Ground truth example
        prediction: Model prediction
        trace: Optional execution trace (unused)

    Returns:
        (score, feedback) tuple for GEPA reflection
    """
    feedback_parts = []

    # Component 1: Hierarchical Classification (70% weight)
    # --------------------------------------------------------

    # Greenhouse classification
    gh_correct = (prediction.is_greenhouse == example.is_greenhouse)
    gh_score = 1.0 if gh_correct else 0.0

    if not gh_correct:
        feedback_parts.append(
            f"MISCLASSIFIED is_greenhouse: predicted {prediction.is_greenhouse}, "
            f"expected {example.is_greenhouse}"
        )

    # Growlight classification (only if greenhouse=YES in ground truth)
    if example.is_greenhouse == "YES":
        gl_correct = (prediction.uses_growlight == example.uses_growlight)
        gl_score = 1.0 if gl_correct else 0.0

        if not gl_correct:
            feedback_parts.append(
                f"MISCLASSIFIED uses_growlight: predicted {prediction.uses_growlight}, "
                f"expected {example.uses_growlight}"
            )

            # Specific Dutch term guidance
            company_lower = example.location_name.lower()
            pred_terms_lower = [t.lower() for t in prediction.dutch_terms_found]

            if "assimilatie" in company_lower:
                if not any("assimilatie" in t for t in pred_terms_lower):
                    feedback_parts.append(
                        "MISSED 'assimilatie' terminology in company name - "
                        "this is a strong signal for growlight usage"
                    )

            # Check for common Dutch grower surnames that often use lighting
            lighting_surnames = ["kwekerij", "teler", "rozen", "orchidee"]
            for surname in lighting_surnames:
                if surname in company_lower:
                    feedback_parts.append(
                        f"Company name contains '{surname}' - "
                        f"search for '{surname} + assimilatiebelichting' might help"
                    )
    else:
        # If not a greenhouse, growlight should be UNKNOWN
        gl_score = 1.0 if prediction.uses_growlight == "UNKNOWN" else 0.0

        if gl_score == 0.0:
            feedback_parts.append(
                f"LOGIC ERROR: is_greenhouse=NO but uses_growlight={prediction.uses_growlight} "
                f"(should be UNKNOWN)"
            )

    classification_score = (gh_score + gl_score) / 2

    # Component 2: Dutch Terminology Detection (15% weight)
    # ------------------------------------------------------

    DUTCH_TERMS = {
        'assimilatiebelichting', 'assimilatieverlichting', 'kunstlicht',
        'kassen', 'kwekerij', 'teler', 'glastuinbouw',
        'belichte', 'led-belichting', 'son-t', 'groeilicht',
        'teeltspecialist', 'assimilatielampen'
    }

    found_terms = set(t.lower() for t in prediction.dutch_terms_found)
    matching_terms = found_terms & DUTCH_TERMS

    dutch_score = len(matching_terms) / max(len(DUTCH_TERMS), 1)

    if len(matching_terms) == 0:
        feedback_parts.append(
            "NO Dutch terminology found - search strategy may be ineffective. "
            "Try searches like: '{company} assimilatiebelichting', "
            "'{company} kwekerij belichte teelt'"
        )
    elif len(matching_terms) >= 3:
        feedback_parts.append(
            f"GOOD: Found {len(matching_terms)} Dutch terms: {', '.join(list(matching_terms)[:5])}"
        )
    else:
        feedback_parts.append(
            f"Found {len(matching_terms)} Dutch terms - could find more with better queries"
        )

    # Component 3: Evidence Quality (15% weight)
    # -------------------------------------------

    SOURCE_TIER_SCORES = {
        'company_website': 2.0,
        'supplier_case_study': 2.0,
        'job_posting': 2.0,
        'trade_media_nl': 1.0,
        'general_web': 0.5
    }

    if not prediction.evidence_sources:
        evidence_score = 0.0
        feedback_parts.append(
            "NO evidence sources provided - predictions must be backed by URLs and quotes"
        )
    else:
        total_tier_score = sum(
            SOURCE_TIER_SCORES.get(source.get('tier', 'general_web'), 0.5)
            for source in prediction.evidence_sources
        )
        max_possible_score = len(prediction.evidence_sources) * 2.0
        evidence_score = total_tier_score / max_possible_score if max_possible_score > 0 else 0.0

        tier2_count = sum(
            1 for s in prediction.evidence_sources
            if s.get('tier') in ['company_website', 'supplier_case_study', 'job_posting']
        )

        if tier2_count == 0:
            feedback_parts.append(
                "LOW-QUALITY sources (no tier-2 evidence) - "
                "prioritize company websites, supplier case studies, or job postings"
        )
        else:
            feedback_parts.append(
                f"GOOD: {tier2_count} tier-2 sources found (high-quality evidence)"
            )

    # Weighted Total Score
    # --------------------

    total_score = (
        classification_score * 0.70 +
        dutch_score * 0.15 +
        evidence_score * 0.15
    )

    # Final Feedback Assembly
    # -----------------------

    if not feedback_parts:
        feedback = (
            f"✓ CORRECT classification | "
            f"Found {len(matching_terms)} Dutch terms | "
            f"{len(prediction.evidence_sources)} sources"
        )
    else:
        feedback = " | ".join(feedback_parts)

    # Add confidence check
    if prediction.confidence > 0.8 and tier2_count == 0:
        feedback += " | WARNING: High confidence without tier-2 evidence is unreliable"

    return total_score, feedback


# Backward compatibility wrapper for non-GEPA optimizers
def dutch_aware_f1_score_only(example: dspy.Example, prediction: dspy.Prediction) -> float:
    """Simple F1 score without feedback (for BootstrapFewShot compatibility)."""
    score, _ = dutch_aware_hierarchical_f1(example, prediction)
    return score
```

---

### Phase 4: GEPA Optimizer Configuration (Days 7-9)

**File:** `src/sevenrad_ee/operations/optimize_greenhouse_detection.py` (updated)

```python
"""DSPy greenhouse detection optimization with GEPA or MIPROv2."""

import dspy
from pathlib import Path

# Try GEPA first (best for small datasets), fallback to MIPROv2
try:
    from dspy.teleprompt.gepa import GEPA
    OPTIMIZER_TYPE = "GEPA"
except ImportError:
    try:
        from dspy.teleprompt import MIPROv2
        OPTIMIZER_TYPE = "MIPROv2"
    except ImportError:
        from dspy.teleprompt import BootstrapFewShot
        OPTIMIZER_TYPE = "BootstrapFewShot"

print(f"Using optimizer: {OPTIMIZER_TYPE}")


def configure_optimizer(
    metric_func,
    teacher_model: str = "openai/gpt-4-turbo",
    student_model: str = "perplexity/sonar",
) -> Any:
    """
    Configure the best available optimizer for greenhouse detection.

    Priority:
    1. GEPA (best for 60-65 examples, achieves 80.7% → 97.8% in research)
    2. MIPROv2 (reliable fallback, optimizes instructions + examples)
    3. BootstrapFewShot (baseline, optimizes examples only)

    Args:
        metric_func: Evaluation metric (must return (score, feedback) for GEPA)
        teacher_model: Powerful model for generating examples/reflection
        student_model: Production model being optimized

    Returns:
        Configured optimizer instance
    """
    if OPTIMIZER_TYPE == "GEPA":
        console.print("[cyan]Using GEPA optimizer (research: 80.7% → 97.8%)[/cyan]")

        # GEPA configuration for small datasets
        optimizer = GEPA(
            metric=metric_func,  # Must return (score, feedback) tuple

            # Evolutionary parameters
            generations=15,  # Number of prompt evolution iterations
            population_size=8,  # Prompts to maintain in Pareto frontier
            mutation_probability=0.5,  # Chance of reflective mutation

            # Models
            reflection_model=dspy.LM(teacher_model),  # For analyzing failures
            task_model=dspy.LM(student_model),  # Model being optimized

            # Validation strategy
            validation_strategy='cross_validate',
            num_folds=5,  # 5-fold CV during optimization

            # Performance
            num_threads=4,
            cache_dir=Path("cache/gepa"),
        )

        console.print(
            f"  Generations: 15 | Population: 8 | Validation: 5-fold CV\n"
            f"  Teacher: {teacher_model}\n"
            f"  Student: {student_model}"
        )

    elif OPTIMIZER_TYPE == "MIPROv2":
        console.print("[cyan]Using MIPROv2 optimizer (reliable fallback)[/cyan]")

        optimizer = MIPROv2(
            metric=metric_func,  # Can accept (score, feedback) or just score
            prompt_model=dspy.LM(teacher_model),
            task_model=dspy.LM(student_model),

            # Auto-tuning
            auto="medium",  # Auto-select hyperparameters (light/medium/heavy)

            # Bayesian optimization parameters
            num_candidates=10,  # Instruction variants to try
            init_temperature=1.0,

            # Few-shot parameters
            max_bootstrapped_demos=4,
            max_labeled_demos=6,

            # Performance
            num_threads=4,
        )

        console.print(
            f"  Candidates: 10 | Auto-tune: medium | Demos: 4+6\n"
            f"  Expected F1: 90-93%"
        )

    else:  # BootstrapFewShot
        console.print("[yellow]Using BootstrapFewShot (baseline)[/yellow]")

        # Wrap metric if it returns tuple
        def wrapped_metric(example, prediction):
            result = metric_func(example, prediction)
            if isinstance(result, tuple):
                return result[0]  # Return score only
            return result

        optimizer = BootstrapFewShot(
            metric=wrapped_metric,
            max_bootstrapped_demos=5,
            max_labeled_demos=10,
            teacher_settings=dict(lm=dspy.LM(teacher_model)),
        )

        console.print(
            f"  Demos: 5+10 | Teacher: {teacher_model}\n"
            f"  Expected F1: 85-90%"
        )

    return optimizer


# Usage in optimization pipeline
def run_optimization():
    """Run the optimization with best available optimizer."""

    # Load expanded dataset (60-65 examples)
    all_examples = load_all_greenhouse_data()

    # Train/val split (80/20 stratified)
    from sklearn.model_selection import train_test_split

    train_set, val_set = train_test_split(
        all_examples,
        test_size=0.2,
        stratify=[ex.uses_growlight for ex in all_examples],
        random_state=42
    )

    console.print(f"\nDataset: {len(train_set)} train, {len(val_set)} validation")

    # Configure optimizer
    optimizer = configure_optimizer(
        metric_func=dutch_aware_hierarchical_f1,
        teacher_model="openai/gpt-4-turbo",  # Or gemini/gemini-2.5-pro
        student_model="perplexity/sonar-pro",
    )

    # Compile optimized program
    base_program = GreenhouseDetector()

    with Progress(...) as progress:
        task = progress.add_task("Optimizing with {OPTIMIZER_TYPE}...", total=None)

        optimized_program = optimizer.compile(
            base_program,
            trainset=train_set,
            valset=val_set,  # GEPA/MIPRO use this for internal validation
        )

        progress.update(task, completed=True)

    return optimized_program, val_set
```

**Teacher Model Selection:**

| Model | Strengths | Dutch Support | Cost | Recommendation |
|-------|-----------|---------------|------|----------------|
| GPT-4 Turbo | Best reasoning, 128K context | Good | $$$ | **Best choice** |
| GPT-5 Pro | Cutting-edge, extended thinking | Excellent | $$$$ | If budget allows |
| Gemini 2.5 Pro | 1M context, strong multilingual | **Excellent** | $$ | **Best value for Dutch** |
| Claude 3.5 Sonnet | Strong reasoning, coding | Good | $$$ | Alternative |

**Recommendation:** Use **Gemini 2.5 Pro** as teacher model for Dutch language support + cost efficiency.

---

### Phase 5: Repeated Nested Cross-Validation (Day 10)

**File:** `src/sevenrad_ee/ai/cross_validation.py` (new)

```python
"""Robust cross-validation framework for small datasets."""

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from typing import Callable, Any
import dspy
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def repeated_nested_cv(
    program_class: type,
    optimizer_func: Callable,
    all_examples: list[dspy.Example],
    metric_func: Callable,
    n_repeats: int = 10,
    n_folds: int = 5,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Repeated nested cross-validation for robust F1 estimation.

    This addresses the severe overfitting problem by:
    1. Using stratified K-fold to maintain class balance
    2. Repeating CV multiple times with different random seeds
    3. Reporting mean ± std for reliability assessment
    4. Tracking train/val gap to detect overfitting

    Args:
        program_class: DSPy program class to instantiate
        optimizer_func: Function that takes (trainset, valset) and returns optimizer
        all_examples: Full dataset (60-65 examples)
        metric_func: Evaluation metric function
        n_repeats: Number of CV repetitions (default: 10)
        n_folds: Number of folds (default: 5)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with:
        - mean_f1: Mean validation F1 across all folds
        - std_f1: Standard deviation
        - train_f1: Mean training F1
        - overfitting_gap: train_f1 - mean_f1
        - fold_scores: List of all fold scores
        - detailed_results: Per-fold breakdown
    """
    console.print("\n[bold cyan]Repeated Nested Cross-Validation[/bold cyan]")
    console.print(f"Repeats: {n_repeats} | Folds: {n_folds} | Total runs: {n_repeats * n_folds}\n")

    # Extract labels for stratification
    labels = [ex.uses_growlight for ex in all_examples]

    # Repeated stratified K-fold
    rkf = RepeatedStratifiedKFold(
        n_splits=n_folds,
        n_repeats=n_repeats,
        random_state=random_seed
    )

    val_scores = []
    train_scores = []
    detailed_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        total_iterations = n_repeats * n_folds
        task = progress.add_task(
            "Running CV...",
            total=total_iterations
        )

        fold_num = 0
        for train_idx, val_idx in rkf.split(all_examples, labels):
            fold_num += 1

            # Create train/val splits
            train_fold = [all_examples[i] for i in train_idx]
            val_fold = [all_examples[i] for i in val_idx]

            progress.update(
                task,
                description=f"Fold {fold_num}/{total_iterations}: Optimizing...",
                completed=fold_num - 1
            )

            # Optimize on train fold
            optimizer = optimizer_func(train_fold, val_fold)
            base_program = program_class()

            optimized_program = optimizer.compile(
                base_program,
                trainset=train_fold
            )

            # Evaluate on train fold (to measure overfitting)
            train_predictions = [
                optimized_program(location_name=ex.location_name, location_area=ex.location_area)
                for ex in train_fold
            ]
            train_fold_score = np.mean([
                metric_func(ex, pred)[0] if isinstance(metric_func(ex, pred), tuple)
                else metric_func(ex, pred)
                for ex, pred in zip(train_fold, train_predictions)
            ])
            train_scores.append(train_fold_score)

            # Evaluate on validation fold
            val_predictions = [
                optimized_program(location_name=ex.location_name, location_area=ex.location_area)
                for ex in val_fold
            ]
            val_fold_score = np.mean([
                metric_func(ex, pred)[0] if isinstance(metric_func(ex, pred), tuple)
                else metric_func(ex, pred)
                for ex, pred in zip(val_fold, val_predictions)
            ])
            val_scores.append(val_fold_score)

            detailed_results.append({
                'fold': fold_num,
                'train_score': train_fold_score,
                'val_score': val_fold_score,
                'train_size': len(train_fold),
                'val_size': len(val_fold),
            })

            progress.update(task, completed=fold_num)

    # Calculate aggregate statistics
    mean_val_f1 = np.mean(val_scores)
    std_val_f1 = np.std(val_scores)
    mean_train_f1 = np.mean(train_scores)
    overfitting_gap = mean_train_f1 - mean_val_f1

    # Display results
    console.print("\n[bold]Cross-Validation Results:[/bold]")
    console.print(f"  Mean Validation F1: {mean_val_f1:.2%} ± {std_val_f1:.2%}")
    console.print(f"  Mean Training F1: {mean_train_f1:.2%}")
    console.print(f"  Overfitting Gap: {overfitting_gap:+.2%}")

    # Overfitting assessment
    if overfitting_gap > 0.15:
        console.print("  [red]⚠ SEVERE OVERFITTING (gap >15%)[/red]")
    elif overfitting_gap > 0.10:
        console.print("  [yellow]⚠ Moderate overfitting (gap 10-15%)[/yellow]")
    else:
        console.print("  [green]✓ Good generalization (gap <10%)[/green]")

    # Stability assessment
    if std_val_f1 > 0.05:
        console.print("  [yellow]⚠ High variance (std >5%) - may need more data[/yellow]")
    else:
        console.print("  [green]✓ Stable performance (std <5%)[/green]")

    return {
        'mean_f1': mean_val_f1,
        'std_f1': std_val_f1,
        'train_f1': mean_train_f1,
        'overfitting_gap': overfitting_gap,
        'fold_scores': val_scores,
        'detailed_results': detailed_results,
    }
```

---

### Phase 6: Dutch Search Query Templates (Day 11)

**File:** `src/sevenrad_ee/ai/search_strategy.py` (new)

```python
"""Dutch-first web search strategy for greenhouse detection."""

from typing import Literal

# Dutch search query templates
DUTCH_SEARCH_QUERIES = {
    'greenhouse_verification': [
        "{company_name} {location} kwekerij",
        "{company_name} {location} glastuinbouw",
        "{company_name} {location} teler",
        "{company_name} tuinbouw nederland",
    ],

    'growlight_detection': [
        "{company_name} assimilatiebelichting",
        "{company_name} assimilatieverlichting",
        "{company_name} belichte teelt",
        "{company_name} kunstlicht kassen",
        "{company_name} LED-belichting",
        "{company_name} SON-T lampen",
        "{company_name} groeilicht",
    ],

    'supplier_validation': [
        "{company_name} Signify belichting",
        "{company_name} Philips Hortilux",
        "{company_name} Priva klimaatcomputer",
        "{company_name} Hoogendoorn automatisering",
    ],

    'negative_indicators': [
        "{company_name} onbelichte teelt",
        "{company_name} daglichtkas",
        "{company_name} biologisch onbelicht",
    ],

    'trade_media': [
        "{company_name} site:groentenfruit.nl",
        "{company_name} site:floraldaily.com/nl",
        "{company_name} kas magazine",
    ],
}


# Evidence source tier classification
SourceTier = Literal["company_website", "supplier_case_study", "job_posting",
                     "trade_media_nl", "general_web"]


def classify_source_tier(url: str, content: str) -> SourceTier:
    """
    Classify evidence source quality tier.

    Tier 2 (Highest Quality):
    - Company website: company domain in URL
    - Supplier case study: Signify, Philips, Priva domains + company mention
    - Job posting: Indeed, LinkedIn with "belichting" in title

    Tier 1 (Medium Quality):
    - Dutch trade media: groentenfruit.nl, floraldaily.com/nl, kasalsenergiebron.nl

    Tier 0.5 (Low Quality):
    - General web: Other sources

    Args:
        url: Source URL
        content: Page content or excerpt

    Returns:
        Source tier classification
    """
    url_lower = url.lower()
    content_lower = content.lower()

    # Tier 2: Company website
    company_tlds = ['.nl/', '.com/', '.eu/']
    if any(tld in url_lower for tld in company_tlds):
        # Check if it's the actual company website (not a news site)
        if not any(news in url_lower for news in ['nieuws', 'news', 'media', 'blog']):
            return "company_website"

    # Tier 2: Supplier case studies
    supplier_domains = ['signify.com', 'philips.com/hortilux', 'priva.com',
                        'hoogendoorn.nl', 'hortilux.com']
    if any(domain in url_lower for domain in supplier_domains):
        return "supplier_case_study"

    # Tier 2: Job postings
    job_domains = ['indeed.nl', 'linkedin.com', 'werkzoeken.nl']
    job_keywords = ['vacature', 'belichting', 'teeltspecialist', 'assimilatie']
    if any(domain in url_lower for domain in job_domains):
        if any(keyword in content_lower for keyword in job_keywords):
            return "job_posting"

    # Tier 1: Dutch trade media
    trade_media = ['groentenfruit.nl', 'floraldaily.com/nl', 'kasalsenergiebron.nl',
                   'tuinbouw.nl', 'emerce.nl/agrifood']
    if any(domain in url_lower for domain in trade_media):
        return "trade_media_nl"

    # Tier 0.5: General web (default)
    return "general_web"


def generate_search_queries(
    company_name: str,
    location: str,
    query_types: list[str] = None
) -> list[str]:
    """
    Generate Dutch-language search queries for greenhouse detection.

    Args:
        company_name: Company name to search for
        location: Geographic location
        query_types: List of query types to generate (default: all)

    Returns:
        List of formatted search queries
    """
    if query_types is None:
        query_types = list(DUTCH_SEARCH_QUERIES.keys())

    all_queries = []

    for query_type in query_types:
        templates = DUTCH_SEARCH_QUERIES.get(query_type, [])

        for template in templates:
            query = template.format(
                company_name=company_name,
                location=location
            )
            all_queries.append(query)

    return all_queries


# Example usage
if __name__ == "__main__":
    # Generate queries for a test company
    queries = generate_search_queries(
        company_name="Porta Nova",
        location="Waddinxveen",
        query_types=['greenhouse_verification', 'growlight_detection']
    )

    print("Generated Dutch search queries:")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")
```

---

## Success Criteria & Validation

### Must-Have Metrics (95% F1 Minimum)

✅ **Cross-Validated F1 ≥ 95%**
- Mean F1 across 10-repeat 5-fold CV
- Measured on 60-65 real examples
- No synthetic data in test folds

✅ **Standard Deviation ≤ 3%**
- Indicates stable, consistent performance
- Low variance across different data splits

✅ **Overfitting Gap < 10%**
- Train F1 - Val F1 < 10 percentage points
- Current gap: 26.6% (severe) → Target: <10%

✅ **Known Company Validation**
- 0 false negatives on 13 known positive companies
- 0 false positives on 6 known negative companies

✅ **Evidence Quality**
- >80% of YES predictions have tier-2 evidence
- >90% of predictions include Dutch terminology

### Nice-to-Have Metrics (98% F1 Stretch)

🎯 **Precision > 0.95 for uses_growlight=YES**
- Minimize false positives
- Critical for credibility

🎯 **Dutch Term Coverage > 90%**
- Most positive cases mention Dutch terminology
- Validates Dutch-first search strategy

🎯 **Hierarchical Logic Perfect**
- 100% compliance with gating rules
- No logical inconsistencies

---

## Expert Analysis Summary

### GPT-5-Pro Key Insights

1. **Hybrid Architecture Recommendation**
   - Initially suggested VIIRS + web search hybrid
   - Acknowledged web-only is valid if constraints require it
   - VIIRS helps recall for sparse-web-presence facilities

2. **Evidence-First Philosophy**
   - "MANDATORY: Evidence sources with URLs + quotes"
   - Hierarchical classification with gating
   - UNKNOWN state critical to avoid forced guessing

3. **Data Quality > Synthetic Data**
   - "Prefer weakly-labeled real examples over synthetic text"
   - Synthetic examples teach artifacts, not patterns
   - Real negatives from "onbelichte teelt" companies

4. **Dutch Language Priority**
   - Source tier system (official > trade media > general)
   - Dutch term extraction as explicit output field
   - Prioritize: assimilatiebelichting, kunstlicht, kassen

5. **Operational Concerns**
   - Rate limits, latency, result drift over time
   - Caching, evidence versioning, de-duplication
   - Identity resolution (aliases, mergers, name changes)

### Gemini-2.5-Pro Key Insights

1. **Overfitting Diagnosis**
   - "89% train vs 62% val is SEVERE overfitting"
   - 4-example validation set → high-variance metrics
   - Cross-validation NON-NEGOTIABLE for reliability

2. **Optimizer Strategy**
   - "MIPROv2 more reliable than GEPA in production"
   - BootstrapFewShot plateau around 90%
   - Teacher-student with stronger model = +10 points

3. **Signature Engineering**
   - "Dutch-specific instructions are powerful and cheap"
   - Start simple, add constraints incrementally
   - Explicit guidance > hoping model figures it out

4. **Realistic Targets**
   - "With ~42 examples, 98% F1 unlikely without leakage"
   - "80-90% F1 realistic with disciplined CV"
   - "95% is stretch with robust evidence + curated data"

5. **Implementation Priorities**
   - Phase 1: Cross-validation baseline (critical)
   - Phase 2: Dutch signature + metric
   - Phase 3: MIPROv2 optimization
   - Don't skip straight to GEPA without baseline

---

## Timeline & Effort Estimate

### Conservative Estimate: 11 Days

| Phase | Tasks | Days | Cumulative |
|-------|-------|------|------------|
| 1 | Fix DSPy + expand data to 60-65 examples | 3 | Day 3 |
| 2 | Hierarchical signature + validation rules | 2 | Day 5 |
| 3 | Dutch-aware feedback metric | 1 | Day 6 |
| 4 | GEPA/MIPRO optimizer configuration | 3 | Day 9 |
| 5 | Repeated nested CV framework | 1 | Day 10 |
| 6 | Dutch search templates + final validation | 1 | Day 11 |

### Aggressive Estimate: 7-8 Days

- Parallelize data expansion (Phase 1) with code development (Phases 2-3)
- Use MIPROv2 instead of GEPA (faster, more reliable)
- Skip some validation refinements

---

## Risk Assessment & Mitigation

### High-Risk Items

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| GEPA not working in GitHub version | Medium | High | Fallback to MIPROv2 (target 90-93% F1) |
| Cannot source 40+ real examples | Medium | Critical | Use weakly-labeled examples from trade media |
| Perplexity rate limits | Low | Medium | Cache results, implement retry with backoff |
| 95% target not achieved | Medium | Medium | Accept 90% as floor with <5% std |

### Medium-Risk Items

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Dutch term extraction quality | Medium | Medium | Manual review + refinement of term list |
| Evidence source classification errors | Medium | Low | Human validation of tier assignments |
| Cross-validation takes too long | Low | Low | Reduce n_repeats from 10 to 5 |

---

## Alternative Approaches

### If 95% Not Achieved with Plan

**Option A: Ensemble Approach**
```python
# Combine multiple specialized classifiers
dutch_classifier = optimize_for_dutch_terms()
english_classifier = optimize_for_english_terms()
species_classifier = optimize_for_species_heuristics()

# Weighted ensemble
final_prediction = ensemble_vote(
    dutch_classifier * 2.0,  # Weight Dutch 2x
    english_classifier * 1.0,
    species_classifier * 0.5
)
```

**Option B: Active Learning Loop**
1. Start with 60-65 labeled examples
2. Predict on unlabeled companies
3. Human review top 20 uncertain predictions
4. Add to training set
5. Retrain and repeat

**Option C: Reduce Scope**
- Focus only on specific greenhouse types (e.g., roses, tomatoes)
- Limit to specific regions (Westland, Zuid-Holland)
- Accept lower recall for higher precision

---

## Key Files to Create/Modify

### Files to Modify ✏️

1. **`src/sevenrad_ee/ai/dspy_greenhouse.py`**
   - Hierarchical signature with Dutch guidance
   - Evidence source structure
   - Pydantic validation models

2. **`src/sevenrad_ee/ai/dspy_evaluation.py`**
   - Dutch-aware hierarchical F1 metric
   - Feedback generation for GEPA reflection
   - Backward-compatible wrappers

3. **`src/sevenrad_ee/ai/dspy_training_data.py`**
   - Load expanded dataset (60-65 examples)
   - Data quality fixes
   - Identity resolution integration

4. **`src/sevenrad_ee/operations/optimize_greenhouse_detection.py`**
   - GEPA/MIPRO/BootstrapFewShot configuration
   - Teacher-student model setup
   - Cross-validation integration

### Files to Create ✨

5. **`src/sevenrad_ee/ai/company_identity.py`**
   - Company identity resolution
   - Alias/trade name handling
   - Temporal tracking (acquisitions, etc.)

6. **`src/sevenrad_ee/ai/cross_validation.py`**
   - Repeated nested CV framework
   - Stratified splitting
   - Overfitting detection

7. **`src/sevenrad_ee/ai/search_strategy.py`**
   - Dutch query templates
   - Source tier classification
   - Evidence quality scoring

8. **`data/dutch_companies_expanded.json`**
   - Full 60-65 example dataset
   - Quality-checked and deduplicated

9. **`data/company_identities.json`**
   - Canonical company mappings
   - Aliases and trade names

10. **`data/flagged_issues.json`**
    - Data quality issues to fix
    - Resolution tracking

---

## Next Steps

### Immediate Actions (Today)

1. ✅ **Fix DSPy Installation**
   ```bash
   uv pip uninstall dspy-ai
   uv pip install "git+https://github.com/stanfordnlp/dspy.git"
   ```

2. ✅ **Verify Optimizer Availability**
   ```python
   from dspy.teleprompt.gepa import GEPA
   from dspy.teleprompt import MIPROv2
   print("✓ Optimizers available")
   ```
   **Status:** GEPA 0.0.18 confirmed available in DSPy GitHub installation

3. ✅ **Create Data Collection Plan**
   - ✅ Consulted Gemini-2.5-Pro for Phase 1 implementation strategy
   - ✅ Defined 4-step sequential approach (Environment → Prototyping → Automation → Validation)
   - ✅ Established risk mitigation strategies and checkpoints
   - ✅ Identified data sources: Perplexity Sonar-Pro API with Dutch domain filtering
   - ✅ Added marimo for interactive HITL validation workflows
   - 📋 Ready to begin implementation

### This Week

4. 📊 **Expand Dataset**
   - Add 18 provided companies (13 positive, 6 negative)
   - Source 20+ additional real examples
   - Fix schenkeveld, ubink, HilverdaFlorist issues

5. 💻 **Implement Core Features**
   - Hierarchical signature
   - Dutch-aware metric
   - Identity resolution module

### Next Week

6. 🔧 **Optimization & Validation**
   - Configure GEPA/MIPRO
   - Run repeated nested CV
   - Achieve 95%+ F1 target

7. 📈 **Production Deployment**
   - Save optimized model
   - Document search strategy
   - Create operational runbook

---

## References & Research

### DSPy Documentation
- [DSPy Optimizers](https://dspy.ai/learn/optimization/optimizers/)
- [GEPA Overview](https://dspy.ai/api/optimizers/GEPA/overview/)
- [MIPROv2 API](https://dspy.ai/api/optimizers/MIPROv2/)

### Research Papers
- **GEPA:** "Reflective Prompt Evolution Can Outperform Reinforcement Learning" (Agrawal et al., 2025)
- **DSPy Framework:** "Programming with Language Models" (Khattab et al., 2024)

### Expert Analysis Sources
- GPT-5-Pro deep analysis (this document)
- Gemini-2.5-Pro deep analysis (this document)
- Research on small-dataset classification with DSPy

### Dutch Greenhouse Industry
- **Glastuinbouw Nederland:** Main industry association
- **Kas als Energiebron:** Sustainability initiative
- **Groenten & Fruit:** Trade publication
- **Floraldaily NL:** Horticultural news

---

## Questions for User

Before proceeding with full implementation, please confirm:

1. ✅ **DSPy from GitHub acceptable?** (development version vs PyPI)
2. ✅ **Teacher model choice?** (GPT-4 Turbo vs Gemini 2.5 Pro vs GPT-5 Pro)
3. ✅ **Budget for API calls?** (Teacher model generates many examples)
4. ❓ **Access to Dutch greenhouse directories?** (Industry associations, member lists)
5. ❓ **Time allocation?** (Full-time for 11 days vs part-time over longer period)

---

**END OF OPTIMIZATION PLAN**

*Generated by dual-model deep analysis (GPT-5-Pro + Gemini-2.5-Pro) via Claude Code*

---

## Implementation Log

### 2025-10-25: Phase 1 Planning & Setup Complete

**Session 1: Expert Planning Consultation**
- Consulted Gemini-2.5-Pro via Zen MCP for detailed Phase 1 implementation strategy
- Established 4-step sequential approach with risk mitigation:
  1. Environment & Foundation (Day 1 AM)
  2. Manual Prototyping (Day 1 PM)
  3. Incremental Automation (Day 2)
  4. Scaled Collection & HITL Validation (Day 3)
- Zen continuation_id: `f378f80f-feea-44b7-9561-9a3271f97e8c` (49 turns remaining)

**Setup Actions:**
- ✅ Added `marimo~=0.9.0` to `data-augmentation` optional dependency group in pyproject.toml
- ✅ Installed marimo 0.9.34 for interactive HITL validation notebooks
- ✅ Verified GEPA optimizer (v0.0.18) available in DSPy GitHub installation
- ✅ Updated IMPROVE_PROMPT.md with Implementation Progress tracking section
- ✅ Confirmed PERPLEXITY_API_KEY environment variable is configured

**Project Configuration:**
- Data collection code location: `src/sevenrad_ee/operations/`
- Dataset format: YAML with Pydantic v2 validation
- HITL validation workflow: Marimo notebooks (`notebooks/phase1_*.marimo`)
- Caching strategy: SHA-256 file-based cache (cache/ directory)

**Next Actions:**
Ready to begin Step 1.1: Fix DSPy installation (remove PyPI, install from GitHub)

**Session 2: DSPy Installation Verification & Cleanup**
- ✅ Verified DSPy 3.0.4b2 from GitHub already installed (from pyproject.toml)
- ✅ Removed conflicting dspy-ai 3.0.3 from PyPI (broken version)
- ✅ Confirmed all optimizers work: BootstrapFewShot, MIPROv2, GEPA
- ✅ Step 1.1 complete - ready for Step 1.2: SHA-256 caching infrastructure
