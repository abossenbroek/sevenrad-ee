# F1 Optimization Strategy: Critical Analysis & Revised Approach

**Date**: 2025-11-18
**Status**: Diagnostic Plan - Awaiting Execution
**Expert Consultation**: Gemini 2.5 Pro via Zen MCP

---

## Executive Summary

**CRITICAL INSIGHT**: Traditional DSPy optimization may be fundamentally mismatched for a RAG-with-web-search system. The dynamic nature of Perplexity's web search creates non-deterministic behavior that can make prompt optimization brittle and non-generalizable.

**RECOMMENDATION**: **DO NOT** immediately proceed with the planned 65-example GEPA optimization. Instead, follow a phased diagnostic and model-upgrade approach.

---

## Problem Statement

### Current Situation
- **Dataset**: Expanded from 19 to 65 companies (36 POSITIVE, 29 NEGATIVE)
- **Current Performance**: 63.39% F1 (±6.65%) on 19-example baseline
- **Target**: 75-85% F1 with 65 examples
- **Plan**: Run 8-10 hour GEPA optimization (see `GEPA_NEXT_STEP.md`)

### Critical Question Raised
**Does traditional DSPy optimization make sense for a RAG-with-live-web-search system?**

---

## Expert Analysis: Why Traditional Optimization May Fail

### The RAG Web Search Problem

**Your System Architecture**:
```
User Query → Perplexity API → Live Web Search → RAG → LLM → Classification
```

**Key Insight**: The "model" is not just the LLM—it's the entire pipeline including live web search, which introduces significant variability.

### Why Traditional DSPy Optimization Assumes Determinism

Traditional optimizers (GEPA, MIPROv2, BootstrapFewShot) operate under these assumptions:
1. For a given input, the model behavior is deterministic or stochastically consistent
2. A good prompt reliably guides the model to correct answers
3. Performance improvements transfer to new inputs

**Your system violates all three assumptions.**

### Risks of Naive GEPA Optimization

1. **Temporal Overfitting**:
   - Optimized prompt becomes effective for web results available *during optimization*
   - If those results change (news article drops off front page), effectiveness plummets
   - You're optimizing for a **transient state of the internet**

2. **Noisy Optimization Signal**:
   - Good prompt might fail due to poor search results (noise)
   - Mediocre prompt might succeed due to lucky search (noise)
   - Optimizer can't distinguish prompt quality from search luck
   - Prevents convergence on genuinely robust solutions

3. **Brittle Generalization**:
   - Few-shot examples optimized for "clean" company data (established firms with good web presence)
   - Fails on "sparse" company data (small firms, limited online presence)
   - Not learning general reasoning—learning to handle specific data quality levels

---

## Root Cause Analysis: Train/Validation Gap

### Most Likely Explanation

**Web Resource Availability Differences** (not traditional overfitting)

**Hypothesis**:
- **Training set**: Well-established companies (Apple, Shell, Toyota)
  - Dedicated investor relations pages
  - Wikipedia entries
  - Extensive news coverage
  - RAG finds clear, factual information easily

- **Validation set**: Smaller, private, or niche companies
  - Sparse web presence
  - Marketing-heavy content
  - Fewer third-party articles
  - RAG finds ambiguous, noisy information

**Result**: Performance gap arises because the model classifies *the web's representation* of companies, not the companies themselves. If representations differ systematically between splits, performance will differ.

### Why This Matters

If web resource availability is the confounding factor:
- **More training examples won't help** if they have the same quality bias
- **Prompt optimization won't help** if the underlying information is missing
- **Need better model** that handles noisy/sparse information better

---

## Solution: Phased Diagnostic Approach

### Phase 1: Validate the Hypothesis (1-2 hours)

#### 1.1 Qualitative Company Audit

**Action**:
1. Sample 5 companies where model performs **well**
2. Sample 5 companies where model performs **poorly**
3. For each, manually search: "[Company Name] business model", "[Company Name] what they do"
4. Document:
   - Quality of top results (clear vs ambiguous)
   - Availability of factual sources (Wikipedia, reports)
   - Presence of Dutch sources
   - Marketing vs factual content ratio

**Expected Outcome**: Confirm if failing companies have systematically worse web presence

**Script to Create**:
```bash
# scripts/audit_web_presence.py
# Generate list of high/low performing companies for manual review
```

#### 1.2 Web Presence Proxy Metric

**Create Quantitative Metric**:
```python
web_presence_score = {
    'has_wikipedia': 1 if exists else 0,
    'is_public_company': 1 if public else 0,
    'employee_count': normalize(0-1)  # From LinkedIn
}
composite_score = mean(components)
```

**Analysis**:
1. Calculate average score for training vs validation sets
2. Statistical test: Are they significantly different?
3. Correlation: Does `web_presence_score` correlate with per-company F1?
4. If correlation > 0.6 → **web presence is confounding factor**

**Script to Create**:
```bash
# scripts/calculate_web_presence_scores.py
# Outputs: web_presence_analysis.md with statistics
```

**Decision Point**:
- **If strong correlation** → Proceed to Phase 2 (model upgrade)
- **If weak correlation** → Investigate other factors (data quality, classification errors)

---

### Phase 2: Model Upgrade A/B Test (2-3 hours)

#### 2.1 Why Sonar Reasoning Pro?

**Current Model**: `perplexity/sonar`
- Fast, optimized for quick factual answers
- No explicit reasoning capability
- May struggle with ambiguous/contradictory evidence

**Upgrade Option**: `perplexity/sonar-reasoning-pro`
- **Chain-of-Thought reasoning** with `<think>` section before answer
- Better at synthesizing contradictory information
- Designed for "logical breakdowns and planning"
- **Pricing**: $2/M input tokens, $8/M output tokens, $6-14 per 1000 requests
- **Context**: 127k tokens (vs 200k for sonar-pro)

**Why This Addresses the Core Problem**:

Your task requires:
- ✅ Synthesizing information from multiple search snippets
- ✅ Differentiating marketing fluff from factual descriptions
- ✅ Deducing business models when not explicitly stated
- ✅ Handling contradictory evidence gracefully

→ **This is EXACTLY what Sonar Reasoning Pro is designed for**

A stronger reasoning model is better equipped to:
- **Signal vs Noise**: Distinguish between marketing and substance
- **Synthesis**: Integrate conflicting/partial information from multiple snippets
- **Inference**: Deduce business model when not explicit

#### 2.2 A/B Test Protocol

**Test Setup**:
```python
# Create stratified validation subset
validation_subset = {
    'size': 20 companies,
    'split': 10 POSITIVE, 10 NEGATIVE,
    'stratify_by': web_presence_score (high/medium/low)
}
```

**Configurations to Compare**:
```python
# Config A: Current baseline
model_a = dspy.LM('perplexity/sonar', api_key=...)

# Config B: Reasoning upgrade
model_b = dspy.LM('perplexity/sonar-reasoning-pro', api_key=...)
```

**CRITICAL**: Use cached research results from `data/research/*.json` to ensure fair comparison (same web search results for both models)

**Metrics to Measure**:
```
Performance:
- F1 Score
- Precision (false positive rate)
- Recall (false negative rate)
- Confusion matrix

Cost:
- API cost per classification ($)
- Latency per classification (seconds)

Quality:
- Reasoning quality in <think> section (qualitative)
- Citation accuracy
- Confidence calibration
```

**Success Criteria**:
- **F1 improvement ≥10%** → **ADOPT** Sonar Reasoning Pro, reconsider optimization need
- **F1 improvement 5-10%** → **ADOPT**, proceed with manual prompt refinement
- **F1 improvement <5%** → Investigate data quality issues

**Script to Create**:
```bash
# scripts/ab_test_perplexity_models.py
# Outputs: model_comparison_report.md
```

---

### Phase 3: Conditional Strategy (Depends on Phase 2 Results)

#### Scenario A: Strong Improvement (F1 +10%+)

**Decision**: **Skip traditional GEPA optimization initially**

**Rationale**:
- Model upgrade addresses core issue (handling noisy search results)
- May already achieve 75-85% F1 target
- Avoid risks of optimizing on dynamic web search

**Next Steps**:
1. **Manual Prompt Refinement** (not automated):
   ```python
   # Add to GreenhouseClassification signature
   additional_instructions = """
   Synthesize information from all provided web search results.
   Prioritize factual descriptions from sources like Wikipedia or
   official reports over marketing language from company main pages.
   If primary business activity unclear, state that and provide
   best possible inference with confidence level.
   """
   ```

2. **Stratified Data Split**:
   - Re-split train/validation by web_presence_score
   - Ensure both sets have similar distribution of "hard" companies
   - Prevents web presence bias in evaluation

3. **Re-evaluate on Full Dataset**:
   - If F1 ≥75% → **Deploy to pilot with human review**
   - If F1 <75% → Consider Phase 4

#### Scenario B: Modest Improvement (F1 +5-10%)

**Decision**: Adopt model + Manual refinement

**Actions**:
1. Switch to Sonar Reasoning Pro as base model
2. Manually refine signature based on failure mode analysis
3. Add explicit error handling instructions
4. Re-evaluate on full validation set
5. If F1 ≥75% → Deploy to pilot
6. If F1 <75% → Consider Phase 4

#### Scenario C: No Improvement (F1 <+5%)

**Decision**: Deep-dive into data quality

**Actions**:
1. Review manual classifications for errors
2. Analyze evidence quality in `data/research/*.json` files
3. Check if 3-query strategy retrieves useful information
4. Verify ground truth labels from VIIRS satellite data
5. Consider hybrid human-AI classification approach

---

### Phase 4: Cached-Data GEPA Optimization (ONLY IF NEEDED)

**When to Use**: Only if Phases 1-3 don't achieve 75% F1 target

#### Why Cached Data?

**Problem with Live Optimization**:
```
GEPA iteration 1: Company X → Web search → Good results → Classify correctly
GEPA iteration 20: Company X → Web search → Different results → Classify incorrectly
```
→ Noisy signal prevents optimization convergence

**Solution: Freeze the Web Search Results**:
```
Pre-optimization: Load all data/research/*.json files
GEPA iteration 1-20: Company X → Use cached results → Consistent behavior
```
→ Stable signal enables optimization

#### Implementation

**Step 1: Create Static Dataset**
```python
# scripts/create_static_optimization_dataset.py

def create_static_dataset():
    """Load all cached Perplexity responses."""
    dataset = []
    for company_file in Path("data/research").glob("*.json"):
        data = json.loads(company_file.read_text())
        dataset.append({
            'company': data['company_name'],
            'location': data['location'],
            'cached_queries': data['queries'],  # Pre-fetched results
            'label': data['manual_classification']
        })
    return dataset
```

**Step 2: Modify DSPy to Use Cached Data**
```python
# Disable live Perplexity API calls during optimization
# Feed cached query results directly to predictor
class CachedPerplexityLM(dspy.LM):
    def __init__(self, cache_data):
        self.cache = cache_data

    def __call__(self, query):
        # Return cached response instead of API call
        return self.cache.get(query)
```

**Step 3: Run GEPA on Static Dataset**
```python
# Now GEPA optimizes for:
# "How to reason over real-world messy search results"
# NOT "How to query the current state of the web"

optimizer = GEPA(
    metric=dutch_aware_hierarchical_f1,
    generations=15,
    population_size=8,
    # ... other params
)
```

#### Benefits of Cached-Data Approach

✅ **Stable Optimization**: Same input → Same search results → Consistent signal
✅ **No Temporal Drift**: Optimized prompt valid for months (not tied to current web state)
✅ **Cost-Effective**: No repeated API calls during 21+ GEPA iterations
✅ **Still Realistic**: Learns to handle the *kind* of messiness in real search results
✅ **Generalizable**: Optimizes reasoning strategies, not search strategies

#### Risks & Mitigations

⚠️ **Risk**: Optimized prompt specific to cached data patterns
✅ **Mitigation**: Validate on new companies with live search

⚠️ **Risk**: Cached data may be outdated
✅ **Mitigation**: Use recent cache (data/research from Nov 2025)

**Script to Create**:
```bash
# scripts/optimize_with_cached_data.py
# Implements cached-data GEPA optimization
```

---

## Timeline Comparison

### Original Plan (GEPA_NEXT_STEP.md)
```
Hour 0:00 - Pre-flight check
Hour 0:05 - Launch GEPA optimization
Hour 7:35 - GEPA completes (7.5h)
Hour 7:40 - Launch CV evaluation
Hour 9:50 - CV completes (2.2h)
Hour 10:00 - Analysis
─────────────────────────────────
Total: 10 hours
Cost: $25-40
Risk: HIGH (temporal optimization)
Feedback: 10 hours (all-or-nothing)
```

### Recommended Phased Plan
```
Phase 1: Diagnostic (4 hours)
  Hour 0-2: Qualitative audit
  Hour 2-4: Web presence analysis
  → Decision point: Proceed to Phase 2?

Phase 2: Model A/B Test (3 hours)
  Hour 0-1: Setup test (20 companies)
  Hour 1-2: Run both models
  Hour 2-3: Analyze results
  → Decision point: Which scenario (A/B/C)?

Phase 3: Conditional (varies)
  Scenario A: Manual refinement (4 hours)
  Scenario B: Manual + re-eval (6 hours)
  Scenario C: Data quality dive (8 hours)
  → Decision point: Deploy or Phase 4?

Phase 4: Cached-Data GEPA (8 hours, if needed)
  Hour 0-1: Create static dataset
  Hour 1-8: GEPA optimization
  Hour 8-9: Validation
─────────────────────────────────
Total: 7-15 hours (phased)
Cost: $20-40
Risk: LOW (data-driven decisions)
Feedback: 4 hours (early insights)
```

---

## Cost Analysis

### API Costs

**Perplexity Sonar Reasoning Pro**:
- Input: $2 per 1M tokens
- Output: $8 per 1M tokens
- Requests: $6-14 per 1000 (mode-dependent)

**Gemini 2.5 Pro** (for GEPA teacher model):
- Input: ~$1.25 per 1M tokens
- Output: ~$5 per 1M tokens

**Phase Breakdown**:
- Phase 1: $0 (manual audit)
- Phase 2: $5-10 (20 companies × 2 models)
- Phase 3: $0-5 (manual refinement, limited re-eval)
- Phase 4: $15-25 (cached GEPA, no repeated Perplexity calls)

**Total**: $20-40 (same as original, but phased)

---

## Key Insights from Expert Analysis

### 1. Traditional Optimization Validity

> "Your system violates the assumption that for a given input, the behavior of the LM is deterministic or stochastically consistent. The 'model' is not just the LLM; it's the entire Perplexity API → Web Search → RAG → LLM pipeline. The web search component introduces significant variability."

**Implication**: GEPA optimization on live web search may produce brittle, non-generalizable prompts.

### 2. Most Likely Cause of Train/Validation Gap

> "The most likely cause is a combination of (b) different web resource availability and (c) systematic differences in company types between splits. The performance gap arises because the reasoning process that works for the clean, abundant data of the training set fails when presented with the sparse, noisy data of the validation set."

**Implication**: Need to diagnose web presence correlation before investing in optimization.

### 3. Model Upgrade Priority

> "Yes, absolutely. Prioritize the model upgrade. This is a classic trade-off: improve the 'scaffolding' (the prompt) or improve the 'worker' (the model). In a situation with high input variability and noise (like live web search), a more capable 'worker' is almost always the better investment."

**Implication**: Test Sonar Reasoning Pro before spending 10 hours on GEPA optimization.

### 4. Controlled Optimization Strategy

> "If you still need to squeeze out more performance, you could consider DSPy optimization, but with a crucial modification: **cache the web search results**. Create a static dataset where for each of the 65 companies, you pre-fetch and store the RAG context. Now you have a static training environment."

**Implication**: Cached-data GEPA is safer and more generalizable than live optimization.

---

## Decision Framework

### Phase 1 → Phase 2 Transition

**Go to Phase 2 if**:
- Web presence correlation with F1 > 0.5
- Clear difference in web availability between high/low performers
- Manual audit confirms systematic information quality gaps

**Skip to data quality review if**:
- No clear web presence pattern
- Manual audit shows information is available but classifications are wrong
- Suggests errors in ground truth labels or classification logic

### Phase 2 → Phase 3 Transition

**Scenario A** (F1 +10%+):
- Manual prompt refinement
- Stratified re-split
- Re-evaluate
- If F1 ≥75% → **Deploy to pilot**

**Scenario B** (F1 +5-10%):
- Adopt Sonar Reasoning Pro
- Manual refinement
- If F1 ≥75% → **Deploy to pilot**
- If F1 <75% → **Phase 4**

**Scenario C** (F1 <+5%):
- Deep-dive data quality
- Review classifications
- May need more/better examples

### Phase 3 → Phase 4 Transition

**Go to Phase 4 if**:
- Sonar Reasoning Pro adopted
- Manual refinement complete
- F1 still <75%
- Data quality verified
- **Need that extra 5-10% to hit target**

**Skip Phase 4 if**:
- Already achieved F1 ≥75%
- Data quality issues identified (fix those first)
- Budget/time constraints (deploy pilot instead)

---

## Success Criteria (Revised)

### Phase 1 Success
- ✅ Web presence correlation quantified
- ✅ Manual audit confirms hypothesis
- ✅ Clear understanding of failure modes

### Phase 2 Success
- ✅ Sonar Reasoning Pro improves F1 by ≥5%
- ✅ Cost/latency acceptable for production
- ✅ Reasoning quality demonstrably better

### Phase 3 Success
- ✅ F1 ≥75% achieved through model + manual refinement
- ✅ False positive rate ≤10%
- ✅ Ready for pilot deployment

### Phase 4 Success (if needed)
- ✅ Cached-data GEPA achieves F1 ≥80%
- ✅ Optimized prompt generalizes to new companies
- ✅ Ready for production deployment

---

## Implementation Checklist

### Scripts to Create

- [ ] `scripts/audit_web_presence.py` - Generate high/low performer lists
- [ ] `scripts/calculate_web_presence_scores.py` - Quantitative analysis
- [ ] `scripts/ab_test_perplexity_models.py` - Model comparison
- [ ] `scripts/manual_prompt_refinement.py` - Apply manual improvements
- [ ] `scripts/create_static_optimization_dataset.py` - Cached data prep
- [ ] `scripts/optimize_with_cached_data.py` - Phase 4 GEPA

### Code Modifications

- [ ] Add `configure_perplexity_reasoning_pro()` to optimization module
- [ ] Implement cached Perplexity LM wrapper for Phase 4
- [ ] Add web presence scoring to evaluation pipeline
- [ ] Create stratified splitting by web_presence_score

### Documentation

- [ ] Phase 1 results: `results/diagnostics/web_presence_analysis.md`
- [ ] Phase 2 results: `results/diagnostics/model_comparison_report.md`
- [ ] Phase 3 results: `results/diagnostics/manual_refinement_report.md`
- [ ] Phase 4 results: `results/65examples_cached/optimization_report.md`

---

## Risks & Mitigations

### Risk 1: Web Presence Hypothesis Wrong

**Symptoms**: No correlation between web presence and F1
**Mitigation**: Pivot to data quality review
**Impact**: 4 hours spent on diagnosis (vs 10 hours on failed optimization)

### Risk 2: Sonar Reasoning Pro Doesn't Help

**Symptoms**: F1 improvement <5%
**Mitigation**: Deep-dive into data quality, classification errors
**Impact**: Model upgrade still improves reasoning transparency

### Risk 3: Cached-Data GEPA Doesn't Generalize

**Symptoms**: Optimized prompt works on cached data but fails on live search
**Mitigation**: Validate on new companies, adjust prompt
**Impact**: Still learned useful reasoning strategies

### Risk 4: Can't Achieve 75% F1 Target

**Symptoms**: All phases complete, F1 still <75%
**Options**:
1. Deploy pilot with 65-70% F1 + human review
2. Collect 100+ examples (target 150 total)
3. Hybrid human-AI approach
4. Re-evaluate success criteria

---

## Comparison Table: Original vs Recommended

| Aspect | Original Plan | Recommended Plan |
|--------|---------------|------------------|
| **First Step** | Run GEPA immediately | Diagnose root cause |
| **Model** | `perplexity/sonar` | Test `sonar-reasoning-pro` |
| **Optimization Approach** | Live web search | Cached data (if needed) |
| **Risk of Temporal Drift** | HIGH | LOW |
| **Time to First Insight** | 10 hours | 4 hours |
| **Total Cost** | $25-40 | $20-40 |
| **Decision Points** | 1 (end of process) | 3 (phased) |
| **Adaptability** | All-or-nothing | Adjust based on findings |
| **Generalizability** | Brittle (web state) | Stable (reasoning patterns) |

---

## Expert Recommendation Summary

> "Here is a concrete, prioritized action plan:
>
> **Phase 1: Diagnosis & Foundational Improvements (Do this now)**
> 1. Conduct the Qualitative Audit
> 2. Re-evaluate Your Data Split (stratified by web presence)
> 3. Run the Model A/B Test (Sonar vs Sonar Reasoning Pro)
>
> **Phase 2: Targeted Refinements (Do this next)**
> 4. Manual Prompt Refinement (Not Optimization)
> 5. Error Analysis
>
> **Phase 3: Advanced Strategies (Consider these later, if needed)**
> 6. Controlled Optimization with cached web search results
>
> This phased approach prioritizes understanding the problem and making high-leverage changes first, while deferring complex and potentially brittle optimizations."

---

## Next Steps

### Immediate Action (This Week)

1. **Review this analysis** with stakeholders
2. **Decide**: Proceed with phased plan or stick with original?
3. **If phased plan**: Start Phase 1 diagnostic (4 hours)
4. **If original plan**: Acknowledge risks and proceed with GEPA

### Phase 1 Execution (If Approved)

```bash
# Day 1-2: Manual audit (2 hours)
uv run python scripts/audit_web_presence.py

# Day 2-3: Quantitative analysis (2 hours)
uv run python scripts/calculate_web_presence_scores.py

# Day 3: Review results (1 hour)
# Decision: Proceed to Phase 2?
```

### Phase 2 Execution (If Phase 1 Confirms Hypothesis)

```bash
# Day 4: Model A/B test (3 hours)
uv run python scripts/ab_test_perplexity_models.py \
  --validation-subset 20 \
  --models sonar,sonar-reasoning-pro \
  --use-cached-data

# Day 4: Analyze results (1 hour)
# Decision: Which scenario (A/B/C)?
```

---

## Conclusion

The planned 65-example GEPA optimization may not address the root cause of performance limitations. Before investing 10 hours and $25-40 in traditional optimization:

1. **Validate the hypothesis**: Is web resource availability the confounding factor?
2. **Test the solution**: Does Sonar Reasoning Pro handle noisy data better?
3. **Apply manual refinements**: Can we achieve target without automated optimization?
4. **Only if needed**: Run cached-data GEPA for stable, generalizable optimization

**Key Advantage**: Phased approach provides early feedback and adapts based on findings, rather than committing to a single approach that may optimize for the wrong thing.

---

**Prepared By**: Claude Code with Gemini 2.5 Pro Expert Analysis
**Date**: 2025-11-18
**Status**: Awaiting approval to proceed with Phase 1
