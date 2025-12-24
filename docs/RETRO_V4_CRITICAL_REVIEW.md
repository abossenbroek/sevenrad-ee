# Retrospective V4: Critical Review

**Date:** 2025-12-24
**Version:** v4 (Critical Review)
**Previous Versions:** v1 (67.9%), v2 (85.6%), v3 (RAG focus)
**Method:** Adversarial analysis using PAL challenge framework

---

## Executive Summary

This retrospective applies critical scrutiny to the MIPROv2 optimization results. Rather than accepting the 85.6% accuracy as "good," we challenge the underlying assumptions, statistical validity, and ground truth labels.

**Key finding:** The claimed 85.6% accuracy is statistically meaningless with n=13, and at least one ground truth label (de Bruijn tomatoes) appears suspicious.

---

## Retrospective Progression

| Version | Date | Score | Key Focus |
|---------|------|-------|-----------|
| v1 | 2025-12-22 | 67.9% | Architecture fixes, Dutch signature |
| v2 | 2025-12-22 | 85.6% | Kaseigenaar search, gewas-specifieke regels |
| v3 | 2025-12-23 | 85.6% | RAG activation, plausible reasoning, economic reality |
| **v4** | **2025-12-24** | 85.6% | **Critical review, statistical validity, label verification** |

---

## Changes from Previous Retrospectives

### From Retro 3 (Plausible Reasoning)

| Aspect | Retro 3 | Retro 4 |
|--------|---------|---------|
| **Focus** | How to improve RAG activation | Is the evaluation valid at all? |
| **Assumption** | Labels are correct, model needs fixing | Labels may be wrong |
| **Sample size** | Not questioned | Identified as critical flaw |
| **Confidence** | High (target met) | Low (insufficient evidence) |

**New in v4:**
- Statistical validity challenge (n=13 is too small)
- Ground truth label questioning (de Bruijn tomatoes)
- Error cost analysis framework
- Confidence interval analysis (57%-98% at 95% CI)

### From Retro 2 (85.6% Achievement)

| Aspect | Retro 2 | Retro 4 |
|--------|---------|---------|
| **Tone** | Celebratory (target met!) | Skeptical (is this real?) |
| **Question** | "How do we fix the 3 errors?" | "Are those actually errors?" |
| **Metric focus** | Accuracy | Statistical validity |
| **Label trust** | High | Questioned |

**Paradigm shift:** From "how to fix errors" to "are the labels correct?"

---

## Critical Findings

### 1. Statistical Validity Concerns

The test set of n=13 is insufficient for meaningful conclusions.

| Metric | Value | Problem |
|--------|-------|---------|
| Sample size | 13 | Far below minimum for reliable metrics |
| 95% Confidence Interval | 57% - 98% | Range so wide it's nearly meaningless |
| Standard deviation | 19.6% | High variance indicates unreliable signal |
| Training vs Test gap | 93.18% vs 85.6% | Suggests overfitting |

**Minimum sample size:** For reliable classification metrics, need 100-200 samples per class.

**Baseline missing:** What is random chance for 3-class (JA/NEE/ONBEKEND)? Approximately 33%. The improvement over baseline should be the metric, not absolute accuracy.

### 2. Ground Truth Label Questions

| Case | Current Label | Model Prediction | Challenge | Verdict |
|------|---------------|------------------|-----------|---------|
| **de Bruijn (tomatoes)** | NEE | JA | **Suspicious** - Dutch commercial tomato production almost universally uses assimilation lighting | VERIFY |
| **Stolk (Yucca)** | NEE | JA | Plausible - Yucca is shade-tolerant, many growers skip supplemental lighting | KEEP |
| **Jongland** | JA | ONBEKEND | Model uncertainty may indicate genuinely ambiguous evidence | REVIEW |

**de Bruijn analysis:**
- Dutch commercial tomato greenhouses typically use 100-150 W/m² HPS or LED lighting
- "Seasonal-only" tomato production without lights is economically disadvantageous in NL
- The claim of "maart-oktober" being a reason for no lights is questionable
- This is a **probable labeling error**, not a model error

### 3. Horticultural Validation

**"Seasonal tomatoes" claim:**
- Retro 3 proposed: "seizoensgebonden = no growlights"
- **Challenge:** Dutch tomato production in Westland is almost synonymous with year-round lighting
- Commercial growers cannot afford seasonal-only operation due to fixed costs
- "maart-oktober" likely refers to delivery period, not operational period

**"Yucca = no growlights" assumption:**
- Yucca elephantipes is indeed shade-tolerant
- However, faster commercial production often uses supplemental lighting
- This is a probabilistic assumption, not a definitive rule
- Some Yucca growers DO use lighting for faster/better growth

### 4. Missing Metrics

The current evaluation lacks critical metrics for production readiness:

| Missing Metric | Why It Matters |
|----------------|----------------|
| Precision/Recall by class | Different error types have different costs |
| Baseline comparison | What does random guessing achieve? |
| Confusion matrix | Which classes are confused with which? |
| Error cost analysis | False positive vs false negative impact |

### 5. Error Cost Analysis

The cost of errors depends on the use case:

| Use Case | Worse Error Type | Rationale |
|----------|------------------|-----------|
| Energy auditing | False Negative | Missing actual energy users |
| Light pollution studies | Both equally bad | Need complete picture |
| Subsidy fraud detection | False Positive | Wrongful accusations are costly |
| Market research | False Negative | Undercounting market size |

**Current evaluation treats all errors equally** - this may not reflect real-world requirements.

---

## Challenge Summary

The PAL challenge framework was applied to scrutinize the optimization results:

### Questions Raised

1. **Is 85.6% "good" accuracy?**
   - **Answer:** Cannot determine with n=13. Confidence interval is too wide.

2. **Are the error classifications correct?**
   - **Answer:** At least one (de Bruijn) appears to be a labeling error, not a model error.

3. **Is the sample size statistically meaningful?**
   - **Answer:** No. Need 100+ samples for reliable conclusions.

4. **Is "seasonal tomatoes" logic sound?**
   - **Answer:** Questionable. Dutch commercial tomatoes are typically lit year-round.

5. **Why would Yucca indicate no growlights?**
   - **Answer:** Probabilistic assumption, not a rule. Some Yucca growers do use lights.

6. **What's the cost of false positives vs false negatives?**
   - **Answer:** Undefined. Depends on use case.

---

## Open Questions

These questions remain unresolved and require investigation:

1. **Is the de Bruijn label correct?**
   - How was this label determined?
   - Can it be verified with satellite/nighttime imagery?

2. **What is the source of ground truth labels?**
   - Manual labeling? Domain expert? Web research?
   - What is the expected label noise rate?

3. **What is acceptable accuracy for production use?**
   - Depends on use case and error costs
   - 85% may be excellent or insufficient

4. **Should we report precision/recall instead of accuracy?**
   - Class imbalance makes accuracy misleading
   - Precision/recall per class more informative

5. **What is the baseline (random chance)?**
   - For 3-class: ~33%
   - For 2-class (ignoring ONBEKEND): ~50%

---

## Lessons Learned

### What We Got Right in Previous Retros
- Architecture improvements (RAG integration) were real gains
- Domain knowledge additions (gewas-specifieke regels) helped
- Economic reality principle is valid

### What We Missed
- Statistical validity of small test set
- Questioning ground truth labels
- Error cost analysis
- Baseline comparison

### Meta-Lesson
**Celebrate metrics cautiously.** The 85.6% number was presented as a success, but deeper analysis reveals it may be:
- Statistically unreliable (small sample)
- Partially based on incorrect labels
- Missing context (baseline, error costs)

---

## Appendix: PAL Challenge Analysis

The following challenge was posed to scrutinize the findings:

> MIPROv2 Optimization for greenhouse detection achieved 85.6% accuracy with:
> - 8/13 perfect scores (100%)
> - 2/13 partial scores (62.5-85%)
> - 3/13 errors (55%)
>
> Challenge critically:
> 1. Is 85.6% actually "good" for this use case?
> 2. Are the error classifications correct?
> 3. Is n=13 statistically meaningful?
> 4. Is "seasonal tomatoes" logic sound?
> 5. Why would Yucca = no growlights?
> 6. What's the cost of false positives vs false negatives?

**Result:** Multiple assumptions were invalidated or questioned, leading to this critical retrospective.

---

## Conclusion

The 85.6% accuracy claimed in Retro 2 and maintained through Retro 3 is not a reliable metric. The sample size is too small, at least one ground truth label is suspicious, and the error cost framework is missing.

This does not mean the optimization work was wasted - the architecture improvements and domain knowledge additions are valuable. However, **production deployment decisions should not be based on n=13 test results**.

**Bottom line:** We have a potentially good model, but we lack the statistical evidence to prove it.
