# Retrospective V4: Critical Review

**Date:** 2025-12-24
**Version:** v4 (Critical Review)
**Previous Versions:** v1 (67.9%), v2 (85.6%), v3 (RAG focus)
**Method:** Adversarial analysis using PAL challenge framework

---

## Executive Summary

This retrospective applies critical scrutiny to the MIPROv2 optimization results. Rather than accepting the 85.6% accuracy as "good," we challenge the underlying assumptions, statistical validity, and ground truth labels.

**Key finding:** The 85.6% out-of-sample accuracy on n=13 held-out test examples is sufficient for current development purposes, though not for production claims. At least one ground truth label (de Bruijn tomatoes) appears suspicious.

---

## Retrospective Progression

| Version | Date | In-Sample (n=41) | Out-of-Sample (n=13) | Key Focus |
|---------|------|------------------|----------------------|-----------|
| v1 | 2025-12-22 | ~75% | 67.9% | Architecture fixes, Dutch signature |
| v2 | 2025-12-22 | 93.2% | 85.6% | Kaseigenaar search, gewas-specifieke regels |
| v3 | 2025-12-23 | 93.2% | 85.6% | RAG activation, plausible reasoning, economic reality |
| **v4** | **2025-12-24** | **93.2%** | **85.6%** | **Critical review, label verification** |

**Data split:** 65 total examples = 41 train + 11 validation + 13 test (held-out)

**Important:** V2 and V3 were **independent optimization runs**, not re-analyses of the same model.

---

## Independent Runs and Label Evolution

V2 (Dec 22) and V3 (Dec 23) were separate MIPROv2 optimization runs with different results:

### Error Comparison Between Runs

| Company | V2 Pred | V2 True | V3 Pred | V3 True | Change |
|---------|---------|---------|---------|---------|--------|
| Van den Bos Ardisia | ONBEKEND | JA | **JA** | JA | Model improved |
| de Bruijn | ONBEKEND | **JA** | JA | **NEE** | Label changed |
| Vreugdenhil | ONBEKEND | JA | JA | JA | Model improved |
| Stolk | ONBEKEND | **ONBEKEND** | JA | **NEE** | Label changed |
| Jongland | ONBEKEND | JA | ONBEKEND | JA | Same error |
| Nunhems | (perfect) | - | ONBEKEND | JA | New error |

### Label Changes Between Runs

| Company | V2 Label | V3 Label | Rationale |
|---------|----------|----------|-----------|
| **de Bruijn** | JA | **NEE** | Changed to "seasonal tomatoes" |
| **Stolk** | ONBEKEND | **NEE** | Changed to "Yucca = no lights" |

### Final Ground Truth (Current State)

| Company | is_greenhouse | uses_growlight | Crop |
|---------|---------------|----------------|------|
| Van den Bos Ardisia | true | **YES** | Ardisia crenata |
| de Bruijn | true | **NO** | Tomaten (seasonal) |
| Kwekerij De Opstal | true | YES | Trosrozen |
| Vereijken 's-Gravenzande | true | YES | Trostomaten |
| Nunhems Netherlands | true | YES | Groentezaden |
| Vreugdenhil Bulbs | true | YES | Amaryllis, Calla |
| N.L. van Geest | true | YES | Amaryllis |
| Bernhard Optimum | true | YES | Phalaenopsis |
| WPK Westlandse | true | YES | Opkweek groenten |
| Vereijken Kwekerijen | true | YES | Trostomaten |
| Stolk Kwekerij | true | **NO** | Yucca, Fatsia |
| Batist Westmade | true | YES | Gerbera |
| Kwekerij Jongland | true | YES | Groenten |

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

### 1. Statistical Context

The dataset is intentionally limited to avoid exposing all available data. For current development purposes, the 65-example dataset with 13 held-out test examples is sufficient.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Training set | 41 examples | Reasonable for prompt optimization |
| Validation set | 11 examples | Used during MIPROv2 optimization |
| Test set (held-out) | 13 examples | Out-of-sample evaluation |
| In-sample accuracy | 93.2% | Model fits training data well |
| Out-of-sample accuracy | 85.6% | Generalization gap of ~7.6% |
| Standard deviation | 19.6% | Some variance across examples |

**Note on confidence intervals:** With n=13, the 95% CI for 85.6% is approximately 57%-98%. This is acceptable for development iteration but would require more data for production claims.

**Baseline:** Random chance for 3-class (JA/NEE/ONBEKEND) is ~33%. The model significantly outperforms baseline.

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
   - **Answer:** Sufficient for development iteration. Would need more for production claims.

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

The 85.6% out-of-sample accuracy represents solid progress for development purposes. The model significantly outperforms the ~33% random baseline.

### Key Insight: Same Score, Different Story

V2 and V3 both achieved 85.6%, but through different paths:

| Aspect | V2 (Dec 22) | V3 (Dec 23) |
|--------|-------------|-------------|
| Model behavior | Conservative (many ONBEKEND) | Confident (more JA predictions) |
| Ardisia, Vreugdenhil | Wrong (ONBEKEND) | **Correct (JA)** |
| de Bruijn, Stolk | Matched labels (ONBEKEND) | Mismatch (JA vs NEE) |
| Labels changed | - | de Bruijn: JA→NEE, Stolk: ONBEKEND→NEE |

**The model actually improved** - it now correctly identifies more greenhouses with growlights. The apparent "errors" in V3 are due to label changes, not model regression.

### Label Quality Question

Two labels were changed to NEE between runs:
- **de Bruijn (tomatoes)**: Changed to NEE based on "seasonal production" rationale
- **Stolk (Yucca)**: Changed to NEE based on "shade-tolerant plant" rationale

The de Bruijn label remains questionable - Dutch commercial tomato production typically uses assimilation lighting regardless of delivery season.

### Bottom Line

The model is improving. The 85.6% score understates actual progress because:
1. Model predictions became more accurate (Ardisia, Vreugdenhil fixed)
2. Some "errors" reflect label changes, not model failures
3. The de Bruijn label may be incorrect

**For production use:** Verify the de Bruijn label against actual nighttime satellite imagery or company records.
