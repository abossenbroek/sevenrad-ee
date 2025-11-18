# Phase 2: Perplexity Model Comparison Report

**Date**: 2025-11-18 18:46:31
**Purpose**: Establish baseline F1 score and select student model for Phase 3 GEPA optimization
**Phase**: 2.5 (Repeated CV for Variance Reduction)

---

## Executive Summary

This report details the comparison between Perplexity Sonar and Sonar Reasoning Pro
to establish a baseline F1 score and select a student model for Phase 3 GEPA optimization.

**Variance Reduction**: Using repeated CV to reduce std dev from 21% to <10% target

**Recommendation**: Insufficient evidence for a difference; use faster/cheaper Sonar
**Selected Model for Phase 3**: `perplexity/sonar`
**Established Baseline F1**: 59.95%

---

## Summary Results

| Model                 | Mean F1     | Std Dev F1 | 95% Confidence Interval |
|-----------------------|-------------|------------|-------------------------|
| Sonar                 | 59.95%      | 23.20% | [0.00%, 100.00%] |
| Sonar Reasoning Pro   | 51.62% | 24.13% | [0.00%, 100.00%] |

**Difference**: +-8.33% F1 (Sonar Reasoning Pro - Sonar)

---

## Statistical Analysis

### Method

- **Evaluation**: Stratified 10-fold × 3 repetitions with bootstrap resampling
- **Total Folds**: 30 (10 folds × 3 repetitions)
- **Test**: Wilcoxon signed-rank test (paired, non-parametric)
- **Hypothesis**: Sonar Pro F1 > Sonar F1 (one-sided test)
- **Bootstrap**: 1000 samples per fold for confidence interval estimation
- **Random State**: 42 (reproducible results)

### Results

- **p-value**: 0.9978
  - Interpretation: Not statistically significant (p≥0.05)
- **Effect Size (Cohen's d)**: -0.65
  - Interpretation: Small effect (d<0.3)

### Fold-by-Fold Breakdown

| Fold | Sonar F1 | Sonar Reasoning Pro F1 | Difference |
|------|----------|------------------------|------------|
| 1    | 40.00%    | 40.00%      | +0.00%     |
| 2    | 66.67%    | 40.00%      | -26.67%     |
| 3    | 85.71%    | 85.71%      | +0.00%     |
| 4    | 85.71%    | 66.67%      | -19.05%     |
| 5    | 66.67%    | 66.67%      | +0.00%     |
| 6    | 66.67%    | 40.00%      | -26.67%     |
| 7    | 100.00%    | 100.00%      | +0.00%     |
| 8    | 0.00%    | 0.00%      | +0.00%     |
| 9    | 0.00%    | 0.00%      | +0.00%     |
| 10    | 50.00%    | 50.00%      | +0.00%     |
| 11    | 40.00%    | 0.00%      | -40.00%     |
| 12    | 40.00%    | 40.00%      | +0.00%     |
| 13    | 66.67%    | 66.67%      | +0.00%     |
| 14    | 66.67%    | 66.67%      | +0.00%     |
| 15    | 85.71%    | 57.14%      | -28.57%     |
| 16    | 66.67%    | 40.00%      | -26.67%     |
| 17    | 80.00%    | 80.00%      | +0.00%     |
| 18    | 50.00%    | 50.00%      | +0.00%     |
| 19    | 50.00%    | 50.00%      | +0.00%     |
| 20    | 80.00%    | 80.00%      | +0.00%     |
| 21    | 66.67%    | 40.00%      | -26.67%     |
| 22    | 85.71%    | 66.67%      | -19.05%     |
| 23    | 85.71%    | 85.71%      | +0.00%     |
| 24    | 40.00%    | 40.00%      | +0.00%     |
| 25    | 66.67%    | 40.00%      | -26.67%     |
| 26    | 66.67%    | 66.67%      | +0.00%     |
| 27    | 50.00%    | 50.00%      | +0.00%     |
| 28    | 50.00%    | 50.00%      | +0.00%     |
| 29    | 50.00%    | 40.00%      | -10.00%     |
| 30    | 50.00%    | 50.00%      | +0.00%     |


### Repetition-by-Repetition Breakdown

Showing mean F1 per repetition (each repetition uses a different random shuffle):

| Repetition | Sonar Mean F1 | Sonar Pro Mean F1 | Difference |
|------------|---------------|-------------------|------------|
| 1          | 56.14%         | 48.90%           | -7.24%      |
| 2          | 62.57%         | 53.05%           | -9.52%      |
| 3          | 61.14%         | 52.90%           | -8.24%      |


**Repetition Variance Analysis:**
- Sonar std dev across repetitions: 3.38%
- Sonar Pro std dev across repetitions: 2.35%

This shows the stability of each model across different data shuffles.

---

## Decision Criteria

Following expert-validated decision criteria from Phase 2 strategy:

1. **Strong Evidence** (p<0.05 AND effect_size>0.5):
   - Select Sonar Reasoning Pro
   - High confidence in meaningful improvement

2. **Moderate Evidence** (effect_size>0.3):
   - Select Sonar Reasoning Pro if budget allows
   - Noticeable but not definitively significant improvement

3. **Insufficient Evidence** (all other cases):
   - Select cheaper/faster Sonar
   - Models are statistically indistinguishable

**This comparison**: Insufficient evidence for a difference; use faster/cheaper Sonar

---

## Conclusion

- **Selected Model**: perplexity/sonar
- **Baseline F1**: 59.95%
- **Next Step**: Proceed to Phase 3 (GEPA optimization) using selected model

### Notes

- With n=65 and 5-fold CV, each test set has ~13 examples
- Bootstrap provides confidence intervals despite small sample size
- Stratification ensures both classes represented in each fold
- Paired test increases statistical power by controlling for data variation

---

## Methodology Reference

For full details on the Phase 2 approach, see:
- `OPTIMIZATION_STRATEGY.md` - Complete 4-phase optimization strategy
- `scripts/evaluate_with_cv.py` - Statistical evaluation harness
- `src/sevenrad_ee/ai/data_utils.py` - Data loading utilities

---

*Report generated by Phase 2: Baseline & Model Selection*
*Part of DSPy RAG Optimization Strategy for Greenhouse Classification*