# Phase 2: Perplexity Model Comparison Report

**Date**: 2025-11-18 23:49:24
**Purpose**: Establish baseline F1 score and select student model for Phase 3 GEPA optimization
**Phase**: 2.5 (Repeated CV for Variance Reduction)

---

## Executive Summary

This report details the comparison between Perplexity Sonar and Sonar Reasoning Pro
to establish a baseline F1 score and select a student model for Phase 3 GEPA optimization.

**Variance Reduction**: Using repeated CV to reduce std dev from 21% to <10% target

**Recommendation**: Insufficient evidence for a difference; use faster/cheaper Sonar
**Selected Model for Phase 3**: `perplexity/sonar`
**Established Baseline F1**: 60.22%

---

## Summary Results

| Model                 | Mean F1     | Std Dev F1 | 95% Confidence Interval |
|-----------------------|-------------|------------|-------------------------|
| Sonar                 | 60.22%      | 23.05% | [0.00%, 100.00%] |
| Sonar Reasoning Pro   | 50.62% | 26.27% | [0.00%, 100.00%] |

**Difference**: +-9.60% F1 (Sonar Reasoning Pro - Sonar)

---

## Statistical Analysis

### Method

- **Evaluation**: Stratified 10-fold × 10 repetitions with bootstrap resampling
- **Total Folds**: 100 (10 folds × 10 repetitions)
- **Test**: Wilcoxon signed-rank test (paired, non-parametric)
- **Hypothesis**: Sonar Pro F1 > Sonar F1 (one-sided test)
- **Bootstrap**: 1000 samples per fold for confidence interval estimation
- **Random State**: 42 (reproducible results)

### Results

- **p-value**: 1.0000
  - Interpretation: Not statistically significant (p≥0.05)
- **Effect Size (Cohen's d)**: -0.59
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
| 31    | 66.67%    | 66.67%      | +0.00%     |
| 32    | 66.67%    | 40.00%      | -26.67%     |
| 33    | 66.67%    | 57.14%      | -9.52%     |
| 34    | 40.00%    | 40.00%      | +0.00%     |
| 35    | 85.71%    | 66.67%      | -19.05%     |
| 36    | 66.67%    | 66.67%      | +0.00%     |
| 37    | 50.00%    | 50.00%      | +0.00%     |
| 38    | 0.00%    | 0.00%      | +0.00%     |
| 39    | 50.00%    | 50.00%      | +0.00%     |
| 40    | 100.00%    | 80.00%      | -20.00%     |
| 41    | 85.71%    | 85.71%      | +0.00%     |
| 42    | 66.67%    | 40.00%      | -26.67%     |
| 43    | 40.00%    | 40.00%      | +0.00%     |
| 44    | 66.67%    | 40.00%      | -26.67%     |
| 45    | 40.00%    | 40.00%      | +0.00%     |
| 46    | 66.67%    | 66.67%      | +0.00%     |
| 47    | 0.00%    | 0.00%      | +0.00%     |
| 48    | 50.00%    | 40.00%      | -10.00%     |
| 49    | 100.00%    | 100.00%      | +0.00%     |
| 50    | 80.00%    | 50.00%      | -30.00%     |
| 51    | 66.67%    | 66.67%      | +0.00%     |
| 52    | 85.71%    | 85.71%      | +0.00%     |
| 53    | 66.67%    | 0.00%      | -66.67%     |
| 54    | 85.71%    | 85.71%      | +0.00%     |
| 55    | 40.00%    | 0.00%      | -40.00%     |
| 56    | 40.00%    | 40.00%      | +0.00%     |
| 57    | 50.00%    | 50.00%      | +0.00%     |
| 58    | 50.00%    | 40.00%      | -10.00%     |
| 59    | 80.00%    | 80.00%      | +0.00%     |
| 60    | 50.00%    | 50.00%      | +0.00%     |
| 61    | 66.67%    | 66.67%      | +0.00%     |
| 62    | 66.67%    | 66.67%      | +0.00%     |
| 63    | 66.67%    | 33.33%      | -33.33%     |
| 64    | 66.67%    | 40.00%      | -26.67%     |
| 65    | 40.00%    | 40.00%      | +0.00%     |
| 66    | 66.67%    | 66.67%      | +0.00%     |
| 67    | 50.00%    | 50.00%      | +0.00%     |
| 68    | 80.00%    | 80.00%      | +0.00%     |
| 69    | 80.00%    | 50.00%      | -30.00%     |
| 70    | 50.00%    | 50.00%      | +0.00%     |
| 71    | 40.00%    | 33.33%      | -6.67%     |
| 72    | 66.67%    | 66.67%      | +0.00%     |
| 73    | 66.67%    | 40.00%      | -26.67%     |
| 74    | 66.67%    | 66.67%      | +0.00%     |
| 75    | 85.71%    | 85.71%      | +0.00%     |
| 76    | 85.71%    | 85.71%      | +0.00%     |
| 77    | 50.00%    | 50.00%      | +0.00%     |
| 78    | 50.00%    | 0.00%      | -50.00%     |
| 79    | 50.00%    | 0.00%      | -50.00%     |
| 80    | 50.00%    | 50.00%      | +0.00%     |
| 81    | 85.71%    | 85.71%      | +0.00%     |
| 82    | 66.67%    | 57.14%      | -9.52%     |
| 83    | 40.00%    | 40.00%      | +0.00%     |
| 84    | 85.71%    | 66.67%      | -19.05%     |
| 85    | 0.00%    | 0.00%      | +0.00%     |
| 86    | 66.67%    | 0.00%      | -66.67%     |
| 87    | 80.00%    | 80.00%      | +0.00%     |
| 88    | 80.00%    | 80.00%      | +0.00%     |
| 89    | 50.00%    | 50.00%      | +0.00%     |
| 90    | 50.00%    | 50.00%      | +0.00%     |
| 91    | 40.00%    | 40.00%      | +0.00%     |
| 92    | 85.71%    | 85.71%      | +0.00%     |
| 93    | 85.71%    | 85.71%      | +0.00%     |
| 94    | 66.67%    | 40.00%      | -26.67%     |
| 95    | 66.67%    | 0.00%      | -66.67%     |
| 96    | 66.67%    | 66.67%      | +0.00%     |
| 97    | 80.00%    | 66.67%      | -13.33%     |
| 98    | 0.00%    | 0.00%      | +0.00%     |
| 99    | 80.00%    | 80.00%      | +0.00%     |
| 100    | 0.00%    | 0.00%      | +0.00%     |


### Repetition-by-Repetition Breakdown

Showing mean F1 per repetition (each repetition uses a different random shuffle):

| Repetition | Sonar Mean F1 | Sonar Pro Mean F1 | Difference |
|------------|---------------|-------------------|------------|
| 1          | 56.14%         | 48.90%           | -7.24%      |
| 2          | 62.57%         | 53.05%           | -9.52%      |
| 3          | 61.14%         | 52.90%           | -8.24%      |
| 4          | 59.24%         | 51.71%           | -7.52%      |
| 5          | 59.57%         | 50.24%           | -9.33%      |
| 6          | 61.48%         | 49.81%           | -11.67%      |
| 7          | 63.33%         | 54.33%           | -9.00%      |
| 8          | 61.14%         | 47.81%           | -13.33%      |
| 9          | 60.48%         | 50.95%           | -9.52%      |
| 10          | 57.14%         | 46.48%           | -10.67%      |


**Repetition Variance Analysis:**
- Sonar std dev across repetitions: 2.26%
- Sonar Pro std dev across repetitions: 2.47%

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
- **Baseline F1**: 60.22%
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