# DSPy Greenhouse Detection Optimization Report (19 Examples)

## Results Summary

### Cross-Validation Performance
- **Mean F1:** 63.39% ± 6.65%
- **Train F1:** 58.23%
- **Overfitting Gap:** -5.16%

### Known Negatives Test
- **False Positive Rate:** 0/6 (0%)

## Success Criteria

❌ Mean F1 ≥ 95%
❌ Std F1 ≤ 3%
✅ Overfitting Gap < 10%
✅ 0% False Positives on Known Negatives

## Baseline Assessment (19 Examples)

This baseline uses only 19 examples to establish expected performance with limited data.

**Expected Behavior:**
- High overfitting (gap >15%) due to small dataset
- High variance (std >5%) across CV folds
- Model memorizes training examples

**Actual Results:**
- Overfitting Gap: -5.16% (ACCEPTABLE)
- Variance: 6.65% (HIGH)

**Next Steps:**
Run 40-example pipeline to compare:
- Expected improvement: Gap <10%, Std <3%
- Better generalization with more diverse examples

## Deployment

Optimized model saved to: `results/19examples/optimized_program.json`

Load model in production:
```python
optimized_program = GreenhouseDetector()
optimized_program.load("results/19examples/optimized_program.json")
```

🤖 Generated with Claude Code
