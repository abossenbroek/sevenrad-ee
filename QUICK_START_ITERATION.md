# Quick Start: Iterative Refinement

## TL;DR - Execute Iteration 1 Now

```bash
# Run 65-example optimization (8-10 hours)
mkdir -p results/iteration_1_65examples
uv run python notebooks/optimize_greenhouse_detector.py \
  --output-dir results/iteration_1_65examples \
  --num-examples 65 \
  --budget medium

# After completion, analyze failures
uv run python scripts/analyze_optimization_failures.py \
  --results-dir results/iteration_1_65examples \
  --output results/iteration_1_65examples/failure_analysis.json

# Generate collection plan for iteration 2
uv run python scripts/generate_targeted_companies.py \
  --analysis results/iteration_1_65examples/failure_analysis.json \
  --current-count 65 \
  --output data/collection_plan_iteration2.json
```

---

## What Each Command Does

### 1. Optimization (~8-10 hours)
Runs GEPA optimization with cross-validation on 65 examples.

**Expected output**:
- `results/iteration_1_65examples/optimized_program.json`
- `results/iteration_1_65examples/cv_results.json`
- Mean F1: ~75-78%

---

### 2. Failure Analysis (~30 seconds)
Identifies patterns in misclassified companies.

**Expected output**:
```
⚠️  Misclassification Patterns
  No Evidence: 27 companies
  Contradictory: 8 companies
  Weak Positive: 12 companies

💡 Recommendations
  1. Collect 35-50 targeted examples
  2. Prioritize well-documented companies
  3. Focus on underrepresented crops
```

---

### 3. Collection Planning (~30 seconds)
Creates targeted company list based on failure patterns.

**Expected output**:
```json
{
  "target_count": 35,
  "crop_targets": {
    "Roses": 6,
    "Orchids": 5,
    "Tomatoes": 5
  },
  "search_strategies": [
    {
      "strategy": "Search for Cucumbers growers",
      "keywords": ["cucumbers kwekerij nederland"],
      "target_count": 4
    }
  ]
}
```

---

## After Iteration 1 Completes

### Review Results
```bash
# Check performance
cat results/iteration_1_65examples/cv_results.json

# View failure analysis
uv run python scripts/analyze_optimization_failures.py \
  --results-dir results/iteration_1_65examples
```

### Prepare Iteration 2
```bash
# 1. Review collection plan
cat data/collection_plan_iteration2.json

# 2. Create companies to research (manual)
# Based on search strategies in plan, create:
#   data/companies_to_research_iteration2.json

# 3. Run batch research
uv run python -m sevenrad_ee.operations.batch_company_research \
  --input data/companies_to_research_iteration2.json \
  --output-dir data/research

# 4. Verify count
uv run python scripts/count_classifications.py
# Should show ~100 total companies

# 5. Run iteration 2 optimization
mkdir -p results/iteration_2_100examples
uv run python notebooks/optimize_greenhouse_detector.py \
  --output-dir results/iteration_2_100examples \
  --num-examples 100
```

---

## Automated Alternative

Run entire iterative process automatically (pauses for manual data collection):

```bash
uv run python scripts/iterative_refinement.py \
  --max-iterations 5 \
  --target-f1 0.95
```

When prompted:
1. Review collection plan
2. Research companies
3. Press Enter to continue

---

## Monitoring Progress

### Check optimization status (while running)
```bash
tail -f results/iteration_1_65examples/optimization.log
```

Look for:
- `Iteration N: F1 = X.XX%` (progress indicators)
- `Best program: F1 = X.XX%` (current best)
- Warnings or errors

### Estimate completion time
- Iterations 1-10: ~3 hours (exploration)
- Iterations 11-20: ~3 hours (refinement)
- Iterations 21+: ~2 hours (fine-tuning)
- CV evaluation: ~2 hours
- **Total**: ~8-10 hours

---

## Decision Tree

```
Run iteration 1
  ↓
IF F1 ≥ 80%:
  → Collect 20-30 examples → 85-90% target ✅

ELIF F1 = 75-80%:
  → Collect 35-50 examples → 82-85% target 🎯
  → Run iteration 2

ELIF F1 = 70-75%:
  → Investigate quality issues
  → Collect 35-50 high-quality examples
  → Run iteration 2

ELSE (F1 < 70%):
  → Review evidence classification logic
  → May need architectural changes ⚠️
```

---

## Files Created

### Optimization Results
- `results/iteration_1_65examples/optimized_program.json` - Best prompt
- `results/iteration_1_65examples/cv_results.json` - Performance metrics
- `results/iteration_1_65examples/optimization.log` - Detailed log

### Analysis Files
- `results/iteration_1_65examples/failure_analysis.json` - Failure patterns
- `data/collection_plan_iteration2.json` - Next collection plan

### Future Iterations
- `results/iteration_2_100examples/` - Second iteration
- `results/iteration_3_120examples/` - Third iteration (if needed)
- `results/final_comparison.md` - Multi-iteration comparison

---

## Support Resources

- **Full Documentation**: `docs/ITERATIVE_REFINEMENT.md`
- **Detailed Plan**: `GEPA_NEXT_STEP.md`
- **Data Requirements**: Run `scripts/estimate_required_datapoints.py`

---

**Ready to start?** Run the first command block at the top of this file.

**Estimated time to first results**: 8-10 hours
**Expected F1**: 75-78%
**Next step**: Targeted data collection for iteration 2
