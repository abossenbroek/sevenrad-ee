# Iterative Refinement Workflow

**Purpose**: Systematic approach to reach 95% F1 target through data-driven iteration cycles.

---

## Overview

The iterative refinement workflow enables automatic improvement through:

1. **Optimize** → Run GEPA optimization on current dataset
2. **Analyze** → Identify failure patterns and weak areas
3. **Target** → Generate focused company list for next batch
4. **Collect** → Research targeted companies
5. **Repeat** → Until target performance achieved

---

## Quick Start

### Option 1: Automated Orchestration (Recommended)

Run the full iterative refinement process:

```bash
uv run python scripts/iterative_refinement.py \
  --max-iterations 5 \
  --target-f1 0.95 \
  --start-iteration 1
```

**Features**:
- Automatically runs optimization, analysis, and planning
- Prompts for manual data collection between iterations
- Generates comparative reports
- Stops when target reached or max iterations hit

**Timeline**: 2-3 weeks (3-4 iterations expected)

---

### Option 2: Manual Step-by-Step

More control over each step of the process:

#### Iteration 1: Baseline (65 examples)

```bash
# 1. Run optimization
mkdir -p results/iteration_1_65examples
uv run python notebooks/optimize_greenhouse_detector.py \
  --output-dir results/iteration_1_65examples \
  --num-examples 65

# 2. Analyze failures
uv run python scripts/analyze_optimization_failures.py \
  --results-dir results/iteration_1_65examples \
  --output results/iteration_1_65examples/failure_analysis.json

# 3. Generate collection plan
uv run python scripts/generate_targeted_companies.py \
  --analysis results/iteration_1_65examples/failure_analysis.json \
  --current-count 65 \
  --output data/collection_plan_iteration2.json

# 4. Review plan and collect data (manual)
cat data/collection_plan_iteration2.json
# ... create companies_to_research_iteration2.json based on plan
uv run python -m sevenrad_ee.operations.batch_company_research

# 5. Proceed to Iteration 2
```

#### Iteration 2: Targeted Expansion (100-115 examples)

```bash
# Repeat steps 1-5 with new dataset size
```

---

## Tools Reference

### 1. `analyze_optimization_failures.py`

Analyzes CV results to identify systematic weaknesses.

**Usage**:
```bash
uv run python scripts/analyze_optimization_failures.py \
  --results-dir results/iteration_1_65examples \
  --data-dir data/research \
  --output results/iteration_1_65examples/failure_analysis.json
```

**Outputs**:
- Fold performance analysis
- Misclassification patterns (weak evidence, contradictory, no evidence)
- Evidence quality metrics
- Underrepresented patterns (crops, locations)
- Actionable recommendations

**Example Output**:
```
📊 Performance Summary
  Mean F1: 77.5%
  Std F1: 5.2%

⚠️  Misclassification Patterns
  No Evidence: 27 companies
  Contradictory: 8 companies

💡 Recommendations
  1. ⚙️  Mean F1 (77.5%) progressing - collect 20-30 targeted examples
  2. 📚 27 companies with no evidence - prioritize well-documented companies
  3. 🌱 Underrepresented crops: Cucumbers, Strawberries - collect more examples
```

---

### 2. `generate_targeted_companies.py`

Creates focused company list based on failure analysis.

**Usage**:
```bash
uv run python scripts/generate_targeted_companies.py \
  --analysis results/iteration_1_65examples/failure_analysis.json \
  --current-count 65 \
  --output data/collection_plan_iteration2.json
```

**Outputs**:
- Target collection count (based on performance)
- Crop type distribution targets
- Location distribution targets
- Quality criteria (evidence strength, signal clarity)
- Search strategies with keywords

**Example Output**:
```
🎯 Collection Target: 35 companies
  Rationale: Significant expansion for 85%+ target

🌱 Crop Type Targets
  Roses: 6 (17%)
  Orchids: 5 (14%)
  Tomatoes: 5 (14%)

🔍 Search Strategies
  1. Search for Cucumbers growers in Netherlands
     Target: 4 companies
     Keywords:
       - cucumbers kwekerij nederland
       - komkommer led belichting
```

---

### 3. `compare_iterations.py`

Generates side-by-side comparison across iterations.

**Usage**:
```bash
uv run python scripts/compare_iterations.py \
  results/iteration_1_65examples \
  results/iteration_2_100examples \
  results/iteration_3_120examples \
  --output results/final_comparison.md
```

**Outputs**:
- Performance comparison table
- Improvements from baseline
- Learning curve analysis
- Marginal efficiency (F1 gained per example)
- Markdown report

**Example Output**:
```
📊 Performance Comparison
| Iteration | Examples | Mean F1 | Status |
|-----------|----------|---------|--------|
| iteration_1 | 65 | 77.0% | 🟡 Good |
| iteration_2 | 100 | 83.5% | 🟢 Excellent |
| iteration_3 | 120 | 87.2% | 🟢 Excellent |

📈 Improvements from Baseline
  iteration_2 vs iteration_1:
    Dataset size: 65 → 100 (1.5x, +35)
    Mean F1: 77.0% → 83.5% (+6.5%, +8.4%)

📉 Learning Curve Analysis
  iteration_1 → iteration_2: +35 examples → +6.5% F1 (0.19% F1/example)
  iteration_2 → iteration_3: +20 examples → +3.7% F1 (0.19% F1/example)
```

---

### 4. `iterative_refinement.py`

Master orchestrator for automated iteration cycles.

**Usage**:
```bash
uv run python scripts/iterative_refinement.py \
  --max-iterations 5 \
  --target-f1 0.95 \
  --start-iteration 1
```

**Features**:
- Runs optimization automatically
- Analyzes failures after each iteration
- Generates collection plans
- Prompts for manual data collection
- Generates final comparative report
- Stops when target reached or minimal improvement

**Decision Logic**:
```python
IF mean_f1 >= target_f1:
  STOP - "Target achieved"

ELIF iteration >= max_iterations:
  STOP - "Max iterations reached"

ELIF improvement < 2%:
  STOP - "Minimal improvement - new approach needed"

ELSE:
  CONTINUE - "Gap to target: X%"
```

---

## Expected Iteration Timeline

### Iteration 1: Baseline (65 examples)
- **Optimization**: 8-10 hours
- **Analysis**: 5 minutes
- **Planning**: 2 minutes
- **Data Collection**: 4-6 hours (35-50 companies)
- **Total**: ~1-2 days

**Expected Results**:
- Mean F1: 75-78%
- Collection target: 35-50 companies
- Next dataset: 100-115 examples

---

### Iteration 2: Targeted Expansion (100 examples)
- **Optimization**: 10-12 hours
- **Analysis**: 5 minutes
- **Planning**: 2 minutes
- **Data Collection**: 2-4 hours (20-30 companies)
- **Total**: ~1-2 days

**Expected Results**:
- Mean F1: 82-85%
- Collection target: 20-30 companies
- Next dataset: 120-130 examples

---

### Iteration 3: Polish (120 examples)
- **Optimization**: 12-14 hours
- **Analysis**: 5 minutes
- **Planning**: 2 minutes
- **Data Collection**: 1-2 hours (10-15 edge cases)
- **Total**: ~1 day

**Expected Results**:
- Mean F1: 88-90%
- Collection target: 10-15 companies (optional)
- Final dataset: 130-145 examples

---

### Iteration 4+: Refinement (optional)
- Quality improvements
- Edge case coverage
- Production feedback integration
- Target: 93-95% F1

---

## Collection Strategies by Iteration

### Iteration 1 → 2: Broad Expansion
**Focus**: Fill pattern gaps
- Underrepresented crop types
- Geographic diversity
- Company size variety
- Evidence quality improvement

**Sources**:
- Supplier customer lists (Signify, Hortilux)
- Trade media databases (GroentenNieuws, OnderGlas)
- Regional directories
- Crop-specific associations

---

### Iteration 2 → 3: Targeted Coverage
**Focus**: Address specific weaknesses
- High-error crops (identified in analysis)
- Contradictory evidence patterns
- Low-confidence classifications
- Boundary cases (medium lighting intensity)

**Sources**:
- Failure mode examples
- Similar companies to misclassified ones
- Edge cases from iteration 2 research

---

### Iteration 3 → 4: Edge Cases
**Focus**: Polish and exceptions
- Rare crop types
- Hybrid lighting systems
- Seasonal variation cases
- Multi-location companies

**Sources**:
- Production misclassifications (if deployed)
- Specialized grower associations
- Research papers and case studies

---

## Quality Control Checklist

### Before Each Iteration
- [ ] Verify no duplicate companies
- [ ] Check data/research/*.json count matches expectation
- [ ] Review classification distribution (aim for 45-55% POSITIVE)
- [ ] Verify all companies have expected_classification in source list
- [ ] Run `scripts/count_classifications.py` to confirm balance

### After Each Optimization
- [ ] Check optimization.log for errors
- [ ] Verify cv_results.json exists and is valid
- [ ] Review mean F1, std F1, overfitting gap
- [ ] Check fold_scores for outliers
- [ ] Run failure analysis before proceeding

### Before Data Collection
- [ ] Review collection plan recommendations
- [ ] Verify search strategies are feasible
- [ ] Check target counts are realistic
- [ ] Ensure quality criteria are clear
- [ ] Plan for manual review time

---

## Troubleshooting

### Issue: Optimization Takes Too Long
**Symptoms**: >12 hours for iteration
**Solutions**:
- Reduce `--budget` from medium to light
- Reduce `--num-repeats` from 10 to 5
- Reduce `--num-folds` from 5 to 3
- Use fewer examples for testing

---

### Issue: No Improvement Between Iterations
**Symptoms**: F1 increase <2%
**Diagnosis**:
1. Check if examples are high quality (not 0+/0- evidence)
2. Review if targeted patterns were actually collected
3. Examine if contradictory evidence cases increased

**Solutions**:
- Review failure analysis more carefully
- Focus on quality over quantity
- Consider refining evidence classification logic
- May need architectural changes (not just more data)

---

### Issue: High Variance (Std F1 >8%)
**Symptoms**: Inconsistent performance across folds
**Diagnosis**:
1. Dataset too small or imbalanced
2. High overlap between patterns
3. Poor evidence quality

**Solutions**:
- Collect more examples (target 100+ minimum)
- Ensure balanced class distribution (45-55%)
- Improve evidence quality criteria
- Add stratification by multiple factors

---

### Issue: Overfitting (Gap >10%)
**Symptoms**: Train F1 >> Val F1
**Diagnosis**:
1. Model memorizing training data
2. Too many similar examples
3. Insufficient diversity

**Solutions**:
- Increase dataset diversity
- Reduce GEPA optimization iterations
- Add regularization to metric
- Cross-validate with different splits

---

## Cost & Time Estimates

### Per Iteration Costs
| Phase | Time | API Cost | Notes |
|-------|------|----------|-------|
| Optimization | 8-12h | $25-40 | Gemini 2.5 Pro (medium budget) |
| Analysis | 5min | $0 | Local processing |
| Planning | 2min | $0 | Local processing |
| Data Collection | 2-6h | $10-25 | Perplexity + Gemini |
| **Total** | **~1-2 days** | **$35-65** | Per iteration |

### Full Refinement Cycle (3-4 iterations)
- **Time**: 2-3 weeks
- **Cost**: $100-250
- **Examples**: 65 → 120-150
- **F1**: 63% → 88-92%

---

## Success Metrics by Iteration

### Iteration 1 (65 examples)
- ✅ Mean F1: 75-78%
- ✅ Std F1: 4-6%
- ✅ Overfitting gap: <5%
- ✅ False positives: ≤1

### Iteration 2 (100 examples)
- ✅ Mean F1: 82-85%
- ✅ Std F1: 3-5%
- ✅ Overfitting gap: <5%
- ✅ False positives: 0

### Iteration 3 (120 examples)
- ✅ Mean F1: 88-90%
- ✅ Std F1: 2-4%
- ✅ Overfitting gap: <3%
- ✅ False positives: 0

### Production Ready (150+ examples)
- ✅ Mean F1: 93-95%
- ✅ Std F1: <3%
- ✅ Overfitting gap: <3%
- ✅ False positives: 0
- ✅ Precision: >95%
- ✅ Recall: >90%

---

## Next Steps

1. **Run Iteration 1**: Execute 65-example optimization
   ```bash
   uv run python notebooks/optimize_greenhouse_detector.py \
     --output-dir results/iteration_1_65examples
   ```

2. **Analyze Results**: Identify failure patterns
   ```bash
   uv run python scripts/analyze_optimization_failures.py \
     --results-dir results/iteration_1_65examples
   ```

3. **Plan Iteration 2**: Generate targeted company list
   ```bash
   uv run python scripts/generate_targeted_companies.py \
     --analysis results/iteration_1_65examples/failure_analysis.json
   ```

4. **Repeat Until Target**: Continue iterations until 95% F1 achieved

---

**Document Status**: ✅ Ready for Execution
**Last Updated**: 2025-11-11
**Prepared By**: Claude Code Assistant
