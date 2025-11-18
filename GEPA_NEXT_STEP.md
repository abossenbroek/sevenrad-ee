# GEPA 65-Example Optimization Plan

**Date Created**: 2025-11-11
**Status**: Ready to Execute
**Estimated Duration**: 8-10 hours

## Executive Summary

Run full GEPA optimization with 65-example dataset (36 POSITIVE, 29 NEGATIVE) to measure performance improvement from 19-example baseline. Expected to achieve 75-85% F1 score (vs 63.4% baseline) through increased dataset diversity and better class balance.

---

## Objectives

### Primary Goal
Demonstrate that scaling from 19 to 65 examples (3.4x increase) yields significant performance improvement in Dutch greenhouse grow light detection.

### Success Criteria
| Metric | 19-Example Baseline | 65-Example Target | Status |
|--------|---------------------|-------------------|--------|
| **Mean F1** | 63.39% | ≥75% | ⏳ Pending |
| **Std F1** | 6.65% | ≤5% | ⏳ Pending |
| **Overfitting Gap** | -5.16% | <5% | ⏳ Pending |
| **False Positive Rate** | 0% (0/6) | 0% | ⏳ Pending |

### Research Questions
1. Does dataset size scaling improve F1 score by 10-20%?
2. Does better class balance (1.24:1 vs 8:2) reduce variance?
3. Does GEPA optimization benefit more from larger datasets?
4. Do contradictory evidence cases improve classification accuracy?

---

## Prerequisites Check

### ✅ Dataset Ready
- [x] 65 companies researched with 3-query Perplexity strategy
- [x] All classifications resolved (no NEEDS_MANUAL_REVIEW)
- [x] Class balance: 36:29 (1.24:1 ratio)
- [x] Ground truth labels from VIIRS satellite data
- [x] Committed to git (commit `77fc7f2`)

### ✅ Code Ready
- [x] `notebooks/optimize_greenhouse_detector.py` exists
- [x] DSPy GEPA optimizer configured with Gemini 2.5 Pro
- [x] Dutch-aware hierarchical F1 metric implemented
- [x] Cross-validation pipeline with stratified splitting
- [x] Evidence classification with semantic understanding

### ✅ Environment Ready
```bash
# Verify API keys
echo $GEMINI_API_KEY  # Should output: AIza...
echo $PERPLEXITY_API_KEY  # Should output: pplx-...

# Verify dependencies
uv pip list | grep -E "(dspy|google-generativeai)"
# Should show: dspy-ai, google-generativeai

# Check available disk space (need ~500MB for results)
df -h .
```

### ⚠️ Resource Requirements
- **Time**: 8-10 hours (GEPA optimization ~7h + CV evaluation ~2h)
- **API Costs**: ~$25-40 (Gemini 2.5 Pro with medium budget)
- **Disk Space**: ~500 MB (optimization artifacts + CV results)
- **Memory**: ~4 GB (for DSPy optimization state)

---

## Step-by-Step Execution Plan

### Phase 1: Pre-Flight Check (5 minutes)

#### 1.1 Verify Dataset Integrity
```bash
# Count companies and verify classifications
uv run python scripts/count_classifications.py
# Expected output:
#   Total companies: 65
#   POSITIVE: 36 (55.4%)
#   NEGATIVE: 29 (44.6%)
#   ✅ All companies classified! Ready for GEPA optimization.
```

#### 1.2 Create Output Directory
```bash
mkdir -p results/65examples
```

#### 1.3 Review Configuration
```bash
# Check optimizer configuration in notebooks/optimize_greenhouse_detector.py
grep -A 10 "GEPA_CONFIG" notebooks/optimize_greenhouse_detector.py
# Verify:
#   - budget: medium (balances cost vs performance)
#   - max_bootstrapped_demos: 4
#   - max_labeled_demos: 8
#   - num_threads: 4
```

---

### Phase 2: GEPA Optimization (~7 hours)

#### 2.1 Launch Optimization
```bash
# Run in background to prevent terminal disconnect
nohup uv run python notebooks/optimize_greenhouse_detector.py \
  --output-dir results/65examples \
  --num-examples 65 \
  --budget medium \
  > results/65examples/optimization.log 2>&1 &

# Save process ID
echo $! > results/65examples/optimizer.pid
```

#### 2.2 Monitor Progress
```bash
# Check log file periodically (every 30 minutes)
tail -n 50 results/65examples/optimization.log

# Look for:
#   - "Starting GEPA optimization with 65 examples"
#   - "Iteration N/M: F1 = X.XX%"
#   - "Best program so far: F1 = X.XX%"
#   - Warning signs: "API rate limit", "Memory error", "Timeout"
```

#### 2.3 Expected GEPA Timeline
| Phase | Duration | What's Happening |
|-------|----------|------------------|
| Initialization | 5 min | Load dataset, configure DSPy |
| Iteration 1-10 | 2-3 hours | Initial prompt exploration |
| Iteration 11-20 | 2-3 hours | Refinement and Pareto optimization |
| Iteration 21+ | 2-3 hours | Fine-tuning best candidates |
| Finalization | 10 min | Save optimized program |

#### 2.4 GEPA Success Indicators
- ✅ F1 improvement trend (baseline → iteration 10 → iteration 20)
- ✅ Pareto front expansion (multiple strong candidates)
- ✅ Convergence (F1 stabilizes after iteration 20)
- ⚠️ No improvement plateau at low F1 (<50%)
- ⚠️ Excessive overfitting (train F1 >> val F1)

---

### Phase 3: Cross-Validation Evaluation (~2 hours)

#### 3.1 Launch CV Evaluation
```bash
# After GEPA completes, run 10-repeat 5-fold CV
uv run python notebooks/evaluate_cv.py \
  --program results/65examples/optimized_program.json \
  --output-dir results/65examples \
  --num-repeats 10 \
  --num-folds 5 \
  > results/65examples/cv_evaluation.log 2>&1
```

#### 3.2 Expected CV Timeline
| Phase | Duration | What's Happening |
|-------|----------|------------------|
| Setup | 2 min | Load optimized program, prepare data |
| 50 CV Folds | 1.5-2 hours | Evaluate each fold (~2-3 min/fold) |
| Aggregation | 2 min | Calculate mean, std, overfitting gap |
| Save Results | 1 min | Write cv_results.json |

#### 3.3 CV Success Indicators
- ✅ Mean F1 ≥75% (vs 63.4% baseline)
- ✅ Std F1 ≤5% (vs 6.65% baseline)
- ✅ Overfitting gap <5% (vs -5.16% baseline)
- ✅ Consistent performance across folds (range <20%)
- ⚠️ High variance (std >8%)
- ⚠️ Severe overfitting (gap >15%)

---

### Phase 4: Results Analysis (30 minutes)

#### 4.1 Generate Comparison Report
```bash
uv run python scripts/compare_optimizations.py \
  --baseline results/19examples \
  --experiment results/65examples \
  --output results/65examples/comparison_report.md
```

#### 4.2 Key Metrics to Compare
| Metric | 19 Examples | 65 Examples | Δ | % Change |
|--------|-------------|-------------|---|----------|
| **Mean F1** | 63.39% | ? | ? | ? |
| **Std F1** | 6.65% | ? | ? | ? |
| **Train F1** | 58.23% | ? | ? | ? |
| **Overfitting Gap** | -5.16% | ? | ? | ? |
| **False Positives** | 0/6 | ? | ? | ? |
| **GEPA Improvement** | +17.4% | ? | ? | ? |

#### 4.3 Analysis Questions
1. **Dataset Size Impact**:
   - Did F1 improve by ≥10% as expected?
   - Did variance decrease with more examples?
   - Is negative overfitting gap maintained?

2. **Class Balance Impact**:
   - Did 1.24:1 balance improve over 8:2 ratio?
   - Are false positives still 0%?
   - Is precision/recall more balanced?

3. **GEPA Optimization Impact**:
   - Did GEPA discover better prompts with more data?
   - Is the improvement over base program larger?
   - Are there new strategies in the optimized prompt?

4. **Evidence Quality Impact**:
   - Do companies with contradictory evidence classify better?
   - Are 0+/0- evidence cases handled correctly?
   - Does tier-2 evidence weighting improve?

---

## Expected Outcomes

### Optimistic Scenario (Best Case)
```
Mean F1: 82-85%
Std F1: 3-4%
Train F1: 78-80%
Overfitting Gap: 2-4%
False Positives: 0/11 (0%)

Interpretation: Excellent generalization, dataset size significantly improved
performance, ready for production deployment.
```

### Realistic Scenario (Expected)
```
Mean F1: 75-78%
Std F1: 4-5%
Train F1: 73-75%
Overfitting Gap: 1-3%
False Positives: 0-1/11 (<10%)

Interpretation: Strong improvement over baseline, acceptable variance,
good candidate for pilot deployment with human review.
```

### Pessimistic Scenario (Concerning)
```
Mean F1: 68-72%
Std F1: 6-8%
Train F1: 65-68%
Overfitting Gap: 3-5%
False Positives: 1-2/11 (10-20%)

Interpretation: Modest improvement, still below 75% target, may need:
- More examples (target 100+)
- Better quality control on classifications
- Refined evidence classification logic
```

---

## Troubleshooting

### Issue: API Rate Limits
**Symptoms**: "Rate limit exceeded" errors in log
**Solution**:
```bash
# Check current rate limits
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
  https://generativelanguage.googleapis.com/v1/models

# Wait 1 minute, resume optimization
uv run python notebooks/optimize_greenhouse_detector.py \
  --output-dir results/65examples \
  --resume results/65examples/checkpoint.json
```

### Issue: Memory Errors
**Symptoms**: "MemoryError" or "Killed" in log
**Solution**:
```bash
# Reduce batch size
# Edit notebooks/optimize_greenhouse_detector.py:
#   BATCH_SIZE = 4  # Change to 2

# Or reduce number of threads
#   num_threads = 4  # Change to 2
```

### Issue: Optimization Stalls
**Symptoms**: No progress for 2+ hours, same F1 score
**Solution**:
```bash
# Check if process is still running
ps aux | grep optimize_greenhouse_detector

# If stuck, restart with different random seed
uv run python notebooks/optimize_greenhouse_detector.py \
  --output-dir results/65examples \
  --seed $(date +%s)
```

### Issue: Poor CV Performance
**Symptoms**: Mean F1 <65%, high variance (std >8%)
**Diagnosis**:
1. Check if overfitting (train F1 >> val F1)
2. Examine fold-level results for outliers
3. Review companies with low confidence scores

**Solutions**:
- Increase `min_evidence_threshold` in metric
- Add more NEGATIVE examples (current 29, target 40+)
- Refine evidence classification prompt

---

## Post-Optimization Analysis

### Step 1: Compare Optimized Prompts
```bash
# Extract prompts from both optimizations
uv run python scripts/extract_prompts.py \
  --baseline results/19examples/optimized_program.json \
  --experiment results/65examples/optimized_program.json \
  --output results/65examples/prompt_comparison.md
```

**Questions to Answer**:
- What new strategies did GEPA discover with 65 examples?
- Are there new evidence types or thresholds?
- How did Dutch terminology handling evolve?

### Step 2: Analyze Fold-Level Performance
```bash
# Generate per-fold breakdown
uv run python scripts/analyze_cv_folds.py \
  --cv-results results/65examples/cv_results.json \
  --output results/65examples/fold_analysis.md
```

**Questions to Answer**:
- Which folds performed best/worst?
- Are there systematic patterns in errors?
- Do specific companies consistently misclassify?

### Step 3: Confusion Matrix Analysis
```bash
# Generate confusion matrices
uv run python scripts/generate_confusion_matrix.py \
  --program results/65examples/optimized_program.json \
  --output results/65examples/confusion_matrix.png
```

**Questions to Answer**:
- What is the false positive rate on NEGATIVE cases?
- What is the false negative rate on POSITIVE cases?
- Are NEEDS_MANUAL_REVIEW suggestions appropriate?

### Step 4: Evidence Quality Analysis
```bash
# Analyze evidence patterns
uv run python scripts/analyze_evidence.py \
  --data-dir data/research \
  --output results/65examples/evidence_analysis.md
```

**Questions to Answer**:
- Which evidence types correlate with correct classifications?
- Do tier-2 sources improve accuracy?
- Are Dutch terms sufficient for classification?

---

## Deliverables

### Required Outputs
1. **Optimized Program**: `results/65examples/optimized_program.json`
2. **CV Results**: `results/65examples/cv_results.json`
3. **Optimization Log**: `results/65examples/optimization.log`
4. **Comparison Report**: `results/65examples/comparison_report.md`

### Optional Outputs
5. **Prompt Comparison**: `results/65examples/prompt_comparison.md`
6. **Fold Analysis**: `results/65examples/fold_analysis.md`
7. **Confusion Matrix**: `results/65examples/confusion_matrix.png`
8. **Evidence Analysis**: `results/65examples/evidence_analysis.md`

---

## Commit Strategy

### After Successful Optimization
```bash
git add results/65examples/
git commit -m "Complete 65-example GEPA optimization and CV evaluation

Results:
- Mean F1: X.XX% (vs 63.39% baseline) [+Y.Y%]
- Std F1: X.XX% (vs 6.65% baseline)
- Overfitting Gap: X.XX% (vs -5.16% baseline)
- False Positive Rate: N/29 (vs 0/6 baseline)

GEPA Optimization:
- Base Program F1: X.XX%
- Best Program F1: X.XX% [+Y.Y% improvement]
- Iterations: ~21+
- Runtime: ~Z hours

Cross-Validation:
- 10-repeat 5-fold CV (50 total folds)
- Stratified splitting by class labels
- Runtime: ~2 hours

Comparison with 19-Example Baseline:
- Dataset size: 3.4x increase (19 → 65)
- Class balance: Improved from 8:2 to 36:29
- Performance: [Improved/Similar/Declined]

Ready for [production deployment/pilot testing/further iteration]

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Decision Points

### Go / No-Go for Production
**Criteria for Production Deployment**:
- ✅ Mean F1 ≥80%
- ✅ False Positive Rate ≤5%
- ✅ Overfitting Gap <5%
- ✅ Std F1 ≤4%

**If ALL criteria met**: Deploy to production with monitoring
**If 3/4 criteria met**: Pilot deployment with human review
**If ≤2 criteria met**: Iterate with more data or refined approach

### Next Steps Decision Tree
```
IF Mean F1 ≥80% AND FP Rate ≤5%:
  → Deploy to production
  → Set up monitoring dashboard
  → Create API endpoint

ELSE IF Mean F1 ≥75% AND FP Rate ≤10%:
  → Pilot deployment with manual review
  → Collect production feedback
  → Expand training set with edge cases

ELSE IF Mean F1 ≥70%:
  → Collect 100+ examples (target 150 total)
  → Refine evidence classification logic
  → Consider alternative architectures

ELSE:
  → Deep-dive into failure modes
  → Re-evaluate 3-query strategy
  → Consider hybrid human-AI approach
```

---

## Timeline

### Optimistic Timeline (8 hours)
```
Hour 0:00 - Pre-flight check (5 min)
Hour 0:05 - Launch GEPA optimization
Hour 6:05 - GEPA completes (6h runtime)
Hour 6:10 - Launch CV evaluation
Hour 8:10 - CV completes (2h runtime)
Hour 8:20 - Generate comparison report
Hour 8:30 - Analysis complete ✅
```

### Realistic Timeline (10 hours)
```
Hour 0:00 - Pre-flight check (5 min)
Hour 0:05 - Launch GEPA optimization
Hour 7:35 - GEPA completes (7.5h runtime)
Hour 7:40 - Launch CV evaluation
Hour 9:50 - CV completes (2.2h runtime)
Hour 10:00 - Generate comparison report
Hour 10:30 - Analysis complete ✅
```

### Pessimistic Timeline (14 hours)
```
Hour 0:00 - Pre-flight check (5 min)
Hour 0:05 - Launch GEPA optimization
Hour 2:30 - API rate limit, wait 30 min
Hour 9:05 - GEPA completes (8.5h effective runtime)
Hour 9:10 - Launch CV evaluation
Hour 12:10 - CV completes (3h runtime, slower folds)
Hour 12:20 - Generate comparison report
Hour 13:00 - Debugging low performance
Hour 14:00 - Root cause analysis complete ✅
```

---

## Launch Command

**Ready to execute? Run this command:**

```bash
# Create results directory
mkdir -p results/65examples

# Launch optimization (background execution recommended)
nohup uv run python notebooks/optimize_greenhouse_detector.py \
  --output-dir results/65examples \
  --num-examples 65 \
  --budget medium \
  --num-repeats 10 \
  --num-folds 5 \
  > results/65examples/optimization.log 2>&1 &

# Save process ID for monitoring
echo $! > results/65examples/optimizer.pid

# Monitor progress
tail -f results/65examples/optimization.log
```

**Estimated completion**: 2025-11-11 22:00 (if started at 12:00)

---

## References

- **19-Example Baseline**: `results/19examples/real_optimization_analysis.md`
- **Dataset**: `data/research/` (65 companies)
- **Optimizer**: `notebooks/optimize_greenhouse_detector.py`
- **Metric**: `src/sevenrad_ee/ai/greenhouse_metrics.py`
- **Prior Commits**: `77fc7f2` (dataset expansion), `58a1189` (19-example results)

---

**Document Status**: ✅ Ready for Execution
**Last Updated**: 2025-11-11
**Prepared By**: Claude Code Assistant
