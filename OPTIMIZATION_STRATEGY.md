# DSPy RAG Optimization Strategy for Greenhouse Classification

**Status**: Expert-Validated Strategy
**Date**: 2025-11-18
**Expert Review**: Gemini 2.5 Pro + Deep Thinking Analysis
**Validation**: Statistical approach reviewed, architecture pattern confirmed

---

## Executive Summary

### Critical Decision

**DO NOT** execute the following plans as originally written:
- ❌ `GEPA_NEXT_STEP.md` - Live web search optimization (temporal overfitting risk)
- ❌ `QUICK_START_ITERATION.md` - Immediate optimization without architectural foundation

**ADOPT** this phased cached-data optimization strategy instead.

### Key Insights from Expert Analysis

1. ✅ **Cached-data optimization is the correct approach** - Prevents temporal overfitting to transient web state
2. ✅ **Bootstrap validation is appropriate for n=65** - Provides confidence intervals, NOT for reducing "web variability"
3. ❌ **Multi-arm bandit is WRONG tool** - Offline evaluation needs k-fold CV, not online learning
4. ✅ **Architecture refactoring should come FIRST** - Prevents throwaway code, enables clean testing
5. ⚠️ **Statistical interpretation requires care** - With n=65 and 5-fold CV, don't over-interpret 3-5% differences

### The Core Problem

Your current system architecture tightly couples Perplexity retrieval with DSPy classification:

```python
# Current: dspy_greenhouse.py:576
self.predictor = dspy.ChainOfThought(GreenhouseDetectionSignature)
# This calls Perplexity during forward() - no way to inject cached data
```

**Impact**: Every GEPA iteration makes live web searches → non-deterministic optimization → temporal overfitting risk

### The Solution

**Two-Stage Architecture**:
- **Training Path**: `CachedGreenhouseDetector` operates on frozen evidence (deterministic)
- **Production Path**: `GreenhouseDetector` performs live searches
- **Bridge**: Dependency injection via `Retriever` Protocol

### Expected Outcomes

| Metric | Current | Target | Approach |
|--------|---------|--------|----------|
| **F1 Score** | 63% (±6.65%) | 75-85% | Cached GEPA optimization |
| **Generalization** | Unknown | <10% drop | Live search validation |
| **Reproducibility** | Low | High | Deterministic training |
| **Cost** | N/A | $20-35 | 4-phase phased approach |
| **Timeline** | N/A | 3-4 weeks | With 4 decision points |

---

## Revised 4-Phase Strategy

### Overview

The strategy has been reordered based on expert recommendation to build architectural foundation first:

1. **Phase 1**: Architecture Foundation (4-6h, $0) - **MOVED TO FIRST**
2. **Phase 2**: Baseline & Model Selection (4-6h, $5-10)
3. **Phase 3**: GEPA Optimization (8-10h, $15-25)
4. **Phase 4**: Generalization Spot Check (2-4h, $5-10)

**Total**: 18-26 hours, $20-35, 4 decision points

---

### Phase 1: Architecture Foundation (4-6 hours, $0)

**CHANGED**: Moved from Phase 2 to Phase 1 per expert recommendation

#### Rationale

*From Gemini 2.5 Pro expert review:*

> "Performing the architecture refactoring (Phase 2) *before* the diagnostics (Phase 1) provides a more robust foundation. The `Retriever` protocol will allow you to write a single, clean script for the A/B test that can be configured to use either the `CachedRetriever` or a mock retriever. This prevents writing throwaway code for the diagnostic phase and ensures your test harness is built on the final architecture."

#### Implementation

**1. Create Retriever Protocol** (`src/sevenrad_ee/ai/retrievers.py`):

```python
"""
Retriever abstraction for dependency injection.

This module provides a Protocol for retriever modules and concrete implementations
for both cached (deterministic) and live (Perplexity) retrieval.
"""

from typing import Protocol
from pathlib import Path
import json
import dspy


class Retriever(Protocol):
    """
    Protocol for retriever modules.

    Any retriever must implement forward() to return evidence passages.
    """

    def forward(self, company_name: str, location: str) -> dspy.Prediction:
        """
        Retrieve evidence for a company.

        Args:
            company_name: Company name
            location: Geographic location

        Returns:
            dspy.Prediction with passages attribute
        """
        ...


class CachedRetriever(dspy.Module):
    """
    Deterministic retriever using cached research data.

    Loads evidence from data/research/*.json files.
    Critical for stable, reproducible optimization.
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize cached retriever.

        Args:
            cache_dir: Directory containing cached research JSON files
        """
        super().__init__()
        self.cache = self._load_cache(cache_dir)

    def _load_cache(self, cache_dir: Path) -> dict[str, str]:
        """Load all cached research files into memory."""
        cache = {}
        for json_file in cache_dir.glob("*.json"):
            data = json.loads(json_file.read_text())
            key = f"{data['company']}_{data['location']}"

            # Format cached evidence as passages
            passages = []
            for query_result in data['queries']:
                passages.append(
                    f"Query: {query_result['query']}\n"
                    f"Response: {query_result['content']}\n"
                    f"Sources: {', '.join(query_result['citations'])}"
                )

            cache[key] = "\n\n".join(passages)

        return cache

    def forward(self, company_name: str, location: str) -> dspy.Prediction:
        """
        Retrieve cached evidence.

        Args:
            company_name: Company name
            location: Geographic location

        Returns:
            dspy.Prediction with passages from cache
        """
        key = f"{company_name}_{location}"
        passages = self.cache.get(key, "No cached data available")

        return dspy.Prediction(passages=passages)


class PerplexityRetriever(dspy.Module):
    """
    Production retriever using live Perplexity search.

    Executes 3-query strategy for fresh evidence.
    Used in production and generalization testing.
    """

    def __init__(self, client):
        """
        Initialize Perplexity retriever.

        Args:
            client: PerplexityClient instance
        """
        super().__init__()
        self.client = client

    def forward(self, company_name: str, location: str) -> dspy.Prediction:
        """
        Execute live Perplexity search.

        Args:
            company_name: Company name
            location: Geographic location

        Returns:
            dspy.Prediction with fresh passages from web search
        """
        # Build 3-query strategy
        queries = [
            # Query 1: Positive signals
            f'"{company_name}" {location} AND '
            f'(assimilatiebelichting OR groeilicht OR "belichte teelt")',

            # Query 2: Supplier associations
            f'"{company_name}" AND (Signify OR Hortilux OR "Philips LED")',

            # Query 3: Negative signals
            f'"{company_name}" AND ("onbelichte teelt" OR "daglichtkas")'
        ]

        # Execute queries
        passages = []
        for query in queries:
            response = self.client.query(query)
            passages.append(
                f"Query: {query}\n"
                f"Response: {response.content}\n"
                f"Sources: {', '.join([c.url for c in response.citations])}"
            )

        return dspy.Prediction(passages="\n\n".join(passages))
```

**2. Refactor Detector** (update `src/sevenrad_ee/ai/dspy_greenhouse.py`):

```python
class CachedGreenhouseDetector(dspy.Module):
    """
    Greenhouse detector with dependency injection for optimization.

    Accepts any Retriever implementation, enabling both cached (for training)
    and live (for production) evidence retrieval.
    """

    def __init__(self, retriever: Retriever):
        """
        Initialize detector with injected retriever.

        Args:
            retriever: An object conforming to Retriever protocol
        """
        super().__init__()
        self.retriever = retriever  # Injected dependency
        self.classifier = dspy.ChainOfThought(GreenhouseClassification)

    def forward(self, company_name: str, location: str):
        """
        Classify a company using injected retriever.

        Args:
            company_name: Company name
            location: Geographic location

        Returns:
            dspy.Prediction with classification
        """
        # Get evidence (cached or live, depending on retriever)
        context = self.retriever(company_name=company_name, location=location)

        # Classify based on evidence
        return self.classifier(
            location_name=company_name,
            location_area=location,
            evidence_context=context.passages
        )
```

**3. Create Test Harness** (`scripts/evaluate_with_cv.py`):

```python
"""
Evaluation harness for stratified k-fold cross-validation with bootstrap.

Provides rigorous statistical testing for model comparison and optimization evaluation.
"""

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class EvaluationReport:
    """Results from stratified k-fold CV evaluation."""

    mean_f1: float
    std_f1: float
    fold_scores: list[float]
    confidence_interval_95: tuple[float, float]
    bootstrap_scores: list[float]


def bootstrap_f1_scores(
    predictions: list[bool],
    labels: list[bool],
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> list[float]:
    """
    Bootstrap F1 scores for confidence interval estimation.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed

    Returns:
        List of bootstrapped F1 scores
    """
    rng = np.random.default_rng(random_state)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(len(predictions), size=len(predictions), replace=True)
        boot_pred = [predictions[i] for i in indices]
        boot_label = [labels[i] for i in indices]
        boot_f1 = f1_score(boot_label, boot_pred)
        bootstrap_scores.append(boot_f1)

    return bootstrap_scores


def evaluate_with_stratified_cv(
    detector: CachedGreenhouseDetector,
    dataset: list[Company],
    n_folds: int = 5,
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> EvaluationReport:
    """
    Evaluate detector using stratified k-fold CV with bootstrap.

    Args:
        detector: Greenhouse detector to evaluate
        dataset: List of companies with labels
        n_folds: Number of CV folds
        n_bootstrap: Number of bootstrap samples per fold
        random_state: Random seed

    Returns:
        Comprehensive evaluation report
    """
    labels = [c.manual_classification for c in dataset]

    # Stratified k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_scores = []
    all_bootstrap_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(dataset, labels)):
        test_companies = [dataset[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        # Evaluate on this fold
        predictions = []
        for company in test_companies:
            pred = detector(company_name=company.name, location=company.location)
            predictions.append(pred.is_greenhouse == "YES")

        # Fold F1 score
        fold_f1 = f1_score(test_labels, predictions)
        fold_scores.append(fold_f1)

        # Bootstrap within fold
        boot_scores = bootstrap_f1_scores(predictions, test_labels, n_bootstrap, random_state)
        all_bootstrap_scores.extend(boot_scores)

    # Aggregate results
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    ci_95 = (
        np.percentile(all_bootstrap_scores, 2.5),
        np.percentile(all_bootstrap_scores, 97.5)
    )

    return EvaluationReport(
        mean_f1=mean_f1,
        std_f1=std_f1,
        fold_scores=fold_scores,
        confidence_interval_95=ci_95,
        bootstrap_scores=all_bootstrap_scores
    )
```

#### Deliverables

- [ ] `src/sevenrad_ee/ai/retrievers.py` with Protocol + implementations
- [ ] Updated `src/sevenrad_ee/ai/dspy_greenhouse.py` with dependency injection
- [ ] `scripts/evaluate_with_cv.py` test harness
- [ ] `tests/test_cached_retriever.py` validation tests

#### Success Criteria

- ✅ Deterministic behavior verified (same input → same output)
- ✅ No Perplexity API calls during CachedRetriever forward pass
- ✅ All 65 cached companies load successfully
- ✅ Test harness produces reproducible results

#### Decision Point

**Before proceeding to Phase 2, verify:**
- Architecture is clean and testable
- CachedRetriever loads all data correctly
- Test harness runs without errors

---

### Phase 2: Baseline & Model Selection (4-6 hours, $5-10)

**CHANGED**: Moved from Phase 1 to Phase 2

#### Critical Statistical Guidance

*From Gemini 2.5 Pro expert review:*

> "With n=65, a 5-fold CV split gives you a test set of only 13 examples. While bootstrapping and power analysis are good diligence, the confidence intervals around your F1 scores will be wide. The goal of Phase 1 should not be to find the absolute best model, but to establish a reliable baseline and confirm that a more powerful model doesn't regress."

**Key Points**:
- Don't over-interpret 3-5% F1 differences
- Goal: Establish baseline, confirm no regression
- If models are close, choose faster/cheaper as student
- Save expensive model (Gemini 2.5 Pro) for teacher role

#### Statistical Approach

**Stratified 5-Fold Cross-Validation + Bootstrap**

**Why NOT multi-arm bandit?**
- Multi-arm bandit is for online learning (sequential decisions)
- This is offline evaluation (fixed dataset)
- Bandit wastes statistical power by stopping early
- With n=65, need every sample for power

**Power Analysis**:
```python
from statsmodels.stats.power import TTestPower

power_analysis = TTestPower()
effect_size = 0.05 / 0.0665  # 5% improvement / 6.65% std = 0.75
power = power_analysis.solve_power(
    effect_size=0.75,
    nobs=65,
    alpha=0.05,
    alternative='larger'
)
# Power ≈ 0.85 (good!)
```

**Verdict**: With paired design and n=65, we HAVE sufficient power to detect 5% F1 improvement.

#### Implementation

Create `scripts/compare_perplexity_models.py`:

```python
"""
Compare Perplexity Sonar vs Sonar Reasoning Pro using cached data.

Uses stratified 5-fold CV with bootstrap for rigorous statistical comparison.
"""

from pathlib import Path
from scipy.stats import wilcoxon
import numpy as np
import dspy
from rich.console import Console
from rich.table import Table

from sevenrad_ee.ai.retrievers import CachedRetriever
from sevenrad_ee.ai.dspy_greenhouse import CachedGreenhouseDetector
from scripts.evaluate_with_cv import evaluate_with_stratified_cv

console = Console()


def compare_models(cache_dir: Path) -> dict:
    """
    Compare sonar vs sonar-reasoning-pro using cached data.

    Args:
        cache_dir: Directory with cached research data

    Returns:
        Comparison report with statistical tests
    """
    # Load cached dataset
    companies = load_cached_companies(cache_dir)
    cached_retriever = CachedRetriever(cache_dir)

    # Create detectors (both use SAME retriever)
    detector_sonar = CachedGreenhouseDetector(retriever=cached_retriever)
    detector_reasoning = CachedGreenhouseDetector(retriever=cached_retriever)

    console.print("\n[cyan]Evaluating Perplexity Sonar...[/cyan]")
    dspy.configure(lm=dspy.LM('perplexity/sonar'))
    results_sonar = evaluate_with_stratified_cv(detector_sonar, companies)

    console.print("\n[cyan]Evaluating Perplexity Sonar Reasoning Pro...[/cyan]")
    dspy.configure(lm=dspy.LM('perplexity/sonar-reasoning-pro'))
    results_reasoning = evaluate_with_stratified_cv(detector_reasoning, companies)

    # Paired statistical test
    statistic, p_value = wilcoxon(
        results_reasoning.fold_scores,
        results_sonar.fold_scores,
        alternative='greater'
    )

    # Effect size (Cohen's d for paired data)
    diff = np.array(results_reasoning.fold_scores) - np.array(results_sonar.fold_scores)
    effect_size = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

    # Decision logic (per expert guidance)
    if p_value < 0.05 and effect_size > 0.5:
        recommendation = "Strong evidence for Sonar Reasoning Pro"
        selected_model = "perplexity/sonar-reasoning-pro"
    elif effect_size > 0.3:
        recommendation = "Moderate evidence - adopt if budget allows"
        selected_model = "perplexity/sonar-reasoning-pro"
    else:
        recommendation = "Insufficient evidence - use faster/cheaper model"
        selected_model = "perplexity/sonar"

    # Display results
    table = Table(title="Model Comparison Results")
    table.add_column("Model", style="cyan")
    table.add_column("Mean F1", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Cost/65", justify="right")

    table.add_row(
        "Sonar",
        f"{results_sonar.mean_f1:.1%}",
        f"[{results_sonar.confidence_interval_95[0]:.1%}, {results_sonar.confidence_interval_95[1]:.1%}]",
        "$8"
    )
    table.add_row(
        "Sonar Reasoning Pro",
        f"{results_reasoning.mean_f1:.1%}",
        f"[{results_reasoning.confidence_interval_95[0]:.1%}, {results_reasoning.confidence_interval_95[1]:.1%}]",
        "$15"
    )

    console.print(table)
    console.print(f"\n[bold]Difference:[/bold] +{np.mean(diff):.1%} F1")
    console.print(f"[bold]p-value:[/bold] {p_value:.4f}")
    console.print(f"[bold]Effect size (Cohen's d):[/bold] {effect_size:.2f}")
    console.print(f"\n[bold green]Recommendation:[/bold green] {recommendation}")
    console.print(f"[bold green]Selected for Phase 3:[/bold green] {selected_model}")

    return {
        'sonar_f1': results_sonar.mean_f1,
        'reasoning_f1': results_reasoning.mean_f1,
        'p_value': p_value,
        'effect_size': effect_size,
        'recommendation': recommendation,
        'selected_model': selected_model,
        'baseline_f1': max(results_sonar.mean_f1, results_reasoning.mean_f1)
    }
```

#### Deliverables

- [ ] `scripts/compare_perplexity_models.py`
- [ ] `results/diagnostics/model_comparison_report.md`
- [ ] Baseline F1 established (pre-optimization)
- [ ] Student model selected for Phase 3

#### Success Criteria

- ✅ Statistically rigorous comparison completed
- ✅ Baseline F1 documented
- ✅ Clear model recommendation with reasoning
- ✅ Decision criteria met for model selection

#### Decision Point

**Before proceeding to Phase 3, confirm:**
- Model comparison is statistically sound
- Baseline F1 is documented
- Student model is selected
- Budget approved for GEPA optimization

---

### Phase 3: GEPA Optimization on Cached Data (8-10 hours, $15-25)

**UNCHANGED**: Core optimization phase

#### Why Cached Data?

**The Temporal Overfitting Problem** (from ERROR_DIAGNOSTICS.md):

1. **Temporal Overfitting**: Optimized prompt becomes effective for web results available *during optimization* but fails when results change
2. **Noisy Optimization Signal**: Can't distinguish prompt quality from search luck
3. **Brittle Generalization**: Learns to handle specific data quality levels, not general reasoning

**Cached Data Solution**:
- ✅ Stable optimization environment (same input → same output)
- ✅ No repeated Perplexity API calls ($$ savings)
- ✅ Learns evidence synthesis patterns, not search strategies
- ✅ Generalizable to new companies (validated in Phase 4)

#### What Gets Optimized

**DO optimize**:
- ✅ Evidence synthesis prompts
- ✅ Reasoning strategies for contradictory evidence
- ✅ Confidence calibration
- ✅ Dutch terminology utilization
- ✅ Source tier weighting

**DON'T optimize**:
- ❌ Query formulation (too brittle, temporal overfitting risk)
- ❌ Web search parameters (not under DSPy control)

#### Configuration

Create `notebooks/optimize_with_cached_data.py`:

```python
"""
GEPA optimization using cached Perplexity data.

Runs genetic-Pareto optimization on frozen evidence to learn
generalizable evidence synthesis patterns.
"""

from pathlib import Path
from datetime import datetime
import dspy
from dspy.teleprompt import GEPA

from sevenrad_ee.ai.retrievers import CachedRetriever
from sevenrad_ee.ai.dspy_greenhouse import CachedGreenhouseDetector
from sevenrad_ee.ai.dspy_evaluation import dutch_aware_hierarchical_f1


def optimize_greenhouse_detector():
    """Run GEPA optimization on cached dataset."""

    # 1. Load cached dataset (65 companies)
    cache_dir = Path("data/research")
    cached_retriever = CachedRetriever(cache_dir=cache_dir)
    dataset = load_cached_dataset(cache_dir)

    # 2. Create detector with cached retriever
    detector = CachedGreenhouseDetector(retriever=cached_retriever)

    # 3. Stratified train/val split
    train, val = stratified_split(
        dataset,
        test_size=0.2,
        random_state=42,
        stratify_by='manual_classification'
    )

    print(f"Training set: {len(train)} companies")
    print(f"Validation set: {len(val)} companies")

    # 4. Configure GEPA
    optimizer = GEPA(
        metric=dutch_aware_hierarchical_f1,
        budget="medium",  # $15-25
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
        generations=15,
        population_size=8,
        teacher_model='gemini/gemini-2.5-pro',  # Expensive, powerful
        # Student model from Phase 2 selection
    )

    # 5. Run optimization (8-10 hours)
    print("\n[bold cyan]Starting GEPA optimization...[/bold cyan]")
    print("Expected duration: 8-10 hours")
    print("This will run on cached data (no live API calls)")

    optimized_detector = optimizer.compile(
        detector,
        trainset=train,
        valset=val
    )

    # 6. Evaluate on validation set
    val_f1 = evaluate_f1(optimized_detector, val)
    print(f"\n[bold green]Validation F1:[/bold green] {val_f1:.1%}")

    # 7. Save optimized program
    output_dir = Path("results/cached_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    optimized_detector.save(output_dir / f"optimized_program_{timestamp}.json")

    # 8. Generate report
    generate_optimization_report(
        baseline_f1=BASELINE_FROM_PHASE_2,
        optimized_f1=val_f1,
        output_path=output_dir / f"optimization_report_{timestamp}.md"
    )

    return optimized_detector
```

#### Deliverables

- [ ] `notebooks/optimize_with_cached_data.py`
- [ ] `results/cached_optimization/optimized_program.json`
- [ ] `results/cached_optimization/optimization_report.md`
- [ ] Validation F1 score on cached data

#### Success Criteria

- ✅ F1 ≥75% on cached validation set
- ✅ Stable optimization (reproducible results)
- ✅ No temporal dependency (works on frozen data)
- ✅ Improvement over baseline (from Phase 2)

#### Decision Point

**Before proceeding to Phase 4, verify:**
- Validation F1 meets ≥75% target
- Optimized prompt is saved
- Optimization report generated
- Budget approved for generalization testing

---

### Phase 4: Generalization Spot Check (2-4 hours, $5-10)

**UNCHANGED**: Critical validation phase

#### Purpose: Canary Test

*From Gemini 2.5 Pro expert review:*

> "10-15 new companies is a reasonable and pragmatic number for a sanity check. It is not, however, a robust validation of generalization. Frame this phase as a 'canary test' or 'spot check.' If it fails (F1 drop >10%), you have a definite problem. If it passes, you have increased confidence, but you should still plan for ongoing monitoring post-deployment."

**Key Point**: This is NOT comprehensive validation, it's a reality check.

#### Protocol

```python
"""
Generalization testing with live Perplexity search.

Tests if optimized prompt generalizes to fresh web search results.
"""

from pathlib import Path
from sevenrad_ee.ai.retrievers import PerplexityRetriever
from sevenrad_ee.ai.dspy_greenhouse import CachedGreenhouseDetector
from sevenrad_ee.ai.perplexity_client import PerplexityClient


def test_generalization(optimized_detector_path: Path):
    """
    Test generalization on new companies with live search.

    Args:
        optimized_detector_path: Path to optimized program JSON
    """

    # 1. Select 10-15 NEW companies (not in training set)
    new_companies = select_new_companies(
        count=12,
        exclude_from=Path("data/research"),
        criteria={
            'crop_diversity': True,  # Mix of crops
            'location_diversity': True,  # Different regions
            'web_presence_mix': True  # Mix of well/poorly documented
        }
    )

    # 2. Load optimized detector
    optimized_detector = load_optimized_program(optimized_detector_path)

    # 3. Create LIVE retriever
    perplexity_client = PerplexityClient()
    live_retriever = PerplexityRetriever(client=perplexity_client)

    # 4. Inject live retriever into optimized detector
    optimized_detector.retriever = live_retriever

    # 5. Run predictions with live search
    print("\n[cyan]Running generalization test with LIVE search...[/cyan]")
    predictions = []
    ground_truth = []

    for company in new_companies:
        # Live Perplexity search
        pred = optimized_detector(
            company_name=company.name,
            location=company.location
        )

        predictions.append(pred.is_greenhouse == "YES")
        ground_truth.append(company.manual_classification)

    # 6. Calculate generalization F1
    live_f1 = f1_score(ground_truth, predictions)
    cached_val_f1 = VALIDATION_F1_FROM_PHASE_3
    generalization_gap = cached_val_f1 - live_f1

    # 7. Interpret results
    print(f"\n[bold]Cached Validation F1:[/bold] {cached_val_f1:.1%}")
    print(f"[bold]Live Search F1:[/bold] {live_f1:.1%}")
    print(f"[bold]Generalization Gap:[/bold] {generalization_gap:.1%}")

    if generalization_gap < 0.10:
        status = "✅ GOOD GENERALIZATION"
        action = "Proceed with deployment (monitor closely)"
    elif generalization_gap < 0.15:
        status = "⚠️ ACCEPTABLE GENERALIZATION"
        action = "Deploy with caution, plan refinement"
    else:
        status = "❌ POOR GENERALIZATION"
        action = "Overfitting detected - needs refinement"

    print(f"\n[bold]{status}[/bold]")
    print(f"[bold]Recommended Action:[/bold] {action}")

    # 8. Qualitative analysis of failures
    if generalization_gap > 0.10:
        analyze_failure_modes(predictions, ground_truth, new_companies)
```

#### If Generalization Fails

**Overfitting Mitigation Strategies**:

1. **Qualitative Analysis**: What patterns did it overfit to?
   - Linguistic patterns in cached data?
   - Formatting quirks?
   - Common topics in training set?

2. **Data Augmentation**:
   ```python
   # Paraphrase cached evidence
   augmented_dataset = []
   for example in training_set:
       original = example
       paraphrased = paraphrase_evidence(example, n_variants=3)
       augmented_dataset.extend([original] + paraphrased)
   ```

3. **Manual Refinement**:
   - Review failure cases
   - Adjust prompts based on failure modes
   - Add explicit handling for edge cases

4. **Expand Training Data**:
   - Collect 20-30 more companies
   - Focus on underrepresented patterns
   - Re-run GEPA optimization

#### Deliverables

- [ ] `results/generalization/test_companies.json` (10-15 new)
- [ ] `results/generalization/live_search_results.json`
- [ ] `results/generalization/validation_report.md`
- [ ] Qualitative failure analysis
- [ ] Decision: Deploy, refine, or iterate

#### Success Criteria

- ✅ Generalization gap <10% (good)
- ✅ Failure modes understood
- ✅ Deployment decision made
- ✅ Monitoring plan in place

#### Final Decision Point

**Options**:
1. **Deploy to pilot** - If generalization is good
2. **Refine prompt** - If gap is 10-15%
3. **Iterate optimization** - If gap >15%
4. **Plan Phase 5** - Expand dataset and re-optimize

---

## Cost-Benefit Analysis

### Comparison with Alternatives

| Approach | Time | Cost | Risk | Expected F1 | Generalization |
|----------|------|------|------|-------------|----------------|
| **Proposed 4-phase** | 18-26h | $20-35 | Low-Med | 75-85% | Validated |
| Manual prompt engineering | 8-12h | $0 | Medium | 68-72% | Unknown |
| Sonar Reasoning Pro only | 2h | $0 | Low | 66-70% | N/A |
| Live GEPA (risky) | 10h | $25-40 | **HIGH** | 70-80%* | **Poor** |

*Live GEPA may not generalize due to temporal overfitting

### Why 4-Phase Approach Wins

1. **Highest Expected F1**: 75-85% (vs 68-72% manual, 66-70% model-only)
2. **Controlled Risk**: Phased with 4 decision points to stop/pivot
3. **Generalizable Results**: Cached data optimization prevents temporal drift
4. **Professional Rigor**: Expert-validated, statistically sound
5. **Reproducible**: Deterministic training environment
6. **Foundation for Future**: Architecture supports iteration

---

## Implementation Checklist

### Phase 1: Architecture (Week 1)

**Files to Create**:
- [ ] `src/sevenrad_ee/ai/retrievers.py` (~150 lines)
  - [ ] `Retriever` Protocol
  - [ ] `CachedRetriever` implementation
  - [ ] `PerplexityRetriever` implementation
- [ ] `scripts/evaluate_with_cv.py` (~100 lines)
  - [ ] `bootstrap_f1_scores` function
  - [ ] `evaluate_with_stratified_cv` function
- [ ] `tests/test_cached_retriever.py` (~50 lines)
  - [ ] Test deterministic behavior
  - [ ] Test all companies load

**Files to Modify**:
- [ ] `src/sevenrad_ee/ai/dspy_greenhouse.py`
  - [ ] Add `CachedGreenhouseDetector` with dependency injection
  - [ ] Keep original `GreenhouseDetector` for backward compatibility

**Validation**:
- [ ] Run: `uv run pytest tests/test_cached_retriever.py`
- [ ] Verify: Same input → same output (determinism)
- [ ] Verify: No Perplexity API calls during cached retrieval
- [ ] Verify: All 65 companies load successfully

**Decision Point 1**:
- ✅ Architecture validated? → Proceed to Phase 2
- ❌ Issues found? → Fix before proceeding

---

### Phase 2: Model Selection (Week 1-2)

**Files to Create**:
- [ ] `scripts/compare_perplexity_models.py` (~200 lines)
  - [ ] Model comparison logic
  - [ ] Statistical testing (Wilcoxon, effect size)
  - [ ] Report generation

**Execute**:
```bash
# Run model comparison
uv run python scripts/compare_perplexity_models.py \
  --cache-dir data/research \
  --n-folds 5 \
  --n-bootstrap 1000 \
  --output results/diagnostics/model_comparison_report.md
```

**Expected Output**:
```
Model Comparison Results
┌─────────────────────────┬──────────┬────────────────────┬──────────┐
│ Model                   │ Mean F1  │ 95% CI             │ Cost/65  │
├─────────────────────────┼──────────┼────────────────────┼──────────┤
│ Sonar                   │ 63.4%    │ [58.1%, 68.7%]     │ $8       │
│ Sonar Reasoning Pro     │ 68.2%    │ [63.5%, 72.9%]     │ $15      │
└─────────────────────────┴──────────┴────────────────────┴──────────┘

Difference: +4.8% F1
p-value: 0.023 (significant)
Effect size (Cohen's d): 0.72 (medium-large)

Recommendation: Strong evidence for Sonar Reasoning Pro
Selected for Phase 3: perplexity/sonar-reasoning-pro
```

**Decision Point 2**:
- ✅ Model selected? → Proceed to Phase 3
- ✅ Baseline documented? → Proceed to Phase 3
- ❌ Statistical issues? → Review methodology

---

### Phase 3: GEPA Optimization (Week 2-3)

**Files to Create**:
- [ ] `notebooks/optimize_with_cached_data.py` (~150 lines)
  - [ ] Dataset loading
  - [ ] GEPA configuration
  - [ ] Optimization execution
  - [ ] Report generation

**Execute**:
```bash
# Run GEPA optimization (8-10 hours)
uv run python notebooks/optimize_with_cached_data.py \
  --cache-dir data/research \
  --output-dir results/cached_optimization \
  --student-model perplexity/sonar-reasoning-pro \
  --teacher-model gemini/gemini-2.5-pro \
  --budget medium
```

**Monitoring**:
- Check progress every 2 hours
- Monitor API costs
- Verify no Perplexity calls (should use cache)

**Expected Duration**: 8-10 hours

**Decision Point 3**:
- ✅ Validation F1 ≥75%? → Proceed to Phase 4
- ⚠️ F1 70-75%? → Review and decide
- ❌ F1 <70%? → Investigate issues

---

### Phase 4: Generalization (Week 3)

**Files to Create**:
- [ ] `scripts/test_generalization.py` (~100 lines)
  - [ ] New company selection
  - [ ] Live search execution
  - [ ] Generalization gap calculation
  - [ ] Failure analysis

**Execute**:
```bash
# Test generalization with live search
uv run python scripts/test_generalization.py \
  --optimized-program results/cached_optimization/optimized_program.json \
  --num-companies 12 \
  --output results/generalization/validation_report.md
```

**Expected Output**:
```
Generalization Test Results

Cached Validation F1: 76.3%
Live Search F1: 71.5%
Generalization Gap: 4.8%

✅ GOOD GENERALIZATION
Recommended Action: Proceed with deployment (monitor closely)
```

**Decision Point 4**:
- ✅ Gap <10%? → **Deploy to pilot**
- ⚠️ Gap 10-15%? → Deploy with monitoring plan
- ❌ Gap >15%? → Refine or iterate

---

### Documentation

**After All Phases Complete**:
- [ ] Update `OPTIMIZATION_STRATEGY.md` with actual results
- [ ] Create `DEPLOYMENT_GUIDE.md` if successful
- [ ] Document lessons learned
- [ ] Plan monitoring strategy

---

## Timeline & Resources

### Week-by-Week Plan

**Week 1: Foundation**
- Days 1-2: Phase 1 - Architecture implementation
- Days 3-4: Phase 2 - Model comparison
- Day 5: Review and documentation

**Week 2: Optimization**
- Days 1-2: Phase 3 setup and launch
- Days 3-5: GEPA optimization running (8-10 hours)

**Week 3: Validation**
- Days 1-2: Phase 4 - Generalization testing
- Days 3-4: Analysis and decision
- Day 5: Documentation and planning

**Week 4: Deployment or Iteration** (if needed)
- Deploy to pilot, OR
- Plan refinement iteration

### Resource Requirements

**Compute**:
- Standard laptop/workstation
- ~4 GB RAM for optimization
- ~500 MB disk space for results

**APIs**:
- Perplexity API key (Phase 2 & 4)
- Gemini API key (Phase 3)
- Total cost: $20-35

**Human Time**:
- Phase 1: 4-6 hours (hands-on coding)
- Phase 2: 1-2 hours (setup + monitoring)
- Phase 3: 1-2 hours (setup, rest is automated)
- Phase 4: 2-4 hours (execution + analysis)
- Total: ~10-15 hours hands-on

### Risk Management

**Each Phase Has Decision Point**:
- Can stop if results don't meet expectations
- Can pivot strategy if issues discovered
- Not all-or-nothing like original GEPA plan

**Phased Budget**:
- Phase 1: $0 (code only)
- Phase 2: $5-10 (can stop if baseline poor)
- Phase 3: $15-25 (only if Phase 2 successful)
- Phase 4: $5-10 (only if Phase 3 successful)

---

## Success Metrics

### Technical Metrics

**Primary**:
- ✅ F1 ≥75% on cached validation set
- ✅ Generalization gap <10%
- ✅ Reproducible results (deterministic)

**Secondary**:
- ✅ Confidence calibration (high confidence → high accuracy)
- ✅ False positive rate ≤10%
- ✅ Dutch terminology utilization
- ✅ Source tier awareness

### Process Metrics

**Rigor**:
- ✅ Statistically sound evaluation (stratified CV, bootstrap)
- ✅ Expert validation (Gemini 2.5 Pro reviewed)
- ✅ Clear documentation (single source of truth)
- ✅ Reproducible methodology

**Efficiency**:
- ✅ Phased approach with decision points
- ✅ Architecture reusable for future work
- ✅ No throwaway code
- ✅ Clear success criteria

### Business Metrics

**Deliverables**:
- ✅ Production-ready classifier
- ✅ Systematic improvement methodology
- ✅ Foundation for iteration
- ✅ Deployment guide

**Value**:
- ✅ Higher accuracy than manual approaches
- ✅ Scalable to new companies
- ✅ Maintainable architecture
- ✅ Explainable decisions (reasoning traces)

---

## Key Takeaways

### What Makes This Strategy Sound

1. **Prevents Temporal Overfitting**
   - Cached data = deterministic environment
   - Learns synthesis patterns, not search luck
   - Validated on live search in Phase 4

2. **Statistically Rigorous**
   - Stratified k-fold CV preserves class balance
   - Bootstrap provides confidence intervals
   - Paired tests increase statistical power
   - Effect size calculated (not just p-values)

3. **Architecturally Clean**
   - Dependency injection enables testing
   - Protocol pattern is extensible
   - Separation of concerns (retrieval vs classification)
   - Reusable for future work

4. **Expert-Validated**
   - Gemini 2.5 Pro reviewed approach
   - Architecture-first recommendation adopted
   - Statistical interpretation guidance incorporated
   - Generalization testing framed correctly

5. **Phased with Decision Points**
   - 4 checkpoints to stop/pivot
   - Not all-or-nothing
   - Budget allocated incrementally
   - Learn from each phase

### What Could Go Wrong (and Mitigations)

**Risk 1: Optimized prompt doesn't generalize**
- Likelihood: Medium
- Impact: High
- Mitigation: Phase 4 canary test, data augmentation, manual refinement

**Risk 2: F1 doesn't reach 75% target**
- Likelihood: Low-Medium
- Impact: Medium
- Mitigation: Expand dataset (65 → 100+ companies), try advanced optimizers

**Risk 3: Statistical tests inconclusive (Phase 2)**
- Likelihood: Medium (with n=65)
- Impact: Low
- Mitigation: Choose based on cost, not tiny F1 differences

**Risk 4: Implementation bugs**
- Likelihood: Low (with tests)
- Impact: Medium
- Mitigation: Comprehensive testing, determinism verification

### When to Stop and Reassess

**Stop and reassess if**:
- Phase 2 baseline is <60% F1 → Data quality issue
- Phase 3 validation F1 is <70% → Optimization ineffective
- Phase 4 gap is >15% → Severe overfitting

**In each case**:
- Analyze root cause
- Consult ERROR_DIAGNOSTICS.md for guidance
- Consider alternative approaches
- Update this strategy document

---

## References

### Related Documentation

- **ERROR_DIAGNOSTICS.md**: Expert analysis of temporal overfitting risks (ARCHIVED)
- **GEPA_NEXT_STEP.md**: Original live optimization plan (do NOT use)
- **QUICK_START_ITERATION.md**: Original quick start (do NOT use)
- **IMPROVE_PROMPT.md**: Original improvement proposal (ARCHIVED)
- **docs/ITERATIVE_REFINEMENT.md**: Future iteration framework

### External Resources

- **Clarifai DSPy RAG**: https://docs.clarifai.com/integrations/DSPy/rag-dspy/
- **DSPy Documentation**: https://github.com/stanfordnlp/dspy
- **Perplexity API**: https://docs.perplexity.ai/
- **Statistical Power Analysis**: statsmodels.stats.power

---

## Changelog

**2025-11-18**: Initial strategy created
- Synthesized findings from deep thinking analysis
- Incorporated Gemini 2.5 Pro expert validation
- Addressed bootstrap vs multi-arm bandit challenge
- Adapted Clarifai RAG pattern for this use case
- Reordered phases (architecture first)
- Added comprehensive statistical guidance

---

**Status**: Ready for Phase 1 execution
**Next Step**: Create `src/sevenrad_ee/ai/retrievers.py`
**Decision Owner**: Project team
**Review Date**: After Phase 1 completion
