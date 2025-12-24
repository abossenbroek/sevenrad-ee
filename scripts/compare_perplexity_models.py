"""
Compare Perplexity Sonar vs Sonar Reasoning Pro using cached data.

This script executes Phase 2 of the DSPy optimization strategy to establish a
baseline F1 score and select the best student model for Phase 3 GEPA optimization.

Uses stratified 5-fold cross-validation with bootstrapping for rigorous statistical
comparison. Analyzes both models using frozen cached data to ensure deterministic,
reproducible evaluation.

Statistical Approach:
    - Stratified k-fold CV: Preserves class balance (critical with n=65)
    - Bootstrap CI: Quantifies uncertainty in F1 estimates
    - Paired Wilcoxon test: Non-parametric comparison of fold scores
    - Cohen's d: Effect size measurement for practical significance

Decision Criteria (from expert analysis):
    - Strong evidence (p<0.05 AND d>0.5): Select Sonar Reasoning Pro
    - Moderate evidence (d>0.3): Select Sonar Reasoning Pro if budget allows
    - Insufficient evidence: Use cheaper/faster Sonar

Example Usage:
    # Full comparison with report
    uv run python scripts/compare_perplexity_models.py \\
        --cache-dir data/research \\
        --output-path results/diagnostics/model_comparison_report.md

    # Quick test with first 10 companies
    uv run python scripts/compare_perplexity_models.py \\
        --cache-dir data/research \\
        --dry-run
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.stats import wilcoxon

try:
    import dspy
except ImportError as e:
    msg = (
        "dspy-ai package is required. Install with: "
        "uv pip install -e '.[dev]' or pip install dspy-ai~=2.5.0"
    )
    raise ImportError(msg) from e

from scripts.evaluate_with_cv import (
    EvaluationReport,
    evaluate_with_repeated_stratified_cv,
    evaluate_with_stratified_cv,
)
from sevenrad_ee.ai.data_utils import load_cached_companies
from sevenrad_ee.ai.dspy_greenhouse import GreenhouseDetector
from sevenrad_ee.ai.retrievers import CachedRetriever

# --- Configuration ---
# Model names following project's DSPy configuration pattern
# Based on OPTIMIZATION_STRATEGY.md and official Perplexity API docs (2025)
MODELS_TO_COMPARE = {
    "Sonar": "perplexity/sonar",  # Basic search (Llama 3.3 70B)
    "Sonar Reasoning Pro": "perplexity/sonar-reasoning-pro",  # Reasoning + search (DeepSeek-R1)
}
CHEAPER_MODEL_KEY = "Sonar"
RANDOM_STATE = 42
# Phase 2.5: Repeated CV for variance reduction (21% → <10% std dev target)
N_FOLDS = 10  # Increased from 5 for more stable estimates
N_REPETITIONS = 3  # Multiple shuffles for robust estimation
N_BOOTSTRAP = 1000

console = Console()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Perplexity models for greenhouse detection (Phase 2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full comparison with report generation
  uv run python scripts/compare_perplexity_models.py \\
      --cache-dir data/research \\
      --output-path results/diagnostics/model_comparison_report.md

  # Dry run with first 10 companies (testing)
  uv run python scripts/compare_perplexity_models.py \\
      --cache-dir data/research \\
      --dry-run

Decision Criteria:
  - Strong evidence (p<0.05 AND Cohen's d>0.5): Select Sonar Reasoning Pro
  - Moderate evidence (Cohen's d>0.3): Select Sonar Reasoning Pro if budget allows
  - Insufficient evidence: Use cheaper/faster Sonar

Statistical Method:
  - Stratified 5-fold cross-validation (preserves class balance)
  - Bootstrap resampling (1000 samples per fold for confidence intervals)
  - Paired Wilcoxon signed-rank test (non-parametric comparison)
  - Cohen's d effect size (practical significance measure)
        """,
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory containing cached research JSON files (data/research).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path to save the markdown comparison report (optional).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test with first 10 companies only (for validation).",
    )
    parser.add_argument(
        "--n-repetitions",
        type=int,
        default=N_REPETITIONS,
        choices=range(1, 11),
        metavar="[1-10]",
        help=f"Number of CV repetitions (default: {N_REPETITIONS}). "
        "More repetitions = more stable variance estimates but longer runtime.",
    )

    return parser.parse_args()


def run_evaluation(
    model_name: str,
    model_id: str,
    detector: GreenhouseDetector,
    companies: list[dict[str, str | bool]],
    n_repetitions: int,
    dry_run: bool = False,
) -> EvaluationReport:
    """
    Configure DSPy and run evaluation for a single model.

    Args:
        model_name: Display name for the model (e.g., "Sonar Pro")
        model_id: DSPy model identifier (e.g., "perplexity/sonar-pro")
        detector: GreenhouseDetector instance with CachedRetriever
        companies: List of company dictionaries with ground truth labels
        n_repetitions: Number of CV repetitions (1 for single-pass, >1 for repeated)
        dry_run: If True, use only first 10 companies for testing

    Returns:
        EvaluationReport with F1 scores and confidence intervals

    Raises:
        RuntimeError: If DSPy configuration fails

    """
    console.print(f"\n[cyan]Evaluating {model_name} ({model_id})...[/cyan]")

    # Configure DSPy LM for this run
    try:
        lm = dspy.LM(model_id, max_tokens=4096)
        dspy.configure(lm=lm)
    except Exception as e:
        console.print(
            f"[bold red]✗ Error configuring DSPy for {model_id}: {e}[/bold red]"
        )
        console.print(
            "\n[yellow]Tip:[/yellow] Ensure your PERPLEXITY_API_KEY is set correctly:"
        )
        console.print("  export PERPLEXITY_API_KEY='your-api-key'")
        raise RuntimeError from e

    # Use subset for dry run
    eval_companies = companies[:10] if dry_run else companies

    if dry_run:
        console.print(f"[yellow]Dry run: Using first {len(eval_companies)} companies[/yellow]")

    # Use repeated CV for variance reduction (Phase 2.5)
    if n_repetitions > 1:
        console.print(
            f"[cyan]Running {N_FOLDS}-fold CV × {n_repetitions} repetitions "
            f"(total: {N_FOLDS * n_repetitions} folds)[/cyan]"
        )
        return evaluate_with_repeated_stratified_cv(
            detector,
            eval_companies,
            n_folds=N_FOLDS,
            n_repetitions=n_repetitions,
            n_bootstrap=N_BOOTSTRAP,
            random_state=RANDOM_STATE,
        )
    else:
        # Single-pass CV (backward compatibility)
        console.print(f"[cyan]Running {N_FOLDS}-fold CV (single pass)[/cyan]")
        return evaluate_with_stratified_cv(
            detector,
            eval_companies,
            n_folds=N_FOLDS,
            n_bootstrap=N_BOOTSTRAP,
            random_state=RANDOM_STATE,
        )


def create_report(
    results: dict[str, EvaluationReport],
    stats: dict[str, float | str],
    output_path: Path,
) -> None:
    """
    Generate and save markdown report of the comparison.

    Args:
        results: Dictionary mapping model names to EvaluationReport instances
        stats: Statistical summary including p-value, effect size, recommendation
        output_path: Path where markdown report should be saved

    """
    # Extract numeric values for formatting
    p_value = float(stats["p_value"])
    effect_size = float(stats["effect_size"])
    difference = float(stats["difference"])
    baseline_f1 = float(stats["baseline_f1"])
    recommendation = str(stats["recommendation"])
    selected_model = str(stats["selected_model"])
    # Determine if repeated CV was used
    n_reps = results['Sonar'].n_repetitions
    cv_method = (
        f"Stratified {N_FOLDS}-fold × {n_reps} repetitions"
        if n_reps > 1
        else f"Stratified {N_FOLDS}-fold"
    )
    total_folds = N_FOLDS * n_reps

    report_content = f"""# Phase 2: Perplexity Model Comparison Report

**Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Purpose**: Establish baseline F1 score and select student model for Phase 3 GEPA optimization
**Phase**: {f"2.5 (Repeated CV for Variance Reduction)" if n_reps > 1 else "2 (Baseline & Model Selection)"}

---

## Executive Summary

This report details the comparison between Perplexity Sonar and Sonar Reasoning Pro
to establish a baseline F1 score and select a student model for Phase 3 GEPA optimization.

{f"**Variance Reduction**: Using repeated CV to reduce std dev from 21% to <10% target" if n_reps > 1 else ""}

**Recommendation**: {recommendation}
**Selected Model for Phase 3**: `{selected_model}`
**Established Baseline F1**: {baseline_f1:.2%}

---

## Summary Results

| Model                 | Mean F1     | Std Dev F1 | 95% Confidence Interval |
|-----------------------|-------------|------------|-------------------------|
| Sonar                 | {results['Sonar'].mean_f1:.2%}      | {results['Sonar'].std_f1:.2%} | [{results['Sonar'].confidence_interval_95[0]:.2%}, {results['Sonar'].confidence_interval_95[1]:.2%}] |
| Sonar Reasoning Pro   | {results['Sonar Reasoning Pro'].mean_f1:.2%} | {results['Sonar Reasoning Pro'].std_f1:.2%} | [{results['Sonar Reasoning Pro'].confidence_interval_95[0]:.2%}, {results['Sonar Reasoning Pro'].confidence_interval_95[1]:.2%}] |

**Difference**: +{difference:.2%} F1 (Sonar Reasoning Pro - Sonar)

---

## Statistical Analysis

### Method

- **Evaluation**: {cv_method} with bootstrap resampling
- **Total Folds**: {total_folds} ({N_FOLDS} folds × {n_reps} repetitions)
- **Test**: Wilcoxon signed-rank test (paired, non-parametric)
- **Hypothesis**: Sonar Pro F1 > Sonar F1 (one-sided test)
- **Bootstrap**: 1000 samples per fold for confidence interval estimation
- **Random State**: {RANDOM_STATE} (reproducible results)

### Results

- **p-value**: {p_value:.4f}
  - Interpretation: {"Statistically significant (p<0.05)" if p_value < 0.05 else "Not statistically significant (p≥0.05)"}
- **Effect Size (Cohen's d)**: {effect_size:.2f}
  - Interpretation: {"Large effect (d>0.8)" if effect_size > 0.8 else "Medium effect (d>0.5)" if effect_size > 0.5 else "Small-medium effect (d>0.3)" if effect_size > 0.3 else "Small effect (d<0.3)"}

### Fold-by-Fold Breakdown

| Fold | Sonar F1 | Sonar Reasoning Pro F1 | Difference |
|------|----------|------------------------|------------|
"""

    # Add fold-by-fold comparison
    for i, (sonar_f1, pro_f1) in enumerate(
        zip(results["Sonar"].fold_scores, results["Sonar Reasoning Pro"].fold_scores), 1
    ):
        diff = pro_f1 - sonar_f1
        report_content += f"| {i}    | {sonar_f1:.2%}    | {pro_f1:.2%}      | {diff:+.2%}     |\n"

    # Add repetition-by-repetition breakdown if repeated CV was used
    if n_reps > 1 and results["Sonar"].repetition_scores:
        report_content += f"""

### Repetition-by-Repetition Breakdown

Showing mean F1 per repetition (each repetition uses a different random shuffle):

| Repetition | Sonar Mean F1 | Sonar Pro Mean F1 | Difference |
|------------|---------------|-------------------|------------|
"""
        for i, (sonar_rep_f1, pro_rep_f1) in enumerate(
            zip(results["Sonar"].repetition_scores, results["Sonar Reasoning Pro"].repetition_scores), 1
        ):
            diff_rep = pro_rep_f1 - sonar_rep_f1
            report_content += f"| {i}          | {sonar_rep_f1:.2%}         | {pro_rep_f1:.2%}           | {diff_rep:+.2%}      |\n"

        sonar_rep_std = np.std(results["Sonar"].repetition_scores, ddof=1) if len(results["Sonar"].repetition_scores) > 1 else 0.0
        pro_rep_std = np.std(results["Sonar Reasoning Pro"].repetition_scores, ddof=1) if len(results["Sonar Reasoning Pro"].repetition_scores) > 1 else 0.0

        report_content += f"""

**Repetition Variance Analysis:**
- Sonar std dev across repetitions: {sonar_rep_std:.2%}
- Sonar Pro std dev across repetitions: {pro_rep_std:.2%}

This shows the stability of each model across different data shuffles.
"""

    report_content += f"""
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

**This comparison**: {recommendation}

---

## Conclusion

- **Selected Model**: {selected_model}
- **Baseline F1**: {baseline_f1:.2%}
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
"""

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content.strip())

    console.print(f"\n[green]✓ Report saved to {output_path}[/green]")


def main(args: argparse.Namespace) -> int:
    """
    Main function to run the model comparison.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)

    """
    # Display header
    console.print(
        Panel.fit(
            "[bold cyan]Phase 2: Baseline & Model Selection[/bold cyan]\n"
            "Comparing Perplexity Sonar vs Sonar Reasoning Pro using cached data\n"
            "Statistical method: Stratified 5-fold CV + Bootstrap",
            border_style="cyan",
        )
    )

    # Step 1: Load dataset
    try:
        console.print(f"\n[bold]Loading companies from {args.cache_dir}...[/bold]")
        companies = load_cached_companies(args.cache_dir)
        console.print(f"[green]✓ Loaded {len(companies)} companies[/green]")

        if len(companies) != 65 and not args.dry_run:
            console.print(
                f"[yellow]⚠ Warning: Expected 65 companies, but found {len(companies)}[/yellow]"
            )

        if not companies:
            console.print(
                "[bold red]✗ No companies loaded. Aborting.[/bold red]",
                style="bold red",
            )
            return 1

    except (FileNotFoundError, ValueError) as e:
        console.print(f"[bold red]✗ Error loading data: {e}[/bold red]")
        return 1

    # Step 2: Setup retriever and detector
    console.print("\n[bold]Setting up CachedRetriever and GreenhouseDetector...[/bold]")
    try:
        cached_retriever = CachedRetriever(cache_dir=args.cache_dir)
        detector = GreenhouseDetector(retriever=cached_retriever)
        console.print("[green]✓ Detector configured with cached retriever[/green]")
    except Exception as e:
        console.print(f"[bold red]✗ Error setting up detector: {e}[/bold red]")
        return 1

    # Step 3: Evaluate each model
    cv_desc = (
        f"{N_FOLDS}-fold × {args.n_repetitions} repetitions"
        if args.n_repetitions > 1
        else f"{N_FOLDS}-fold"
    )
    console.print(f"\n[bold]Running stratified {cv_desc} cross-validation...[/bold]")
    results: dict[str, EvaluationReport] = {}

    try:
        for model_name, model_id in MODELS_TO_COMPARE.items():
            results[model_name] = run_evaluation(
                model_name,
                model_id,
                detector,
                companies,
                n_repetitions=args.n_repetitions,
                dry_run=args.dry_run,
            )
    except (RuntimeError, Exception) as e:
        console.print(f"[bold red]✗ Evaluation failed: {e}[/bold red]")
        return 1

    # Step 4: Perform statistical comparison
    console.print("\n[bold]Performing statistical analysis...[/bold]")
    scores_sonar = results[CHEAPER_MODEL_KEY].fold_scores
    scores_pro = results["Sonar Reasoning Pro"].fold_scores

    # Paired Wilcoxon test (one-sided: is Pro better?)
    statistic, p_value = wilcoxon(scores_pro, scores_sonar, alternative="greater")

    # Effect size (Cohen's d for paired data)
    diff = np.array(scores_pro) - np.array(scores_sonar)
    effect_size = (
        np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0
    )

    # Step 5: Apply decision logic (from expert analysis)
    if p_value < 0.05 and effect_size > 0.5:
        recommendation = "Strong evidence for Sonar Reasoning Pro"
        selected_model = MODELS_TO_COMPARE["Sonar Reasoning Pro"]
    elif effect_size > 0.3:
        recommendation = "Moderate evidence for Sonar Reasoning Pro; adopt if budget allows"
        selected_model = MODELS_TO_COMPARE["Sonar Reasoning Pro"]
    else:
        recommendation = "Insufficient evidence for a difference; use faster/cheaper Sonar"
        selected_model = MODELS_TO_COMPARE[CHEAPER_MODEL_KEY]

    baseline_f1 = max(
        results[CHEAPER_MODEL_KEY].mean_f1, results["Sonar Reasoning Pro"].mean_f1
    )

    # Step 6: Display results in a table
    table = Table(title="Model Comparison Results", show_header=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Mean F1", justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_column("95% CI", justify="right")

    for name, res in results.items():
        table.add_row(
            name,
            f"{res.mean_f1:.2%}",
            f"{res.std_f1:.2%}",
            f"[{res.confidence_interval_95[0]:.2%}, {res.confidence_interval_95[1]:.2%}]",
        )

    console.print("\n")
    console.print(table)

    # Display statistical analysis
    console.print(f"\n[bold]Statistical Analysis:[/bold]")
    console.print(f"  Difference: +{np.mean(diff):.2%} F1 (Sonar Reasoning Pro - Sonar)")
    console.print(f"  p-value: {p_value:.4f}")
    console.print(f"  Effect size (Cohen's d): {effect_size:.2f}")

    console.print(f"\n[bold green]Recommendation:[/bold green] {recommendation}")
    console.print(f"[bold green]Selected for Phase 3:[/bold green] {selected_model}")
    console.print(f"[bold yellow]Established Baseline F1:[/bold yellow] {baseline_f1:.2%}")

    # Step 7: Generate and save report (if requested)
    if args.output_path and not args.dry_run:
        stats_summary = {
            "p_value": p_value,
            "effect_size": effect_size,
            "difference": np.mean(diff),
            "recommendation": recommendation,
            "selected_model": selected_model,
            "baseline_f1": baseline_f1,
        }
        create_report(results, stats_summary, args.output_path)

    if args.dry_run:
        console.print(
            "\n[yellow]Dry run completed. Run without --dry-run for full evaluation.[/yellow]"
        )

    return 0


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
