"""
DSPy greenhouse detection optimization with three-phase validation.

This module implements a systematic three-phase optimization pipeline:
1. Phase 1: Baseline evaluation with unoptimized predictor
2. Phase 2: BootstrapFewShot optimization on training set
3. Phase 3: Validation on hold-out examples

Documentation Type: How-to Guide for running DSPy optimization.
"""

# ruff: noqa: E501
# Long lines in markdown report strings are acceptable for formatting

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import dspy
    from dspy.teleprompt import BootstrapFewShot
except ImportError as e:
    msg = "dspy-ai package required. Install with: uv pip install -e '.[dev]'"
    raise ImportError(msg) from e

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from sevenrad_ee.ai.dspy_evaluation import (
    combined_f1_metric,
    greenhouse_f1_metric,
    growlight_accuracy_metric,
)
from sevenrad_ee.ai.dspy_greenhouse import GreenhouseDetector
from sevenrad_ee.ai.dspy_training_data import (
    get_training_set,
    get_validation_set,
)

console = Console()
logger = logging.getLogger(__name__)


# Type aliases for results
PredictionResult = dict[str, Any]
PhaseResults = dict[str, Any]

# Performance thresholds
STRONG_VALIDATION_THRESHOLD = 0.90
MODERATE_VALIDATION_THRESHOLD = 0.75


def configure_perplexity_api() -> dspy.LM:
    """
    Configure Perplexity Sonar API as DSPy language model.

    Returns:
        Configured DSPy LM instance

    Raises:
        ValueError: If PERPLEXITY_API_KEY environment variable not set

    """
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        msg = (
            "PERPLEXITY_API_KEY environment variable not set. "
            "Export with: export PERPLEXITY_API_KEY='your-key'"
        )
        raise ValueError(msg)

    lm = dspy.LM(
        "perplexity/sonar",
        api_key=api_key,
        api_base="https://api.perplexity.ai",
    )

    return lm


def evaluate_example(
    detector: GreenhouseDetector,
    example: dspy.Example,
) -> PredictionResult:
    """
    Evaluate a single example and return detailed results.

    Args:
        detector: GreenhouseDetector instance
        example: Ground truth example

    Returns:
        Dictionary with prediction, metrics, and metadata

    """
    # Run prediction
    prediction = detector(
        location_name=example.location_name,
        location_area=example.location_area,
    )

    # Convert to Pydantic for validation
    validated = detector.to_pydantic(prediction)

    # Calculate metrics
    greenhouse_score = greenhouse_f1_metric(example, prediction)
    growlight_score = growlight_accuracy_metric(example, prediction)
    combined_score = combined_f1_metric(example, prediction)

    return {
        "location_name": example.location_name,
        "location_area": example.location_area,
        "ground_truth": {
            "is_greenhouse": bool(example.is_greenhouse),
            "uses_growlight": str(example.uses_growlight),
            "species_grown": str(getattr(example, "species_grown", "")),
        },
        "prediction": {
            "is_greenhouse": validated.is_greenhouse,
            "uses_growlight": str(validated.uses_growlight)
            if validated.uses_growlight
            else "None",
            "species_grown": validated.species_grown or [],
            "confidence": validated.confidence,
            "lighting_type": validated.lighting_type or "None",
        },
        "metrics": {
            "greenhouse_f1": greenhouse_score,
            "growlight_accuracy": growlight_score,
            "combined_f1": combined_score,
        },
        "reasoning": validated.reasoning,
        "sources": validated.sources,
    }


def phase1_baseline_evaluation(lm: dspy.LM) -> PhaseResults:
    """
    Phase 1: Evaluate baseline (unoptimized) predictor.

    Args:
        lm: Configured DSPy language model

    Returns:
        Dictionary with baseline results and metrics

    """
    console.print("\n[bold cyan]Phase 1: Baseline Evaluation[/bold cyan]\n")

    with dspy.context(lm=lm):
        detector = GreenhouseDetector()
        training_set = get_training_set()

        predictions: list[PredictionResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Evaluating {len(training_set)} training examples...",
                total=len(training_set),
            )

            for i, example in enumerate(training_set):
                progress.update(
                    task,
                    description=(
                        f"Evaluating {i+1}/{len(training_set)}: "
                        f"{example.location_name}"
                    ),
                    completed=i,
                )

                result = evaluate_example(detector, example)
                predictions.append(result)

            progress.update(task, completed=len(training_set))

    # Calculate aggregate metrics
    avg_greenhouse = sum(p["metrics"]["greenhouse_f1"] for p in predictions) / len(
        predictions
    )
    avg_growlight = sum(p["metrics"]["growlight_accuracy"] for p in predictions) / len(
        predictions
    )
    avg_combined = sum(p["metrics"]["combined_f1"] for p in predictions) / len(
        predictions
    )

    console.print(f"\n[green]✓[/green] Baseline evaluation complete")
    console.print(f"  Greenhouse F1: {avg_greenhouse:.2%}")
    console.print(f"  Growlight Accuracy: {avg_growlight:.2%}")
    console.print(f"  Combined F1: {avg_combined:.2%}")

    return {
        "predictions": predictions,
        "aggregate_metrics": {
            "greenhouse_f1": avg_greenhouse,
            "growlight_accuracy": avg_growlight,
            "combined_f1": avg_combined,
        },
        "n_examples": len(predictions),
    }


def phase2_optimization(lm: dspy.LM, baseline_results: PhaseResults) -> PhaseResults:
    """
    Phase 2: Optimize predictor with BootstrapFewShot.

    Args:
        lm: Configured DSPy language model
        baseline_results: Results from Phase 1 for comparison

    Returns:
        Dictionary with optimized results and improvement metrics

    """
    console.print("\n[bold cyan]Phase 2: BootstrapFewShot Optimization[/bold cyan]\n")

    with dspy.context(lm=lm):
        # Get training data
        training_set = get_training_set()

        # Initialize optimizer
        console.print("Initializing BootstrapFewShot teleprompter...")
        optimizer = BootstrapFewShot(
            metric=combined_f1_metric,
            max_bootstrapped_demos=5,
            max_labeled_demos=10,
        )

        # Compile optimized predictor
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Compiling optimized predictor...", total=None)

            base_detector = GreenhouseDetector()
            optimized_detector = optimizer.compile(
                student=base_detector,
                trainset=training_set,
            )

            progress.update(task, completed=True)

        console.print("[green]✓[/green] Optimization complete")

        # Re-evaluate with optimized model
        predictions: list[PredictionResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Re-evaluating {len(training_set)} examples with optimized model...",
                total=len(training_set),
            )

            for i, example in enumerate(training_set):
                progress.update(
                    task,
                    description=(
                        f"Evaluating {i+1}/{len(training_set)}: "
                        f"{example.location_name}"
                    ),
                    completed=i,
                )

                result = evaluate_example(optimized_detector, example)
                predictions.append(result)

            progress.update(task, completed=len(training_set))

    # Calculate aggregate metrics
    avg_greenhouse = sum(p["metrics"]["greenhouse_f1"] for p in predictions) / len(
        predictions
    )
    avg_growlight = sum(p["metrics"]["growlight_accuracy"] for p in predictions) / len(
        predictions
    )
    avg_combined = sum(p["metrics"]["combined_f1"] for p in predictions) / len(
        predictions
    )

    # Calculate improvements
    baseline_combined = baseline_results["aggregate_metrics"]["combined_f1"]
    improvement = avg_combined - baseline_combined
    improvement_pct = (
        (improvement / baseline_combined * 100) if baseline_combined > 0 else 0
    )

    console.print(f"\n[green]✓[/green] Optimized evaluation complete")
    console.print(f"  Greenhouse F1: {avg_greenhouse:.2%}")
    console.print(f"  Growlight Accuracy: {avg_growlight:.2%}")
    console.print(f"  Combined F1: {avg_combined:.2%}")
    console.print(
        f"\n  [bold]Improvement: {improvement:+.2%} ({improvement_pct:+.1f}%)[/bold]"
    )

    return {
        "predictions": predictions,
        "aggregate_metrics": {
            "greenhouse_f1": avg_greenhouse,
            "growlight_accuracy": avg_growlight,
            "combined_f1": avg_combined,
        },
        "improvement": {
            "absolute": improvement,
            "relative_pct": improvement_pct,
        },
        "optimized_detector": optimized_detector,
        "n_examples": len(predictions),
    }


def phase3_validation(
    lm: dspy.LM,
    optimized_detector: GreenhouseDetector,
) -> PhaseResults:
    """
    Phase 3: Validate optimized model on hold-out set.

    Args:
        lm: Configured DSPy language model
        optimized_detector: Optimized detector from Phase 2

    Returns:
        Dictionary with validation results

    """
    console.print("\n[bold cyan]Phase 3: Validation on Hold-out Set[/bold cyan]\n")

    with dspy.context(lm=lm):
        validation_set = get_validation_set()

        predictions: list[PredictionResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Validating {len(validation_set)} hold-out examples...",
                total=len(validation_set),
            )

            for i, example in enumerate(validation_set):
                progress.update(
                    task,
                    description=(
                        f"Validating {i+1}/{len(validation_set)}: "
                        f"{example.location_name}"
                    ),
                    completed=i,
                )

                result = evaluate_example(optimized_detector, example)
                predictions.append(result)

            progress.update(task, completed=len(validation_set))

    # Calculate aggregate metrics
    avg_greenhouse = sum(p["metrics"]["greenhouse_f1"] for p in predictions) / len(
        predictions
    )
    avg_growlight = sum(p["metrics"]["growlight_accuracy"] for p in predictions) / len(
        predictions
    )
    avg_combined = sum(p["metrics"]["combined_f1"] for p in predictions) / len(
        predictions
    )

    console.print(f"\n[green]✓[/green] Validation complete")
    console.print(f"  Greenhouse F1: {avg_greenhouse:.2%}")
    console.print(f"  Growlight Accuracy: {avg_growlight:.2%}")
    console.print(f"  Combined F1: {avg_combined:.2%}")

    return {
        "predictions": predictions,
        "aggregate_metrics": {
            "greenhouse_f1": avg_greenhouse,
            "growlight_accuracy": avg_growlight,
            "combined_f1": avg_combined,
        },
        "n_examples": len(predictions),
    }


def generate_markdown_report(
    phase1_results: PhaseResults,
    phase2_results: PhaseResults,
    phase3_results: PhaseResults,
    output_path: Path,
) -> None:
    """
    Generate comprehensive markdown report.

    Args:
        phase1_results: Baseline evaluation results
        phase2_results: Optimization results
        phase3_results: Validation results
        output_path: Path to save report

    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Markdown report content (long lines acceptable for formatting)
    report = f"""# DSPy Greenhouse Detection Optimization Report

**Generated:** {timestamp}

## Executive Summary

This report documents the three-phase optimization of greenhouse growlight detection using DSPy's BootstrapFewShot teleprompter.

### Key Findings

- **Training Set**: {phase1_results['n_examples']} examples
- **Validation Set**: {phase3_results['n_examples']} hold-out examples
- **Baseline Combined F1**: {phase1_results['aggregate_metrics']['combined_f1']:.2%}
- **Optimized Combined F1**: {phase2_results['aggregate_metrics']['combined_f1']:.2%}
- **Improvement**: {phase2_results['improvement']['absolute']:+.2%} ({phase2_results['improvement']['relative_pct']:+.1f}%)
- **Validation F1**: {phase3_results['aggregate_metrics']['combined_f1']:.2%}

---

## Phase 1: Baseline Evaluation

Evaluated unoptimized GreenhouseDetector on {phase1_results['n_examples']} training examples.

### Aggregate Metrics

| Metric | Score |
|--------|-------|
| Greenhouse F1 | {phase1_results['aggregate_metrics']['greenhouse_f1']:.2%} |
| Growlight Accuracy | {phase1_results['aggregate_metrics']['growlight_accuracy']:.2%} |
| Combined F1 | {phase1_results['aggregate_metrics']['combined_f1']:.2%} |

### Per-Example Results

| Location | Is GH | Uses Growlight | Combined F1 |
|----------|-------|----------------|-------------|
"""

    for pred in phase1_results["predictions"]:
        report += f"| {pred['location_name']} | "
        report += f"{'✓' if pred['prediction']['is_greenhouse'] else '✗'} | "
        report += f"{pred['prediction']['uses_growlight']} | "
        report += f"{pred['metrics']['combined_f1']:.2%} |\n"

    report += f"""
---

## Phase 2: BootstrapFewShot Optimization

Applied DSPy BootstrapFewShot optimization with:
- `max_bootstrapped_demos=5`
- `max_labeled_demos=10`
- `metric=combined_f1_metric`

### Optimized Metrics

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Greenhouse F1 | {phase1_results['aggregate_metrics']['greenhouse_f1']:.2%} | {phase2_results['aggregate_metrics']['greenhouse_f1']:.2%} | {phase2_results['aggregate_metrics']['greenhouse_f1'] - phase1_results['aggregate_metrics']['greenhouse_f1']:+.2%} |
| Growlight Accuracy | {phase1_results['aggregate_metrics']['growlight_accuracy']:.2%} | {phase2_results['aggregate_metrics']['growlight_accuracy']:.2%} | {phase2_results['aggregate_metrics']['growlight_accuracy'] - phase1_results['aggregate_metrics']['growlight_accuracy']:+.2%} |
| Combined F1 | {phase1_results['aggregate_metrics']['combined_f1']:.2%} | {phase2_results['aggregate_metrics']['combined_f1']:.2%} | {phase2_results['improvement']['absolute']:+.2%} |

**Relative Improvement**: {phase2_results['improvement']['relative_pct']:+.1f}%

---

## Phase 3: Validation on Hold-out Set

Validated optimized model on {phase3_results['n_examples']} never-seen examples.

### Validation Metrics

| Metric | Score |
|--------|-------|
| Greenhouse F1 | {phase3_results['aggregate_metrics']['greenhouse_f1']:.2%} |
| Growlight Accuracy | {phase3_results['aggregate_metrics']['growlight_accuracy']:.2%} |
| Combined F1 | {phase3_results['aggregate_metrics']['combined_f1']:.2%} |

### Validation Examples

| Location | Area | Prediction | Ground Truth | Match |
|----------|------|------------|--------------|-------|
"""

    for pred in phase3_results["predictions"]:
        match = "✓" if pred["metrics"]["combined_f1"] == 1.0 else "✗"
        report += f"| {pred['location_name']} | "
        report += f"{pred['location_area']} | "
        report += f"GH={pred['prediction']['is_greenhouse']}, "
        report += f"GL={pred['prediction']['uses_growlight']} | "
        report += f"GH={pred['ground_truth']['is_greenhouse']}, "
        report += f"GL={pred['ground_truth']['uses_growlight']} | "
        report += f"{match} |\n"

    report += """
---

## Detailed Predictions

### Phase 3 Validation Details
"""

    for pred in phase3_results["predictions"]:
        report += f"""
#### {pred['location_name']} ({pred['location_area']})

**Ground Truth:**
- Is Greenhouse: {pred['ground_truth']['is_greenhouse']}
- Uses Growlight: {pred['ground_truth']['uses_growlight']}
- Species: {pred['ground_truth']['species_grown']}

**Prediction:**
- Is Greenhouse: {pred['prediction']['is_greenhouse']}
- Uses Growlight: {pred['prediction']['uses_growlight']}
- Species: {', '.join(pred['prediction']['species_grown']) if pred['prediction']['species_grown'] else 'None'}
- Confidence: {pred['prediction']['confidence']:.2%}
- Lighting Type: {pred['prediction']['lighting_type']}

**Metrics:**
- Greenhouse F1: {pred['metrics']['greenhouse_f1']:.2%}
- Growlight Accuracy: {pred['metrics']['growlight_accuracy']:.2%}
- Combined F1: {pred['metrics']['combined_f1']:.2%}

**Reasoning:**
{pred['reasoning']}

**Sources:**
"""
        for source in pred["sources"]:
            report += f"- {source}\n"

    report += """
---

## Recommendations

### Model Performance
"""

    if phase2_results["improvement"]["absolute"] > 0:
        report += f"""
✓ Optimization successful with {phase2_results['improvement']['relative_pct']:+.1f}% improvement
"""
    else:
        report += """
⚠ Optimization did not improve performance - consider:
  - Increasing training set size
  - Adjusting optimization parameters
  - Reviewing example quality
"""

    if phase3_results["aggregate_metrics"]["combined_f1"] >= STRONG_VALIDATION_THRESHOLD:
        report += """
✓ Strong validation performance (≥90% F1) indicates good generalization
"""
    elif (
        phase3_results["aggregate_metrics"]["combined_f1"]
        >= MODERATE_VALIDATION_THRESHOLD
    ):
        report += """
⚠ Moderate validation performance (75-90% F1) - consider adding more training examples
"""
    else:
        report += """
✗ Low validation performance (<75% F1) - investigate overfitting or data quality issues
"""

    report += """
### Next Steps

1. **Deploy optimized model** for production use
2. **Monitor performance** on real-world examples
3. **Collect feedback** to expand training set
4. **Re-optimize** quarterly with new data

---

*Report generated with sevenrad-ee DSPy greenhouse detection optimization pipeline*
"""

    output_path.write_text(report)
    console.print(f"\n[green]✓[/green] Report saved to: {output_path}")


def save_optimized_model(
    optimized_detector: GreenhouseDetector,
    output_path: Path,
) -> None:
    """
    Save optimized detector to disk using DSPy's save method.

    Args:
        optimized_detector: Optimized detector from Phase 2
        output_path: Path to save model

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use DSPy's save method instead of pickle
    optimized_detector.save(str(output_path))

    console.print(f"[green]✓[/green] Optimized model saved to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run three-phase DSPy greenhouse detection optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full optimization pipeline
  uv run python -m sevenrad_ee.operations.optimize_greenhouse_detection

  # Specify custom output directory
  uv run python -m sevenrad_ee.operations.optimize_greenhouse_detection \\
    --output-dir results/custom

  # Save model to specific location
  uv run python -m sevenrad_ee.operations.optimize_greenhouse_detection \\
    --model-path models/greenhouse_v2.pkl
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for reports and artifacts (default: results/)",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/greenhouse_detector_optimized.pkl"),
        help=(
            "Path to save optimized model "
            "(default: models/greenhouse_detector_optimized.pkl)"
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Run the three-phase optimization pipeline."""
    # Display header
    console.print(
        Panel.fit(
            "[bold cyan]DSPy Greenhouse Detection Optimization[/bold cyan]\n"
            "Three-Phase Pipeline: Baseline → Optimization → Validation",
            border_style="cyan",
        )
    )

    args = parse_arguments()

    try:
        # Configure Perplexity API
        console.print("\n[bold]Configuring Perplexity Sonar API...[/bold]")
        lm = configure_perplexity_api()
        dspy.configure(lm=lm)
        console.print("[green]✓[/green] API configured")

        # Phase 1: Baseline
        phase1_results = phase1_baseline_evaluation(lm)

        # Phase 2: Optimization
        phase2_results = phase2_optimization(lm, phase1_results)

        # Phase 3: Validation
        phase3_results = phase3_validation(lm, phase2_results["optimized_detector"])

        # Save optimized model
        console.print("\n[bold]Saving optimized model...[/bold]")
        save_optimized_model(
            phase2_results["optimized_detector"],
            args.model_path,
        )

        # Generate report
        console.print("\n[bold]Generating markdown report...[/bold]")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = args.output_dir / f"greenhouse_optimization_report_{timestamp}.md"
        args.output_dir.mkdir(parents=True, exist_ok=True)

        generate_markdown_report(
            phase1_results,
            phase2_results,
            phase3_results,
            report_path,
        )

        # Display summary
        console.print("\n" + "=" * 60)
        console.print("[bold green]Optimization Complete![/bold green]")
        console.print("=" * 60)

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Phase", style="white")
        table.add_column("Combined F1", style="green")

        table.add_row(
            "Phase 1: Baseline",
            f"{phase1_results['aggregate_metrics']['combined_f1']:.2%}",
        )
        table.add_row(
            "Phase 2: Optimized",
            f"{phase2_results['aggregate_metrics']['combined_f1']:.2%}",
        )
        table.add_row(
            "Phase 3: Validation",
            f"{phase3_results['aggregate_metrics']['combined_f1']:.2%}",
        )

        console.print(table)

        improvement_abs = phase2_results["improvement"]["absolute"]
        improvement_pct = phase2_results["improvement"]["relative_pct"]
        console.print(
            f"\n[bold]Improvement:[/bold] {improvement_abs:+.2%} "
            f"({improvement_pct:+.1f}%)"
        )

        return 0

    except Exception as e:
        console.print(f"\n[red]✗[/red] Error: {e}", style="bold red")
        logger.exception("Optimization failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
