"""
Run full Phase 3 GEPA optimization with PerplexityLM.

This script runs the complete GEPA optimization using:
- Student Model: PerplexityLM (llama-3.1-sonar-large-128k-chat) with native structured outputs
- Teacher Model: Gemini 2.5 Pro for reflection and feedback
- Optimization: GEPA with auto='medium' for comprehensive optimization

Expected runtime: 1-3 hours depending on dataset size
Expected cost: Based on dry run estimate (see dry_run_cost_estimate.py)

Documentation Type: Script (Code to Run)
Part of: Phase 3 - GEPA Optimization Strategy
"""

import argparse
import contextvars
import json
import logging
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

try:
    import dspy
    from dspy.teleprompt import GEPA
except ImportError as e:
    msg = "dspy-ai package is required. Install with: uv pip install -e '.[dev]'"
    raise ImportError(msg) from e

from sevenrad_ee.ai.dspy_evaluation import gepa_compatible_metric
from sevenrad_ee.ai.dspy_greenhouse import GreenhouseDetector
from sevenrad_ee.ai.dspy_perplexity import PerplexityLM
from sevenrad_ee.ai.retrievers import CachedRetriever
from sevenrad_ee.ai.utils.logging_setup import (
    get_logger,
    phase_var,
    request_id_var,
    setup_logging,
)
from sklearn.model_selection import train_test_split

# Initialize logging FIRST (before creating any loggers)
setup_logging(log_file="gepa_optimization_debug.log")

console = Console()
logger = get_logger(__name__)

# Random seed for reproducibility
SEED = 42

# Metrics tracking
metrics = {
    "optimization": {
        "start_time": None,
        "end_time": None,
        "total_duration_seconds": 0.0,
    },
    "evaluation": {
        "start_time": None,
        "end_time": None,
        "total_examples": 0,
        "successful_predictions": 0,
        "failed_predictions": 0,
        "total_duration_seconds": 0.0,
    },
}


def load_greenhouse_data(cache_dir: Path) -> list[dspy.Example]:
    """
    Load cached greenhouse data from JSON files.

    Args:
        cache_dir: Directory containing research JSON files

    Returns:
        List of dspy.Example objects with company data

    """
    examples = []
    json_files = list(cache_dir.glob("*.json"))

    console.print(f"[cyan]Loading data from {len(json_files)} JSON files...[/cyan]")

    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text())

            # Extract fields from JSON
            company_name = data.get("company", "")
            location = data.get("location", "")
            is_greenhouse = data.get("is_greenhouse", False)
            uses_growlight = data.get("uses_growlight", "UNKNOWN")

            # Create dspy.Example (using parameter names that match GreenhouseDetector.forward())
            example = dspy.Example(
                location_name=company_name,
                location_area=location,
                is_greenhouse=is_greenhouse,
                uses_growlight=uses_growlight,
            ).with_inputs("location_name", "location_area")

            examples.append(example)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load {json_file}: {e}")
            continue

    console.print(f"[green]✓[/green] Loaded {len(examples)} examples")
    return examples


def run_optimization(
    cache_dir: Path,
    output_dir: Path,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> dict[str, Any]:
    """
    Run full GEPA optimization with PerplexityLM.

    Args:
        cache_dir: Directory with cached research data
        output_dir: Directory to save optimization results
        test_size: Proportion of data to use for final testing
        val_size: Proportion of training data to use for validation

    Returns:
        Dictionary with optimization results and metrics

    """
    # Set random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    all_data = load_greenhouse_data(cache_dir)

    if len(all_data) == 0:
        msg = f"No data found in {cache_dir}"
        raise ValueError(msg)

    console.print(f"[cyan]Total examples: {len(all_data)}[/cyan]")

    # Split data: train+val / test
    train_val_data, test_data = train_test_split(
        all_data,
        test_size=test_size,
        random_state=SEED,
        stratify=[ex.is_greenhouse for ex in all_data],
    )

    # Split train_val into train / val
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_size,
        random_state=SEED,
        stratify=[ex.is_greenhouse for ex in train_val_data],
    )

    console.print(
        f"[cyan]Split: {len(train_data)} train, "
        f"{len(val_data)} val, {len(test_data)} test[/cyan]"
    )

    # Save data splits
    splits_file = output_dir / "data_splits.json"
    splits_file.write_text(
        json.dumps(
            {
                "total": len(all_data),
                "train": len(train_data),
                "val": len(val_data),
                "test": len(test_data),
                "test_size": test_size,
                "val_size": val_size,
                "seed": SEED,
            },
            indent=2,
        )
    )

    # Create retriever and detector
    cached_retriever = CachedRetriever(cache_dir=cache_dir)
    detector = GreenhouseDetector(retriever=cached_retriever)

    # Configure student model (PerplexityLM) for predictions
    console.print("\n[yellow]Configuring models...[/yellow]")
    student_lm = PerplexityLM(
        model="llama-3.1-sonar-large-128k-chat",
        temperature=0,
    )
    dspy.configure(lm=student_lm)
    console.print(
        "[green]✓[/green] Student model: PerplexityLM (llama-3.1-sonar-large-128k-chat)"
    )

    # Configure teacher model (Gemini) for GEPA feedback
    teacher_lm = dspy.LM("gemini/gemini-2.5-pro", temperature=0)
    console.print("[green]✓[/green] Teacher model: Gemini 2.5 Pro")

    # ===== PHASE 1: OPTIMIZATION =====
    logger.info("=" * 80)
    logger.info("STARTING PHASE 1: GEPA OPTIMIZATION")
    logger.info("=" * 80)
    phase_var.set("optimization")
    metrics["optimization"]["start_time"] = time.time()

    # Configure GEPA
    console.print(
        "\n[yellow]Starting GEPA optimization (this may take 1-3 hours)...[/yellow]"
    )

    optimizer = GEPA(
        metric=gepa_compatible_metric,
        auto="medium",  # Use 'medium' preset for production run
        reflection_lm=teacher_lm,
        seed=SEED,
    )

    # Run optimization
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Optimizing with GEPA...", total=None)

        try:
            optimized_detector = optimizer.compile(
                detector,
                trainset=train_data,
                valset=val_data,
            )
            progress.update(task, completed=True)
            console.print("[green]✓[/green] Optimization completed successfully")

        except Exception as e:
            logger.exception("Optimization failed")
            console.print(f"[red]✗[/red] Optimization failed: {e}", style="bold red")
            raise

    metrics["optimization"]["end_time"] = time.time()
    metrics["optimization"]["total_duration_seconds"] = (
        metrics["optimization"]["end_time"] - metrics["optimization"]["start_time"]
    )
    logger.info(
        f"Optimization completed in {metrics['optimization']['total_duration_seconds']/60:.1f} minutes"
    )

    # Save optimized model
    model_file = output_dir / "optimized_detector.json"
    optimized_detector.save(str(model_file))
    console.print(f"[green]✓[/green] Saved optimized model to {model_file}")

    # ===== PHASE 2: EVALUATION =====
    logger.info("=" * 80)
    logger.info("STARTING PHASE 2: TEST SET EVALUATION")
    logger.info(f"Evaluating {len(test_data)} test examples on optimized model")
    logger.info("=" * 80)

    phase_var.set("evaluation")
    metrics["evaluation"]["start_time"] = time.time()
    metrics["evaluation"]["total_examples"] = len(test_data)

    console.print("\n[yellow]Evaluating on test set...[/yellow]")
    test_scores = []
    test_predictions = []

    for i, example in enumerate(test_data):
        # Generate unique ID for this evaluation
        request_id = str(uuid.uuid4())[:8]
        request_id_var.set(request_id)

        logger.info(f"Processing test example {i+1}/{len(test_data)}")
        logger.debug(
            f"Example input: location_name='{example.location_name}', "
            f"location_area='{example.location_area}'"
        )
        logger.debug(
            f"Expected output: is_greenhouse={example.is_greenhouse}, "
            f"uses_growlight={example.uses_growlight}"
        )

        start_time = time.time()

        try:
            prediction = optimized_detector(
                location_name=example.location_name,
                location_area=example.location_area,
            )

            duration = time.time() - start_time

            logger.info(f"Prediction successful in {duration:.2f}s")
            logger.debug(
                f"Prediction output: is_greenhouse={prediction.is_greenhouse}, "
                f"uses_growlight={prediction.uses_growlight}, "
                f"confidence={prediction.confidence}"
            )

            score, feedback = gepa_compatible_metric(
                example, prediction, None, None, None
            )
            test_scores.append(score)
            test_predictions.append(
                {
                    "location_name": example.location_name,
                    "true_is_greenhouse": example.is_greenhouse,
                    "pred_is_greenhouse": prediction.is_greenhouse,
                    "true_uses_growlight": example.uses_growlight,
                    "pred_uses_growlight": prediction.uses_growlight,
                    "score": score,
                    "feedback": feedback,
                    "request_id": request_id,
                    "duration_seconds": duration,
                }
            )

            metrics["evaluation"]["successful_predictions"] += 1
            metrics["evaluation"]["total_duration_seconds"] += duration

        except Exception as e:
            duration = time.time() - start_time

            logger.error(f"TEST EXAMPLE {i+1} FAILED after {duration:.2f}s")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {e!s}")
            logger.exception("Full exception traceback:")

            # Create error dump for reproducibility
            error_dump = {
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                "phase": "evaluation",
                "example_index": i + 1,
                "total_examples": len(test_data),
                "example_data": {
                    "location_name": example.location_name,
                    "location_area": example.location_area,
                    "expected_is_greenhouse": example.is_greenhouse,
                    "expected_uses_growlight": example.uses_growlight,
                },
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                },
            }

            # Add API error details if available
            if hasattr(e, "__cause__") and hasattr(e.__cause__, "response"):
                error_dump["api_response"] = {
                    "status_code": e.__cause__.response.status_code,
                    "body": e.__cause__.response.text,
                }

            error_dump_file = output_dir / f"error_dump_{request_id}.json"
            with open(error_dump_file, "w") as f:
                json.dump(error_dump, f, indent=2)

            logger.error(f"Error dump saved: {error_dump_file}")
            console.print(
                f"[red]✗[/red] Test example {i+1} failed - see error_dump_{request_id}.json"
            )

            metrics["evaluation"]["failed_predictions"] += 1

            # Continue to next example to see all failures
            continue

    # End of evaluation
    metrics["evaluation"]["end_time"] = time.time()

    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info(
        f"Successful: {metrics['evaluation']['successful_predictions']}/{len(test_data)}"
    )
    logger.info(
        f"Failed: {metrics['evaluation']['failed_predictions']}/{len(test_data)}"
    )
    if len(test_scores) > 0:
        logger.info(
            f"Average score: {np.mean(test_scores):.3f} ± {np.std(test_scores):.3f}"
        )
    logger.info("=" * 80)

    # Calculate test metrics
    avg_test_score = np.mean(test_scores)
    std_test_score = np.std(test_scores)

    console.print(f"[green]✓[/green] Test set evaluation completed")
    console.print(
        f"[cyan]Average test score: {avg_test_score:.3f} ± {std_test_score:.3f}[/cyan]"
    )

    # Save test results
    test_results_file = output_dir / "test_results.json"
    test_results_file.write_text(
        json.dumps(
            {
                "test_scores": test_scores,
                "avg_score": float(avg_test_score),
                "std_score": float(std_test_score),
                "predictions": test_predictions,
            },
            indent=2,
        )
    )

    # Create results summary
    results = {
        "optimization_complete": True,
        "model_path": str(model_file),
        "test_results_path": str(test_results_file),
        "avg_test_score": float(avg_test_score),
        "std_test_score": float(std_test_score),
        "num_train": len(train_data),
        "num_val": len(val_data),
        "num_test": len(test_data),
    }

    return results


def display_results(results: dict[str, Any]) -> None:
    """
    Display optimization results in formatted table.

    Args:
        results: Dictionary with optimization results

    """
    table = Table(
        title="Phase 3 GEPA Optimization Results",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Training Examples", f"{results['num_train']:,}")
    table.add_row("Validation Examples", f"{results['num_val']:,}")
    table.add_row("Test Examples", f"{results['num_test']:,}")
    table.add_row("", "")  # Separator
    table.add_row("Test Score (avg)", f"{results['avg_test_score']:.3f}")
    table.add_row("Test Score (std)", f"{results['std_test_score']:.3f}")
    table.add_row("", "")  # Separator
    table.add_row("Optimized Model", str(Path(results["model_path"]).name))
    table.add_row("Test Results", str(Path(results["test_results_path"]).name))

    console.print("\n")
    console.print(table)
    console.print("\n[green]✓ Phase 3 optimization completed successfully![/green]")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments

    """
    parser = argparse.ArgumentParser(
        description="Run full Phase 3 GEPA optimization with PerplexityLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  uv run python scripts/run_phase3_gepa_optimization.py \\
    --cache-dir data/research \\
    --output-dir results/phase3_optimization

  # Custom train/test split
  uv run python scripts/run_phase3_gepa_optimization.py \\
    --cache-dir data/research \\
    --output-dir results/phase3_optimization \\
    --test-size 0.3 \\
    --val-size 0.25
        """,
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory containing cached research JSON files",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save optimization results",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for test set (default: 0.2)",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Proportion of training data for validation (default: 0.2)",
    )

    return parser.parse_args()


def main() -> int:
    """
    Run Phase 3 GEPA optimization.

    Returns:
        Exit code (0 for success, 1 for failure)

    """
    # Display header
    console.print(
        Panel.fit(
            "[bold cyan]Phase 3 GEPA Optimization with PerplexityLM[/bold cyan]\n"
            "Full optimization run using GEPA with auto='medium'\n"
            "[dim]Student: PerplexityLM | Teacher: Gemini 2.5 Pro[/dim]",
            border_style="cyan",
        )
    )

    args = parse_arguments()

    # Verify cache directory exists
    if not args.cache_dir.exists():
        console.print(
            f"[red]✗[/red] Cache directory not found: {args.cache_dir}",
            style="bold red",
        )
        return 1

    # Run optimization
    try:
        results = run_optimization(
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            test_size=args.test_size,
            val_size=args.val_size,
        )
    except Exception as e:
        console.print(f"[red]✗[/red] Optimization failed: {e}", style="bold red")
        logger.exception("Optimization failed")
        return 1

    # Display results
    display_results(results)

    # Save summary
    summary_file = args.output_dir / "optimization_summary.json"
    summary_file.write_text(json.dumps(results, indent=2))
    console.print(f"\n[green]✓[/green] Summary saved to {summary_file}")

    # Print metrics summary
    logger.info("\n" + "=" * 80)
    logger.info("METRICS SUMMARY")
    logger.info("=" * 80)

    opt_duration = (metrics["optimization"]["end_time"] or time.time()) - metrics[
        "optimization"
    ]["start_time"]
    eval_duration = (metrics["evaluation"]["end_time"] or time.time()) - metrics[
        "evaluation"
    ]["start_time"]

    logger.info(f"\nOptimization Phase:")
    logger.info(f"  Duration: {opt_duration/60:.1f} minutes")

    logger.info(f"\nEvaluation Phase:")
    logger.info(f"  Total Examples: {metrics['evaluation']['total_examples']}")
    logger.info(f"  Successful: {metrics['evaluation']['successful_predictions']}")
    logger.info(f"  Failed: {metrics['evaluation']['failed_predictions']}")
    logger.info(f"  Duration: {eval_duration/60:.1f} minutes")
    logger.info("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
