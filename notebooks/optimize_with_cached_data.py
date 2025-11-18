"""
GEPA Optimization with Cached Perplexity Data - Phase 3.

This script runs genetic-Pareto optimization on frozen Perplexity evidence
to learn generalizable evidence synthesis patterns for greenhouse classification.

Expected duration: 8-10 hours
Expected cost: $15-25 (Gemini 2.5 Pro teacher model)
Target: F1 ≥75% on validation set

Documentation Type: Script (Code to Run)
Part of: Phase 3 - GEPA Optimization Strategy
"""

import argparse
import json
import logging
import random
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

from sklearn.model_selection import train_test_split

from sevenrad_ee.ai.dspy_evaluation import (
    calculate_classification_metrics,
    gepa_compatible_metric,
)
from sevenrad_ee.ai.dspy_greenhouse import GreenhouseDetector
from sevenrad_ee.ai.retrievers import CachedRetriever

console = Console()
logger = logging.getLogger(__name__)

# Random seed for reproducibility
SEED = 42


def set_random_seeds(seed: int = SEED) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed value

    """
    random.seed(seed)
    np.random.seed(seed)
    # If using PyTorch:
    # import torch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def load_greenhouse_data(cache_dir: Path) -> list[dspy.Example]:
    """
    Load cached greenhouse data from JSON files.

    Args:
        cache_dir: Directory containing research JSON files

    Returns:
        List of dspy.Example objects with company data

    Raises:
        ValueError: If no data found or loading fails

    """
    examples = []
    json_files = list(cache_dir.glob("*.json"))

    if not json_files:
        msg = f"No JSON files found in {cache_dir}"
        raise ValueError(msg)

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

    if not examples:
        msg = f"Failed to load any valid examples from {cache_dir}"
        raise ValueError(msg)

    console.print(f"[green]✓[/green] Loaded {len(examples)} examples")
    return examples


def display_dataset_statistics(
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
) -> None:
    """
    Display dataset statistics in formatted table.

    Args:
        trainset: Training examples
        valset: Validation examples

    """
    # Count class distribution
    train_greenhouse_count = sum(1 for ex in trainset if ex.is_greenhouse)
    val_greenhouse_count = sum(1 for ex in valset if ex.is_greenhouse)

    train_growlight_count = sum(
        1
        for ex in trainset
        if ex.is_greenhouse and str(ex.uses_growlight).upper() == "YES"
    )
    val_growlight_count = sum(
        1
        for ex in valset
        if ex.is_greenhouse and str(ex.uses_growlight).upper() == "YES"
    )

    # Create statistics table
    table = Table(title="Dataset Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Split", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Greenhouse", justify="right")
    table.add_column("Uses Growlight", justify="right")

    table.add_row(
        "Training",
        str(len(trainset)),
        f"{train_greenhouse_count} ({train_greenhouse_count/len(trainset):.1%})",
        f"{train_growlight_count} ({train_growlight_count/train_greenhouse_count:.1%})"
        if train_greenhouse_count > 0
        else "0 (0%)",
    )

    table.add_row(
        "Validation",
        str(len(valset)),
        f"{val_greenhouse_count} ({val_greenhouse_count/len(valset):.1%})",
        f"{val_growlight_count} ({val_growlight_count/val_greenhouse_count:.1%})"
        if val_greenhouse_count > 0
        else "0 (0%)",
    )

    console.print(table)


def run_optimization(
    cache_dir: Path,
    output_dir: Path,
    student_model: str,
    teacher_model: str,
    train_split: float,
    baseline_f1: float | None,
) -> dict[str, Any]:
    """
    Run GEPA optimization on cached dataset.

    Args:
        cache_dir: Directory with cached research data
        output_dir: Directory to save results
        student_model: Student model name (e.g., 'perplexity/sonar-reasoning-pro')
        teacher_model: Teacher model name (e.g., 'gemini/gemini-2.5-pro')
        train_split: Fraction for training set (0.0-1.0)
        baseline_f1: Baseline F1 from Phase 2 (optional)

    Returns:
        Dictionary with optimization results

    Raises:
        ValueError: If data loading or optimization fails

    """
    # Set random seeds for reproducibility
    set_random_seeds(SEED)

    # Load data
    console.print("\n[bold cyan]Step 1: Loading Data[/bold cyan]")
    all_data = load_greenhouse_data(cache_dir)

    # Split data (stratified by is_greenhouse)
    console.print("\n[bold cyan]Step 2: Splitting Data[/bold cyan]")
    trainset, valset = train_test_split(
        all_data,
        test_size=1.0 - train_split,
        random_state=SEED,
        stratify=[ex.is_greenhouse for ex in all_data],
    )

    display_dataset_statistics(trainset, valset)

    # Create retriever and detector
    console.print("\n[bold cyan]Step 3: Initializing Models[/bold cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating cached retriever...", total=None)
        cached_retriever = CachedRetriever(cache_dir=cache_dir)
        progress.update(task, completed=True)

        task = progress.add_task("Initializing detector...", total=None)
        detector = GreenhouseDetector(retriever=cached_retriever)
        progress.update(task, completed=True)

        task = progress.add_task(f"Configuring student model ({student_model})...", total=None)
        student_lm = dspy.LM(student_model)
        progress.update(task, completed=True)

        task = progress.add_task(
            f"Configuring teacher model ({teacher_model})...", total=None
        )
        teacher_lm = dspy.LM(teacher_model, temperature=0)  # Deterministic
        progress.update(task, completed=True)

    console.print("[green]✓[/green] Models initialized")

    # Configure GEPA optimizer
    console.print("\n[bold cyan]Step 4: Configuring GEPA Optimizer[/bold cyan]")

    optimizer_config = {
        "metric": gepa_compatible_metric,
        "auto": "medium",  # Budget-controlled preset ($15-25 target)
        "reflection_lm": teacher_lm,  # Teacher model for feedback-based optimization
        "seed": SEED,  # Deterministic
    }

    # Display configuration
    config_table = Table(show_header=False)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="white")

    config_table.add_row("Metric", "gepa_compatible_metric")
    config_table.add_row("Auto Preset", optimizer_config["auto"])
    config_table.add_row("Random Seed", str(optimizer_config["seed"]))
    config_table.add_row("Student Model", student_model)
    config_table.add_row("Reflection LM (Teacher)", teacher_model)

    console.print(config_table)

    optimizer = GEPA(**optimizer_config)

    # Run optimization
    console.print("\n[bold cyan]Step 5: Running GEPA Optimization[/bold cyan]")
    console.print("[yellow]Expected duration: 8-10 hours[/yellow]")
    console.print("[yellow]Expected cost: $15-25[/yellow]")
    console.print("[yellow]Target: F1 ≥75% on validation set[/yellow]\n")

    start_time = datetime.now()

    try:
        # Configure DSPy to use student model
        dspy.configure(lm=student_lm)

        optimized_detector = optimizer.compile(
            detector,
            trainset=trainset,
            valset=valset,
        )

        end_time = datetime.now()
        duration = end_time - start_time

        console.print(
            f"\n[green]✓[/green] Optimization completed in "
            f"{duration.total_seconds() / 3600:.1f} hours"
        )

    except Exception as e:
        logger.exception("Optimization failed")
        console.print(f"[red]✗[/red] Optimization failed: {e}", style="bold red")
        raise

    # Evaluate on validation set
    console.print("\n[bold cyan]Step 6: Evaluating Optimized Model[/bold cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running predictions on validation set...", total=None)

        predictions = []
        for example in valset:
            pred = optimized_detector(
                location_name=example.location_name,
                location_area=example.location_area,
            )
            predictions.append(pred)

        progress.update(task, completed=True)

    # Calculate metrics
    metrics = calculate_classification_metrics(predictions, valset)

    # Display results
    results_table = Table(
        title="Optimization Results", show_header=True, header_style="bold cyan"
    )
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", justify="right")

    if baseline_f1 is not None:
        results_table.add_row("Baseline F1", f"{baseline_f1:.1%}")

    results_table.add_row("Greenhouse F1", f"{metrics['greenhouse_f1']:.1%}")
    results_table.add_row("Growlight F1", f"{metrics['growlight_f1']:.1%}")
    results_table.add_row("Greenhouse Precision", f"{metrics['greenhouse_precision']:.1%}")
    results_table.add_row("Greenhouse Recall", f"{metrics['greenhouse_recall']:.1%}")

    if baseline_f1 is not None:
        improvement = metrics["greenhouse_f1"] - baseline_f1
        improvement_color = "green" if improvement > 0 else "red"
        results_table.add_row(
            "Improvement",
            f"[{improvement_color}]{improvement:+.1%}[/{improvement_color}]",
        )

    console.print(results_table)

    # Check if target met
    target_met = metrics["greenhouse_f1"] >= 0.75

    if target_met:
        console.print("\n[bold green]✓ Target F1 ≥75% achieved![/bold green]")
        console.print("[green]Proceed to Phase 4: Generalization Spot Check[/green]")
    else:
        console.print(
            f"\n[yellow]⚠️  Validation F1 is {metrics['greenhouse_f1']:.1%}, "
            f"below 75% target[/yellow]"
        )
        console.print(
            "[yellow]Review results and consider:[/yellow]\n"
            "1. Expanding training data (65 → 100+ companies)\n"
            "2. Manual prompt refinement based on failure modes\n"
            "3. Trying different optimization parameters"
        )

    # Save optimized program
    console.print("\n[bold cyan]Step 7: Saving Results[/bold cyan]")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    program_path = output_dir / f"optimized_program_{timestamp}.json"

    optimized_detector.save(str(program_path))
    console.print(f"[green]✓[/green] Optimized program saved to {program_path}")

    # Compile results
    results = {
        "timestamp": timestamp,
        "duration_hours": duration.total_seconds() / 3600,
        "train_size": len(trainset),
        "val_size": len(valset),
        "baseline_f1": baseline_f1,
        "optimized_metrics": metrics,
        "target_met": target_met,
        "config": {
            "student_model": student_model,
            "teacher_model": teacher_model,
            "train_split": train_split,
            "auto_preset": optimizer_config["auto"],
            "seed": optimizer_config["seed"],
        },
        "program_path": str(program_path),
    }

    return results


def generate_markdown_report(
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Generate markdown optimization report.

    Args:
        results: Optimization results dictionary
        output_path: Path to save report

    """
    metrics = results["optimized_metrics"]
    config = results["config"]

    report = f"""# Phase 3: GEPA Optimization Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Status**: {'✓ SUCCESS' if results['target_met'] else '⚠️ BELOW TARGET'}

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Student Model | {config['student_model']} |
| Teacher Model | {config['teacher_model']} |
| Train/Val Split | {config['train_split']:.0%} / {1-config['train_split']:.0%} |
| Training Size | {results['train_size']} examples |
| Validation Size | {results['val_size']} examples |
| Auto Preset | {config['auto_preset']} |
| Random Seed | {config['seed']} |

---

## Results

### Performance Metrics

| Metric | Value |
|--------|-------|
"""

    if results["baseline_f1"] is not None:
        report += f"| Baseline F1 (Phase 2) | {results['baseline_f1']:.1%} |\n"

    report += f"""| **Greenhouse F1** | **{metrics['greenhouse_f1']:.1%}** |
| Greenhouse Precision | {metrics['greenhouse_precision']:.1%} |
| Greenhouse Recall | {metrics['greenhouse_recall']:.1%} |
| Growlight F1 | {metrics['growlight_f1']:.1%} |
| Growlight Precision | {metrics['growlight_precision']:.1%} |
| Growlight Recall | {metrics['growlight_recall']:.1%} |
"""

    if results["baseline_f1"] is not None:
        improvement = metrics["greenhouse_f1"] - results["baseline_f1"]
        report += f"| **Improvement** | **{improvement:+.1%}** |\n"

    report += f"""
### Optimization Details

- **Duration**: {results['duration_hours']:.1f} hours
- **Optimized Program**: `{results['program_path']}`

---

## Next Steps

"""

    if results["target_met"]:
        report += """✓ **Target F1 ≥75% achieved!**

**Proceed to Phase 4: Generalization Spot Check**

1. Select 10-15 new companies (not in training set)
2. Run live Perplexity search with optimized detector
3. Calculate generalization gap
4. If gap <10%, deploy to pilot
5. If gap >15%, refine or iterate

See `OPTIMIZATION_STRATEGY.md` Phase 4 for details.
"""
    else:
        report += f"""⚠️ **Validation F1 is {metrics['greenhouse_f1']:.1%}, below 75% target**

**Recommended Actions**:

1. **Qualitative Analysis**: Review failure cases on validation set
   - What patterns were misclassified?
   - Are there specific crops or company types that fail?

2. **Data Expansion**: Collect 20-30 more companies
   - Focus on underrepresented patterns
   - Re-run GEPA optimization with larger dataset

3. **Manual Refinement**: Based on failure modes
   - Adjust prompts for edge cases
   - Add explicit handling for common errors

4. **Alternative Optimizers**: Try BootstrapFewShot or MIPROv2
   - May work better with small dataset

Before proceeding to Phase 4, consider if the current F1 ({metrics['greenhouse_f1']:.1%}) is acceptable for a canary test.
"""

    report += """
---

## Reproducibility

To reproduce this optimization:

```bash
# Ensure random seed is set to 42
uv run python notebooks/optimize_with_cached_data.py \\
  --cache-dir data/research \\
  --output-dir results/cached_optimization \\
  --student-model {student_model} \\
  --teacher-model {teacher_model} \\
  --train-split {train_split}
```

All random seeds were set to 42 for deterministic results.
""".format(
        student_model=config["student_model"],
        teacher_model=config["teacher_model"],
        train_split=config["train_split"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    console.print(f"[green]✓[/green] Report saved to {output_path}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments

    """
    parser = argparse.ArgumentParser(
        description="Run GEPA optimization on cached greenhouse data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  uv run python notebooks/optimize_with_cached_data.py \\
    --cache-dir data/research \\
    --output-dir results/cached_optimization

  # Run with custom models and baseline
  uv run python notebooks/optimize_with_cached_data.py \\
    --cache-dir data/research \\
    --output-dir results/cached_optimization \\
    --student-model perplexity/sonar-reasoning-pro \\
    --teacher-model gemini/gemini-2.5-pro \\
    --baseline-f1 0.634

  # Run in tmux for long optimization
  tmux new -s gepa-optimization
  uv run python notebooks/optimize_with_cached_data.py \\
    --cache-dir data/research \\
    --output-dir results/cached_optimization \\
    > results/cached_optimization/run.log 2>&1
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
        "--student-model",
        type=str,
        default="perplexity/sonar-reasoning-pro",
        help="Student model for classification (default: perplexity/sonar-reasoning-pro)",
    )

    parser.add_argument(
        "--teacher-model",
        type=str,
        default="gemini/gemini-2.5-pro",
        help="Teacher model for labeling (default: gemini/gemini-2.5-pro)",
    )

    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Fraction of data for training (default: 0.7 for 70/30 split)",
    )

    parser.add_argument(
        "--baseline-f1",
        type=float,
        help="Baseline F1 score from Phase 2 (for comparison)",
    )

    return parser.parse_args()


def main() -> int:
    """
    Run GEPA optimization.

    Returns:
        Exit code (0 for success, 1 for failure)

    """
    # Display header
    console.print(
        Panel.fit(
            "[bold cyan]Phase 3: GEPA Optimization on Cached Data[/bold cyan]\n"
            "Learn generalizable evidence synthesis patterns\n\n"
            "[yellow]Expected duration: 8-10 hours[/yellow]\n"
            "[yellow]Expected cost: $15-25[/yellow]\n"
            "[yellow]Target: F1 ≥75%[/yellow]",
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
        console.print(
            "\n[yellow]Tip:[/yellow] Ensure Phase 1 architecture is complete "
            "and data is cached in specified directory"
        )
        return 1

    # Validate train split
    if not 0.5 <= args.train_split <= 0.9:
        console.print(
            f"[red]✗[/red] Invalid train split: {args.train_split}. "
            "Must be between 0.5 and 0.9",
            style="bold red",
        )
        return 1

    # Run optimization
    try:
        results = run_optimization(
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            student_model=args.student_model,
            teacher_model=args.teacher_model,
            train_split=args.train_split,
            baseline_f1=args.baseline_f1,
        )
    except Exception as e:
        console.print(f"[red]✗[/red] Optimization failed: {e}", style="bold red")
        logger.exception("Optimization failed")
        return 1

    # Generate report
    timestamp = results["timestamp"]
    report_path = args.output_dir / f"optimization_report_{timestamp}.md"
    generate_markdown_report(results, report_path)

    # Final status
    if results["target_met"]:
        console.print("\n[bold green]✓ Phase 3 Complete - Target Achieved![/bold green]")
        return 0
    else:
        console.print(
            "\n[bold yellow]⚠️  Phase 3 Complete - Below Target[/bold yellow]"
        )
        console.print("[yellow]Review report and consider next steps[/yellow]")
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    raise SystemExit(main())
