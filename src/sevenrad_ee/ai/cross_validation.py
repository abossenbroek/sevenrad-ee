"""
Robust cross-validation framework for small DSPy datasets.

This module provides repeated nested cross-validation functionality to address
overfitting issues and provide reliable performance estimates on small datasets.
It uses stratified K-fold splitting to maintain class balance and tracks both
training and validation performance to detect overfitting.

Phase 5: Repeated nested cross-validation for 95-98% F1 target.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from sklearn.model_selection import RepeatedStratifiedKFold

try:
    import dspy
except ImportError as e:
    msg = "dspy-ai package is required. Install with: uv pip install -e '.[dev]'"
    raise ImportError(msg) from e

logger = logging.getLogger(__name__)
console = Console()

# Phase 5: Overfitting and stability thresholds
SEVERE_OVERFITTING_THRESHOLD = 0.15  # >15% gap = severe overfitting
MODERATE_OVERFITTING_THRESHOLD = 0.10  # 10-15% gap = moderate overfitting
HIGH_VARIANCE_THRESHOLD = 0.05  # >5% std = high variance


def repeated_nested_cv(  # noqa: PLR0913, PLR0915, PLR0912, C901
    program_class: type,
    optimizer_func: Callable[[list[dspy.Example], list[dspy.Example]], Any],
    all_examples: list[dspy.Example],
    metric_func: Callable[..., float | tuple[float, str]],
    n_repeats: int = 10,
    n_folds: int = 5,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Perform repeated nested cross-validation for robust F1 estimation.

    This addresses the severe overfitting problem (89% train vs 62% val) by:
    1. Using stratified K-fold to maintain class balance across folds
    2. Repeating CV multiple times with different random seeds for stability
    3. Reporting mean ± std for reliability assessment
    4. Tracking train/val gap to detect overfitting (>15% = severe)

    The function performs nested cross-validation where each fold is optimized
    independently, preventing information leakage between train and validation sets.

    Args:
        program_class: DSPy program class to instantiate (e.g., GreenhouseDetector)
        optimizer_func: Function that takes (trainset, valset) and returns a
            configured optimizer instance
        all_examples: Full dataset of labeled examples (60-65 examples recommended)
        metric_func: Evaluation metric function. Can return either:
            - float: Simple score (for BootstrapFewShot/MIPROv2)
            - tuple[float, str]: (score, feedback) for GEPA optimizer
        n_repeats: Number of CV repetitions with different random seeds (default: 10)
        n_folds: Number of folds per repetition (default: 5)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
        - mean_f1: Mean validation F1 across all folds (float)
        - std_f1: Standard deviation of validation F1 (float)
        - train_f1: Mean training F1 across all folds (float)
        - overfitting_gap: train_f1 - mean_f1 (positive = overfitting)
        - fold_scores: List of all validation scores (list[float])
        - detailed_results: Per-fold breakdown with train/val scores (list[dict])

    Raises:
        ValueError: If all_examples is empty or has inconsistent label structure
        RuntimeError: If optimizer fails to compile program

    Example:
        >>> from sevenrad_ee.ai.dspy_greenhouse import GreenhouseDetector
        >>> from sevenrad_ee.ai.dspy_evaluation import dutch_aware_hierarchical_f1
        >>>
        >>> def optimizer_factory(trainset, valset):
        ...     from dspy.teleprompt import BootstrapFewShot
        ...     return BootstrapFewShot(
        ...         metric=dutch_aware_hierarchical_f1,
        ...         max_bootstrapped_demos=5
        ...     )
        >>>
        >>> results = repeated_nested_cv(
        ...     program_class=GreenhouseDetector,
        ...     optimizer_func=optimizer_factory,
        ...     all_examples=dataset,
        ...     metric_func=dutch_aware_hierarchical_f1,
        ...     n_repeats=10,
        ...     n_folds=5
        ... )
        >>> print(f"Mean F1: {results['mean_f1']:.2%} ± {results['std_f1']:.2%}")
        >>> print(f"Overfitting gap: {results['overfitting_gap']:+.2%}")

    """
    if not all_examples:
        msg = "all_examples cannot be empty"
        raise ValueError(msg)

    console.print("\n[bold cyan]Repeated Nested Cross-Validation[/bold cyan]")
    console.print(
        f"Repeats: {n_repeats} | Folds: {n_folds} | "
        f"Total runs: {n_repeats * n_folds}\n"
    )

    # Extract labels for stratification
    labels = []
    for ex in all_examples:
        if hasattr(ex, "uses_growlight"):
            labels.append(str(ex.uses_growlight))
        elif hasattr(ex, "is_greenhouse"):
            labels.append(str(ex.is_greenhouse))
        else:
            msg = "Examples must have either 'uses_growlight' or 'is_greenhouse' field"
            raise ValueError(msg)

    # Repeated stratified K-fold
    rkf = RepeatedStratifiedKFold(
        n_splits=n_folds, n_repeats=n_repeats, random_state=random_seed
    )

    val_scores: list[float] = []
    train_scores: list[float] = []
    detailed_results: list[dict[str, Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        total_iterations = n_repeats * n_folds
        task = progress.add_task("Running CV...", total=total_iterations)

        for fold_num, (train_idx, val_idx) in enumerate(
            rkf.split(all_examples, labels), start=1
        ):
            # Create train/val splits
            train_fold = [all_examples[i] for i in train_idx]
            val_fold = [all_examples[i] for i in val_idx]

            progress.update(
                task,
                description=f"Fold {fold_num}/{total_iterations}: Optimizing...",
                completed=fold_num - 1,
            )

            # Optimize on train fold
            try:
                optimizer = optimizer_func(train_fold, val_fold)
                base_program = program_class()

                optimized_program = optimizer.compile(base_program, trainset=train_fold)
            except Exception as e:
                logger.exception("Optimizer failed on fold %d", fold_num)
                msg = f"Optimizer compilation failed on fold {fold_num}: {e}"
                raise RuntimeError(msg) from e

            # Evaluate on train fold (to measure overfitting)
            train_predictions = [
                optimized_program(
                    location_name=ex.location_name, location_area=ex.location_area
                )
                for ex in train_fold
            ]
            train_fold_scores = []
            for ex, pred in zip(train_fold, train_predictions, strict=False):
                result = metric_func(ex, pred)
                score = result[0] if isinstance(result, tuple) else result
                train_fold_scores.append(float(score))

            train_fold_score = float(np.mean(train_fold_scores))
            train_scores.append(train_fold_score)

            # Evaluate on validation fold
            val_predictions = [
                optimized_program(
                    location_name=ex.location_name, location_area=ex.location_area
                )
                for ex in val_fold
            ]
            val_fold_scores = []
            for ex, pred in zip(val_fold, val_predictions, strict=False):
                result = metric_func(ex, pred)
                score = result[0] if isinstance(result, tuple) else result
                val_fold_scores.append(float(score))

            val_fold_score = float(np.mean(val_fold_scores))
            val_scores.append(val_fold_score)

            detailed_results.append(
                {
                    "fold": fold_num,
                    "train_score": train_fold_score,
                    "val_score": val_fold_score,
                    "train_size": len(train_fold),
                    "val_size": len(val_fold),
                }
            )

            progress.update(task, completed=fold_num)

    # Calculate aggregate statistics
    mean_val_f1 = float(np.mean(val_scores))
    std_val_f1 = float(np.std(val_scores))
    mean_train_f1 = float(np.mean(train_scores))
    overfitting_gap = mean_train_f1 - mean_val_f1

    # Display results
    console.print("\n[bold]Cross-Validation Results:[/bold]")
    console.print(f"  Mean Validation F1: {mean_val_f1:.2%} ± {std_val_f1:.2%}")
    console.print(f"  Mean Training F1: {mean_train_f1:.2%}")
    console.print(f"  Overfitting Gap: {overfitting_gap:+.2%}")

    # Overfitting assessment
    if overfitting_gap > SEVERE_OVERFITTING_THRESHOLD:
        console.print("  [red]⚠ SEVERE OVERFITTING (gap >15%)[/red]")
    elif overfitting_gap > MODERATE_OVERFITTING_THRESHOLD:
        console.print("  [yellow]⚠ Moderate overfitting (gap 10-15%)[/yellow]")
    else:
        console.print("  [green]✓ Good generalization (gap <10%)[/green]")

    # Stability assessment
    if std_val_f1 > HIGH_VARIANCE_THRESHOLD:
        console.print(
            "  [yellow]⚠ High variance (std >5%) - may need more data[/yellow]"
        )
    else:
        console.print("  [green]✓ Stable performance (std <5%)[/green]")

    return {
        "mean_f1": mean_val_f1,
        "std_f1": std_val_f1,
        "train_f1": mean_train_f1,
        "overfitting_gap": overfitting_gap,
        "fold_scores": val_scores,
        "detailed_results": detailed_results,
    }
