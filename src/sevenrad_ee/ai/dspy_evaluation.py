"""
Evaluation metrics for DSPy greenhouse growlight detection.

This module provides F1 score and accuracy metrics for evaluating
the performance of the greenhouse detection system.
"""

import logging
from collections.abc import Callable
from typing import Any

try:
    import dspy
except ImportError as e:
    msg = "dspy-ai package is required. Install with: uv pip install -e '.[dev]'"
    raise ImportError(msg) from e

from sevenrad_ee.ai.dspy_greenhouse import GrowlightUsage

logger = logging.getLogger(__name__)


def greenhouse_f1_metric(
    example: dspy.Example,
    prediction: Any,  # noqa: ANN401
    trace: Any | None = None,  # noqa: ANN401
) -> float:
    """
    Calculate F1 score for greenhouse classification.

    This metric focuses on the primary classification: is_greenhouse.
    It returns 1.0 for correct classification, 0.0 for incorrect.

    Args:
        example: Ground truth example with is_greenhouse field
        prediction: Model prediction with is_greenhouse field
        trace: Optional trace information (unused)

    Returns:
        1.0 if classification is correct, 0.0 otherwise

    """
    del trace  # Unused parameter

    # Extract predicted value
    if hasattr(prediction, "is_greenhouse"):
        pred_is_greenhouse = bool(prediction.is_greenhouse)
    else:
        logger.warning("Prediction missing is_greenhouse field")
        return 0.0

    # Extract ground truth
    true_is_greenhouse = bool(example.is_greenhouse)

    # Simple binary accuracy for now
    return 1.0 if pred_is_greenhouse == true_is_greenhouse else 0.0


def growlight_accuracy_metric(
    example: dspy.Example,
    prediction: Any,  # noqa: ANN401
    trace: Any | None = None,  # noqa: ANN401
) -> float:
    """
    Calculate accuracy for growlight classification.

    This metric evaluates the secondary classification: uses_growlight.
    Only applicable when both example and prediction agree it's a greenhouse.

    Args:
        example: Ground truth example
        prediction: Model prediction
        trace: Optional trace information (unused)

    Returns:
        1.0 if growlight classification is correct, 0.0 otherwise.
        Returns 1.0 if not a greenhouse (N/A).

    """
    del trace  # Unused parameter

    # First check if it's a greenhouse
    if hasattr(prediction, "is_greenhouse"):
        pred_is_greenhouse = bool(prediction.is_greenhouse)
    else:
        return 0.0

    true_is_greenhouse = bool(example.is_greenhouse)

    # If not a greenhouse, growlight classification doesn't apply
    if not (true_is_greenhouse and pred_is_greenhouse):
        return 1.0

    # Extract predicted growlight usage
    if hasattr(prediction, "uses_growlight"):
        pred_growlight = prediction.uses_growlight
        # Normalize to string
        if isinstance(pred_growlight, GrowlightUsage):
            pred_growlight_str = pred_growlight.value
        else:
            pred_growlight_str = str(pred_growlight).strip().upper()
    else:
        logger.warning("Prediction missing uses_growlight field")
        return 0.0

    # Extract ground truth
    true_growlight = str(example.uses_growlight).strip().upper()

    # Compare
    return 1.0 if pred_growlight_str == true_growlight else 0.0


def combined_f1_metric(
    example: dspy.Example,
    prediction: Any,  # noqa: ANN401
    trace: Any | None = None,  # noqa: ANN401
) -> float:
    """
    Calculate combined metric for greenhouse and growlight classification.

    This is the primary metric for optimization. It gives full credit (1.0)
    only when both classifications are correct.

    Scoring:
    - is_greenhouse correct + uses_growlight correct = 1.0
    - is_greenhouse correct + uses_growlight wrong = 0.5
    - is_greenhouse wrong = 0.0

    Args:
        example: Ground truth example
        prediction: Model prediction
        trace: Optional trace information (unused)

    Returns:
        Combined F1 score (0.0 to 1.0)

    """
    greenhouse_score = greenhouse_f1_metric(example, prediction, trace)

    # If greenhouse classification is wrong, entire prediction fails
    if greenhouse_score == 0.0:
        return 0.0

    # Greenhouse is correct, now check growlight
    true_is_greenhouse = bool(example.is_greenhouse)

    # If it's not a greenhouse, only greenhouse classification matters
    if not true_is_greenhouse:
        return greenhouse_score

    # It's a greenhouse - check growlight accuracy
    growlight_score = growlight_accuracy_metric(example, prediction, trace)

    # Weighted combination: 50% greenhouse, 50% growlight
    return (greenhouse_score + growlight_score) / 2.0


def evaluate_with_dspy(
    predictor: Any,  # noqa: ANN401
    devset: list[dspy.Example],
    metric: Callable[..., float] = combined_f1_metric,
    num_threads: int = 1,
) -> dict[str, float]:
    """
    Evaluate a DSPy predictor on a development set.

    Args:
        predictor: DSPy Module or Predictor to evaluate
        devset: List of dspy.Example objects for evaluation
        metric: Evaluation metric function (default: combined_f1_metric)
        num_threads: Number of parallel evaluation threads


    Returns:
        Dictionary with evaluation results:
        - 'score': Average metric score
        - 'scores': List of individual scores
        - 'n_examples': Number of examples evaluated

    """
    from dspy.evaluate import Evaluate

    evaluator = Evaluate(
        devset=devset,
        metric=metric,
        num_threads=num_threads,
        display_progress=True,
        display_table=5,
    )

    results = evaluator(predictor)

    return {
        "score": results,
        "n_examples": len(devset),
    }


def calculate_classification_metrics(  # noqa: C901
    predictions: list[Any],
    ground_truth: list[dspy.Example],
) -> dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Computes precision, recall, and F1 for both greenhouse and growlight
    classifications.

    Args:
        predictions: List of model predictions
        ground_truth: List of ground truth examples

    Returns:
        Dictionary with metrics:
        - greenhouse_precision
        - greenhouse_recall
        - greenhouse_f1
        - growlight_precision (for greenhouse=True cases)
        - growlight_recall (for greenhouse=True cases)
        - growlight_f1 (for greenhouse=True cases)

    """
    if len(predictions) != len(ground_truth):
        msg = "Predictions and ground truth must have same length"
        raise ValueError(msg)

    # Greenhouse classification
    gh_tp = gh_fp = gh_fn = 0
    for pred, truth in zip(predictions, ground_truth, strict=False):
        pred_gh = bool(getattr(pred, "is_greenhouse", False))
        true_gh = bool(truth.is_greenhouse)

        if pred_gh and true_gh:
            gh_tp += 1
        elif pred_gh and not true_gh:
            gh_fp += 1
        elif not pred_gh and true_gh:
            gh_fn += 1

    # Growlight classification (only for true greenhouses)
    gl_tp = gl_fp = gl_fn = 0
    for pred, truth in zip(predictions, ground_truth, strict=False):
        if not truth.is_greenhouse:
            continue  # Skip non-greenhouses

        pred_gl = str(getattr(pred, "uses_growlight", "")).strip().upper()
        true_gl = str(truth.uses_growlight).strip().upper()

        if pred_gl == true_gl and pred_gl == "YES":
            gl_tp += 1
        elif pred_gl == "YES" and true_gl != "YES":
            gl_fp += 1
        elif pred_gl != "YES" and true_gl == "YES":
            gl_fn += 1

    # Calculate metrics
    gh_precision = gh_tp / (gh_tp + gh_fp) if (gh_tp + gh_fp) > 0 else 0.0
    gh_recall = gh_tp / (gh_tp + gh_fn) if (gh_tp + gh_fn) > 0 else 0.0
    gh_f1 = (
        2 * (gh_precision * gh_recall) / (gh_precision + gh_recall)
        if (gh_precision + gh_recall) > 0
        else 0.0
    )

    gl_precision = gl_tp / (gl_tp + gl_fp) if (gl_tp + gl_fp) > 0 else 0.0
    gl_recall = gl_tp / (gl_tp + gl_fn) if (gl_tp + gl_fn) > 0 else 0.0
    gl_f1 = (
        2 * (gl_precision * gl_recall) / (gl_precision + gl_recall)
        if (gl_precision + gl_recall) > 0
        else 0.0
    )

    return {
        "greenhouse_precision": gh_precision,
        "greenhouse_recall": gh_recall,
        "greenhouse_f1": gh_f1,
        "growlight_precision": gl_precision,
        "growlight_recall": gl_recall,
        "growlight_f1": gl_f1,
    }
