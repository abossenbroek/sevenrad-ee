"""
Evaluation metrics for DSPy greenhouse growlight detection.

This module provides F1 score and accuracy metrics for evaluating
the performance of the greenhouse detection system.

Phase 3: Dutch-aware hierarchical F1 metric with feedback generation
for GEPA optimizer reflection mechanism.
"""

import logging
from collections.abc import Callable
from typing import Any

try:
    import dspy
except ImportError as e:
    msg = "dspy-ai package is required. Install with: uv pip install -e '.[dev]'"
    raise ImportError(msg) from e

logger = logging.getLogger(__name__)

# Phase 3: Dutch terminology reference for scoring
DUTCH_TERMS = {
    "assimilatiebelichting",
    "assimilatieverlichting",
    "kunstlicht",
    "kassen",
    "kwekerij",
    "teler",
    "glastuinbouw",
    "belichte",
    "led-belichting",
    "son-t",
    "groeilicht",
    "teeltspecialist",
    "assimilatielampen",
    "onbelichte teelt",
    "daglichtkas",
    "hybride belichting",
}

# Phase 3: Source tier scoring weights
SOURCE_TIER_SCORES = {
    "company_website": 2.0,
    "supplier_case_study": 2.0,
    "job_posting": 2.0,
    "trade_media_nl": 1.0,
    "general_web": 0.5,
}

# Phase 3: Thresholds for feedback generation
MIN_DUTCH_TERMS_FOR_GOOD_FEEDBACK = 3
HIGH_CONFIDENCE_THRESHOLD_FOR_WARNING = 0.8


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

    # Extract predicted growlight usage (now a Literal["YES", "NO", "UNKNOWN", "NOT_APPLICABLE"])
    if hasattr(prediction, "uses_growlight"):
        pred_growlight_str = str(prediction.uses_growlight).strip().upper()
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


# Phase 3: Dutch-aware hierarchical F1 metric with feedback


def dutch_aware_hierarchical_f1(  # noqa: C901, PLR0912, PLR0915
    example: dspy.Example,
    prediction: Any,  # noqa: ANN401
    trace: Any | None = None,  # noqa: ANN401
) -> tuple[float, str]:
    """
    Hierarchical F1 metric with Dutch terminology weighting and textual feedback.

    This metric is designed for GEPA optimizer's reflection mechanism. It returns:
    1. A score (0.0-1.0) for optimization
    2. Textual feedback explaining successes/failures for reflection model

    Scoring Components:
    - 70%: Hierarchical classification accuracy (is_greenhouse + uses_growlight)
    - 15%: Dutch terminology detection (rewards finding Dutch terms)
    - 15%: Evidence quality (rewards tier-2 Dutch sources)

    Args:
        example: Ground truth example with is_greenhouse and uses_growlight fields
        prediction: Model prediction with classification and evidence fields
        trace: Optional execution trace (unused)

    Returns:
        Tuple of (score, feedback):
        - score: Float between 0.0 and 1.0
        - feedback: String with detailed explanation for reflection

    Example:
        >>> example = dspy.Example(
        ...     is_greenhouse="YES",
        ...     uses_growlight="YES",
        ...     location_name="Porta Nova"
        ... )
        >>> prediction = ...  # From DSPy model
        >>> score, feedback = dutch_aware_hierarchical_f1(example, prediction)
        >>> print(f"Score: {score:.2f}")
        >>> print(f"Feedback: {feedback}")

    """
    del trace  # Unused parameter

    feedback_parts = []

    # Component 1: Hierarchical Classification (70% weight)
    # --------------------------------------------------------

    # Extract predicted values
    if hasattr(prediction, "is_greenhouse"):
        pred_is_greenhouse = str(prediction.is_greenhouse).strip().upper()
    else:
        logger.warning("Prediction missing is_greenhouse field")
        return 0.0, "FATAL: Prediction missing is_greenhouse field"

    if hasattr(prediction, "uses_growlight"):
        pred_uses_growlight = str(prediction.uses_growlight).strip().upper()
    else:
        logger.warning("Prediction missing uses_growlight field")
        return 0.0, "FATAL: Prediction missing uses_growlight field"

    # Extract ground truth (normalize to uppercase strings)
    true_is_greenhouse = str(example.is_greenhouse).strip().upper()
    true_uses_growlight = str(example.uses_growlight).strip().upper()

    # Greenhouse classification
    gh_correct = pred_is_greenhouse == true_is_greenhouse
    gh_score = 1.0 if gh_correct else 0.0

    if not gh_correct:
        feedback_parts.append(
            f"MISCLASSIFIED is_greenhouse: predicted {pred_is_greenhouse}, "
            f"expected {true_is_greenhouse}"
        )

    # Growlight classification (only if greenhouse=YES in ground truth)
    if true_is_greenhouse == "YES":
        gl_correct = pred_uses_growlight == true_uses_growlight
        gl_score = 1.0 if gl_correct else 0.0

        if not gl_correct:
            feedback_parts.append(
                f"MISCLASSIFIED uses_growlight: predicted {pred_uses_growlight}, "
                f"expected {true_uses_growlight}"
            )

            # Specific Dutch term guidance
            if hasattr(example, "location_name"):
                company_lower = example.location_name.lower()
                pred_terms_lower = []
                if hasattr(prediction, "dutch_terms_found"):
                    if isinstance(prediction.dutch_terms_found, list):
                        pred_terms_lower = [
                            t.lower() for t in prediction.dutch_terms_found
                        ]
                    elif isinstance(prediction.dutch_terms_found, str):
                        pred_terms_lower = [
                            t.strip().lower()
                            for t in prediction.dutch_terms_found.split(",")
                            if t.strip()
                        ]

                if "assimilatie" in company_lower and not any(
                    "assimilatie" in t for t in pred_terms_lower
                ):
                    feedback_parts.append(
                        "MISSED 'assimilatie' terminology in company name - "
                        "this is a strong signal for growlight usage"
                    )

                # Check for common Dutch grower terms
                lighting_terms = ["kwekerij", "teler", "rozen", "orchidee"]
                for term in lighting_terms:
                    if term in company_lower:
                        feedback_parts.append(
                            f"Company name contains '{term}' - "
                            f"search for '{term} + assimilatiebelichting' might help"
                        )
    else:
        # If not a greenhouse, growlight should be UNKNOWN
        gl_score = 1.0 if pred_uses_growlight == "UNKNOWN" else 0.0

        if gl_score == 0.0:
            feedback_parts.append(
                f"LOGIC ERROR: is_greenhouse=NO but "
                f"uses_growlight={pred_uses_growlight} (should be UNKNOWN)"
            )

    classification_score = (gh_score + gl_score) / 2.0

    # Component 2: Dutch Terminology Detection (15% weight)
    # ------------------------------------------------------

    pred_dutch_terms = []
    if hasattr(prediction, "dutch_terms_found"):
        if isinstance(prediction.dutch_terms_found, list):
            pred_dutch_terms = prediction.dutch_terms_found
        elif isinstance(prediction.dutch_terms_found, str):
            # Handle comma-separated string format
            pred_dutch_terms = [
                t.strip() for t in prediction.dutch_terms_found.split(",") if t.strip()
            ]

    found_terms = {t.lower() for t in pred_dutch_terms}
    matching_terms = found_terms & DUTCH_TERMS

    dutch_score = len(matching_terms) / max(len(DUTCH_TERMS), 1)

    if len(matching_terms) == 0:
        feedback_parts.append(
            "NO Dutch terminology found - search strategy may be ineffective. "
            "Try searches like: '{company} assimilatiebelichting', "
            "'{company} kwekerij belichte teelt'"
        )
    elif len(matching_terms) >= MIN_DUTCH_TERMS_FOR_GOOD_FEEDBACK:
        feedback_parts.append(
            f"GOOD: Found {len(matching_terms)} Dutch terms: "
            f"{', '.join(list(matching_terms)[:5])}"
        )
    else:
        feedback_parts.append(
            f"Found {len(matching_terms)} Dutch terms - "
            "could find more with better queries"
        )

    # Component 3: Evidence Quality (15% weight)
    # -------------------------------------------

    pred_evidence_sources = []
    if hasattr(prediction, "evidence_sources"):
        if isinstance(prediction.evidence_sources, list):
            pred_evidence_sources = prediction.evidence_sources
        elif isinstance(prediction.evidence_sources, str):
            # Try parsing JSON string
            try:
                import json

                pred_evidence_sources = json.loads(prediction.evidence_sources)
            except (json.JSONDecodeError, ValueError):
                # If parsing fails, treat as empty
                pred_evidence_sources = []

    if not pred_evidence_sources:
        evidence_score = 0.0
        feedback_parts.append(
            "NO evidence sources provided - "
            "predictions must be backed by URLs and quotes"
        )
    else:
        total_tier_score = 0.0
        for source in pred_evidence_sources:
            if isinstance(source, dict):
                tier = source.get("tier", "general_web")
            elif hasattr(source, "tier"):
                tier = source.tier
            else:
                tier = "general_web"

            total_tier_score += SOURCE_TIER_SCORES.get(tier, 0.5)

        max_possible_score = len(pred_evidence_sources) * 2.0
        evidence_score = (
            total_tier_score / max_possible_score if max_possible_score > 0 else 0.0
        )

        tier2_count = 0
        for source in pred_evidence_sources:
            if isinstance(source, dict):
                tier = source.get("tier", "general_web")
            elif hasattr(source, "tier"):
                tier = source.tier
            else:
                tier = "general_web"

            if tier in ["company_website", "supplier_case_study", "job_posting"]:
                tier2_count += 1

        if tier2_count == 0:
            feedback_parts.append(
                "LOW-QUALITY sources (no tier-2 evidence) - "
                "prioritize company websites, supplier case studies, or job postings"
            )
        else:
            feedback_parts.append(
                f"GOOD: {tier2_count} tier-2 sources found (high-quality evidence)"
            )

    # Weighted Total Score
    # --------------------

    total_score = (
        classification_score * 0.70 + dutch_score * 0.15 + evidence_score * 0.15
    )

    # Final Feedback Assembly
    # -----------------------

    if not feedback_parts:
        feedback = (
            f"✓ CORRECT classification | "
            f"Found {len(matching_terms)} Dutch terms | "
            f"{len(pred_evidence_sources)} sources"
        )
    else:
        feedback = " | ".join(feedback_parts)

    # Add confidence check
    pred_confidence = getattr(prediction, "confidence", None)
    if pred_confidence is not None:
        tier2_count = sum(
            1
            for source in pred_evidence_sources
            if (
                source.get("tier")
                if isinstance(source, dict)
                else getattr(source, "tier", None)
            )
            in ["company_website", "supplier_case_study", "job_posting"]
        )

        if pred_confidence > HIGH_CONFIDENCE_THRESHOLD_FOR_WARNING and tier2_count == 0:
            feedback += (
                " | WARNING: High confidence without tier-2 evidence is unreliable"
            )

    return total_score, feedback


def dutch_aware_f1_score_only(
    example: dspy.Example,
    prediction: Any,  # noqa: ANN401
) -> float:
    """
    Return F1 score without feedback (for BootstrapFewShot/MIPROv2 compatibility).

    This is a backward-compatible wrapper around dutch_aware_hierarchical_f1
    that returns only the score component, dropping the feedback text.

    Use this metric with optimizers that expect a single float return value
    (BootstrapFewShot, MIPROv2). Use gepa_compatible_metric for GEPA optimizer.

    Args:
        example: Ground truth example
        prediction: Model prediction

    Returns:
        Float score between 0.0 and 1.0

    Example:
        >>> from dspy.teleprompt import BootstrapFewShot
        >>> optimizer = BootstrapFewShot(
        ...     metric=dutch_aware_f1_score_only,
        ...     max_bootstrapped_demos=5
        ... )

    """
    score, _ = dutch_aware_hierarchical_f1(example, prediction)
    return score


def gepa_compatible_metric(
    gold: dspy.Example,
    pred: Any,  # noqa: ANN401
    trace: Any | None = None,  # noqa: ANN401
    pred_name: str | None = None,
    pred_trace: Any | None = None,  # noqa: ANN401
) -> tuple[float, str]:
    """
    GEPA-compatible wrapper for dutch_aware_hierarchical_f1 metric.

    GEPA requires metrics to accept 5 arguments: (gold, pred, trace, pred_name, pred_trace).
    This wrapper adapts our 3-argument metric to the GEPA interface.

    Args:
        gold: Ground truth example
        pred: Model prediction
        trace: Optional execution trace (unused)
        pred_name: Name of the predictor being evaluated (unused)
        pred_trace: Trace of the predictor execution (unused)

    Returns:
        Tuple of (score, feedback) for GEPA's reflection mechanism

    Example:
        >>> from dspy.teleprompt import GEPA
        >>> optimizer = GEPA(
        ...     metric=gepa_compatible_metric,
        ...     auto='medium',
        ...     reflection_lm=teacher_lm,
        ...     seed=42
        ... )

    """
    # Call our existing metric (ignores extra GEPA-specific arguments)
    score, feedback = dutch_aware_hierarchical_f1(gold, pred, trace)
    return score, feedback
