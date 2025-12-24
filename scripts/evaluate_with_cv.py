"""
Evaluation harness for stratified k-fold cross-validation with bootstrap.

Provides rigorous statistical testing for model comparison and optimization evaluation.
Used in Phase 2 (model selection) and Phase 3 (GEPA optimization) to ensure
statistically sound performance measurement with confidence intervals.

Key Features:
    - Stratified k-fold CV: Preserves class balance in train/test splits
    - Bootstrap resampling: Provides confidence intervals for F1 scores
    - Paired evaluation: Enables direct model comparison with statistical tests

Example:
    >>> from pathlib import Path
    >>> import dspy
    >>> from sevenrad_ee.ai.retrievers import CachedRetriever
    >>> from sevenrad_ee.ai.dspy_greenhouse import GreenhouseDetector
    >>>
    >>> # Configure DSPy
    >>> dspy.configure(lm=dspy.LM('perplexity/sonar'))
    >>>
    >>> # Create detector with cached retriever
    >>> cache_dir = Path("data/research")
    >>> retriever = CachedRetriever(cache_dir=cache_dir)
    >>> detector = GreenhouseDetector(retriever=retriever)
    >>>
    >>> # Load dataset
    >>> dataset = load_cached_companies(cache_dir)
    >>>
    >>> # Evaluate with CV
    >>> report = evaluate_with_stratified_cv(detector, dataset, n_folds=5)
    >>> print(f"Mean F1: {report.mean_f1:.1%}")
    >>> print(f"95% CI: [{report.confidence_interval_95[0]:.1%}, "
    ...       f"{report.confidence_interval_95[1]:.1%}]")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


@dataclass
class EvaluationReport:
    """
    Results from stratified k-fold CV evaluation with bootstrap.

    Attributes:
        mean_f1: Average F1 score across all folds
        std_f1: Standard deviation of F1 scores across folds
        fold_scores: List of F1 scores for each fold
        confidence_interval_95: Tuple of (lower, upper) bounds for 95% CI
        bootstrap_scores: Full distribution of bootstrapped F1 scores
        n_repetitions: Number of CV repetitions (1 for single-pass, >1 for repeated)
        repetition_scores: Mean F1 per repetition (for repeated CV analysis)

    """

    mean_f1: float
    std_f1: float
    fold_scores: list[float]
    confidence_interval_95: tuple[float, float]
    bootstrap_scores: list[float]
    n_repetitions: int = 1
    repetition_scores: list[float] | None = None


def bootstrap_f1_scores(
    predictions: list[bool],
    labels: list[bool],
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> list[float]:
    """
    Bootstrap F1 scores for confidence interval estimation.

    Uses resampling with replacement to generate a distribution of F1 scores,
    enabling robust confidence interval estimation even with small datasets.

    IMPORTANT: Bootstrap is used for confidence intervals, NOT for reducing
    variance or compensating for web search variability. The variance comes
    from the finite sample size, not from stochastic processes.

    Args:
        predictions: Model predictions (boolean or 0/1)
        labels: Ground truth labels (boolean or 0/1)
        n_bootstrap: Number of bootstrap samples (default: 1000)
        random_state: Random seed for reproducibility

    Returns:
        List of bootstrapped F1 scores for CI calculation

    Example:
        >>> predictions = [True, False, True, True, False]
        >>> labels = [True, False, True, False, False]
        >>> boot_scores = bootstrap_f1_scores(predictions, labels, n_bootstrap=100)
        >>> ci_95 = (np.percentile(boot_scores, 2.5), np.percentile(boot_scores, 97.5))
        >>> print(f"95% CI: [{ci_95[0]:.1%}, {ci_95[1]:.1%}]")

    """
    rng = np.random.default_rng(random_state)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(len(predictions), size=len(predictions), replace=True)
        boot_pred = [predictions[i] for i in indices]
        boot_label = [labels[i] for i in indices]

        # Calculate F1 for this bootstrap sample
        boot_f1 = f1_score(boot_label, boot_pred, zero_division=0.0)
        bootstrap_scores.append(boot_f1)

    return bootstrap_scores


def evaluate_with_stratified_cv(
    detector: Any,
    dataset: list[dict[str, Any]],
    n_folds: int = 5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> EvaluationReport:
    """
    Evaluate detector using stratified k-fold CV with bootstrap.

    Provides rigorous performance measurement with:
    - Stratified splits: Preserves class balance (critical for n=65)
    - Bootstrap CI: Quantifies uncertainty in F1 estimates
    - Fold-level tracking: Enables paired statistical tests

    Statistical Notes:
    - With n=65 and k=5, each test set has ~13 examples
    - Stratification ensures both classes represented in each fold
    - Bootstrap provides confidence intervals despite small n
    - For model comparison, use paired tests (e.g., Wilcoxon) on fold_scores

    Args:
        detector: GreenhouseDetector instance (with CachedRetriever)
        dataset: List of company dictionaries with:
                 - 'company': company name
                 - 'location': location string
                 - 'manual_classification': bool (ground truth)
        n_folds: Number of CV folds (default: 5)
        n_bootstrap: Number of bootstrap samples (default: 1000)
        random_state: Random seed for reproducibility

    Returns:
        EvaluationReport with mean F1, std, fold scores, and 95% CI

    Raises:
        ValueError: If dataset is empty or missing required fields

    Example:
        >>> from pathlib import Path
        >>> import dspy
        >>> from sevenrad_ee.ai.retrievers import CachedRetriever
        >>> from sevenrad_ee.ai.dspy_greenhouse import GreenhouseDetector
        >>>
        >>> # Setup
        >>> dspy.configure(lm=dspy.LM('perplexity/sonar'))
        >>> retriever = CachedRetriever(cache_dir=Path("data/research"))
        >>> detector = GreenhouseDetector(retriever=retriever)
        >>>
        >>> # Load dataset
        >>> dataset = [
        ...     {"company": "Company A", "location": "Amsterdam",
        ...      "manual_classification": True},
        ...     {"company": "Company B", "location": "Rotterdam",
        ...      "manual_classification": False},
        ...     # ... more companies
        ... ]
        >>>
        >>> # Evaluate
        >>> report = evaluate_with_stratified_cv(detector, dataset)
        >>> print(f"F1: {report.mean_f1:.1%} ± {report.std_f1:.1%}")

    """
    if not dataset:
        msg = "Dataset cannot be empty"
        raise ValueError(msg)

    # Validate dataset structure
    required_fields = ["company", "location", "manual_classification"]
    for i, company in enumerate(dataset):
        for field in required_fields:
            if field not in company:
                msg = f"Company {i} missing required field: {field}"
                raise ValueError(msg)

    # Extract labels for stratification
    labels = [company["manual_classification"] for company in dataset]

    # Create stratified k-fold splitter
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_scores = []
    all_bootstrap_scores = []

    # Evaluate each fold
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(dataset, labels)):
        test_companies = [dataset[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        # Run predictions on test set
        predictions = []
        for company in test_companies:
            try:
                pred = detector(
                    location_name=company["company"],
                    location_area=company["location"],
                )

                # Extract boolean prediction (handle different output formats)
                if hasattr(pred, "uses_growlight"):
                    # DSPy Prediction object
                    is_positive = pred.uses_growlight == "YES"
                elif isinstance(pred, dict):
                    # Dictionary format
                    is_positive = pred.get("uses_growlight") == "YES"
                else:
                    # Boolean directly
                    is_positive = bool(pred)

                predictions.append(is_positive)

            except Exception as e:
                # Log error but continue with pessimistic prediction
                print(f"Warning: Prediction failed for {company['company']}: {e}")
                predictions.append(False)

        # Calculate fold F1 score
        fold_f1 = f1_score(test_labels, predictions, zero_division=0.0)
        fold_scores.append(fold_f1)

        print(f"Fold {fold_idx + 1}/{n_folds}: F1 = {fold_f1:.1%}")

        # Bootstrap within fold for CI estimation
        boot_scores = bootstrap_f1_scores(
            predictions, test_labels, n_bootstrap, random_state + fold_idx
        )
        all_bootstrap_scores.extend(boot_scores)

    # Aggregate results
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)

    # 95% confidence interval from bootstrap distribution
    ci_95 = (
        np.percentile(all_bootstrap_scores, 2.5),
        np.percentile(all_bootstrap_scores, 97.5),
    )

    return EvaluationReport(
        mean_f1=mean_f1,
        std_f1=std_f1,
        fold_scores=fold_scores,
        confidence_interval_95=ci_95,
        bootstrap_scores=all_bootstrap_scores,
        n_repetitions=1,
        repetition_scores=None,
    )


def evaluate_with_repeated_stratified_cv(
    detector: Any,
    dataset: list[dict[str, Any]],
    n_folds: int = 10,
    n_repetitions: int = 3,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> EvaluationReport:
    """
    Evaluate detector using repeated stratified k-fold CV with bootstrap.

    This function implements the Phase 2.5 variance reduction strategy by using:
    - More folds (10 instead of 5) for larger, more stable test sets
    - Multiple repetitions (3-5) with different shuffles for robust estimation
    - Bootstrap confidence intervals for uncertainty quantification

    Statistical Improvements over single-pass CV:
    - Reduced sampling bias: Averages across multiple random shuffles
    - More stable variance estimates: More folds × repetitions = more data points
    - Better generalization assessment: Tests model on multiple train/test splits

    Target: Reduce std dev from 21% to <10% for stable Phase 3 baseline.

    Args:
        detector: GreenhouseDetector instance (with CachedRetriever)
        dataset: List of company dictionaries with:
                 - 'company': company name
                 - 'location': location string
                 - 'manual_classification': bool (ground truth)
        n_folds: Number of CV folds per repetition (default: 10)
        n_repetitions: Number of repetitions with different shuffles (default: 3)
        n_bootstrap: Number of bootstrap samples per fold (default: 1000)
        random_state: Random seed for reproducibility

    Returns:
        EvaluationReport with:
        - mean_f1: Grand mean across all folds and repetitions
        - std_f1: Standard deviation across all folds and repetitions
        - fold_scores: F1 scores for each fold (n_folds × n_repetitions total)
        - repetition_scores: Mean F1 per repetition (for analysis)
        - confidence_interval_95: 95% CI from bootstrap distribution
        - bootstrap_scores: Full bootstrap distribution

    Raises:
        ValueError: If dataset is empty or missing required fields

    Example:
        >>> from pathlib import Path
        >>> import dspy
        >>> from sevenrad_ee.ai.retrievers import CachedRetriever
        >>> from sevenrad_ee.ai.dspy_greenhouse import GreenhouseDetector
        >>>
        >>> # Setup
        >>> dspy.configure(lm=dspy.LM('perplexity/sonar'))
        >>> retriever = CachedRetriever(cache_dir=Path("data/research"))
        >>> detector = GreenhouseDetector(retriever=retriever)
        >>>
        >>> # Load dataset
        >>> dataset = [...]  # 65 companies with ground truth
        >>>
        >>> # Evaluate with repeated CV
        >>> report = evaluate_with_repeated_stratified_cv(
        ...     detector, dataset, n_folds=10, n_repetitions=3
        ... )
        >>> print(f"F1: {report.mean_f1:.1%} ± {report.std_f1:.1%}")
        >>> print(f"Repetitions: {report.repetition_scores}")

    """
    if not dataset:
        msg = "Dataset cannot be empty"
        raise ValueError(msg)

    # Validate dataset structure
    required_fields = ["company", "location", "manual_classification"]
    for i, company in enumerate(dataset):
        for field in required_fields:
            if field not in company:
                msg = f"Company {i} missing required field: {field}"
                raise ValueError(msg)

    # Extract labels for stratification
    labels = [company["manual_classification"] for company in dataset]

    # Create repeated stratified k-fold splitter
    rskf = RepeatedStratifiedKFold(
        n_splits=n_folds,
        n_repeats=n_repetitions,
        random_state=random_state,
    )

    fold_scores = []
    all_bootstrap_scores = []
    repetition_scores = []

    # Track which repetition we're in
    current_repetition = 0
    repetition_fold_scores = []

    # Evaluate each fold across all repetitions
    total_folds = n_folds * n_repetitions
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(dataset, labels)):
        # Determine current repetition (folds cycle through repetitions)
        fold_in_rep = fold_idx % n_folds
        if fold_in_rep == 0 and fold_idx > 0:
            # Finished a repetition - save mean for this repetition
            repetition_scores.append(np.mean(repetition_fold_scores))
            current_repetition += 1
            repetition_fold_scores = []

        test_companies = [dataset[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        # Run predictions on test set
        predictions = []
        for company in test_companies:
            try:
                pred = detector(
                    location_name=company["company"],
                    location_area=company["location"],
                )

                # Extract boolean prediction (handle different output formats)
                if hasattr(pred, "uses_growlight"):
                    # DSPy Prediction object
                    is_positive = pred.uses_growlight == "YES"
                elif isinstance(pred, dict):
                    # Dictionary format
                    is_positive = pred.get("uses_growlight") == "YES"
                else:
                    # Boolean directly
                    is_positive = bool(pred)

                predictions.append(is_positive)

            except Exception as e:
                # Log error but continue with pessimistic prediction
                print(f"Warning: Prediction failed for {company['company']}: {e}")
                predictions.append(False)

        # Calculate fold F1 score
        fold_f1 = f1_score(test_labels, predictions, zero_division=0.0)
        fold_scores.append(fold_f1)
        repetition_fold_scores.append(fold_f1)

        print(
            f"Rep {current_repetition + 1}/{n_repetitions}, "
            f"Fold {fold_in_rep + 1}/{n_folds}: F1 = {fold_f1:.1%}"
        )

        # Bootstrap within fold for CI estimation
        boot_scores = bootstrap_f1_scores(
            predictions, test_labels, n_bootstrap, random_state + fold_idx
        )
        all_bootstrap_scores.extend(boot_scores)

    # Save final repetition mean
    if repetition_fold_scores:
        repetition_scores.append(np.mean(repetition_fold_scores))

    # Aggregate results across all folds and repetitions
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores, ddof=1)  # Use sample std dev

    # 95% confidence interval from bootstrap distribution
    ci_95 = (
        np.percentile(all_bootstrap_scores, 2.5),
        np.percentile(all_bootstrap_scores, 97.5),
    )

    print(f"\n[Repeated CV Summary]")
    print(f"Grand Mean F1: {mean_f1:.1%} ± {std_f1:.1%}")
    print(f"Repetition Means: {[f'{s:.1%}' for s in repetition_scores]}")
    print(f"Total Folds Evaluated: {total_folds}")

    return EvaluationReport(
        mean_f1=mean_f1,
        std_f1=std_f1,
        fold_scores=fold_scores,
        confidence_interval_95=ci_95,
        bootstrap_scores=all_bootstrap_scores,
        n_repetitions=n_repetitions,
        repetition_scores=repetition_scores,
    )
