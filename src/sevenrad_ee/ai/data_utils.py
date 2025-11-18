"""
Data loading utilities for the greenhouse classification project.

Provides standardized data structures and loading functions for evaluation
and optimization tasks. Used in Phase 2 (model selection) and Phase 3
(GEPA optimization) of the DSPy optimization strategy.

Key Components:
    - Company: Dataclass representing a company with ground truth label
    - load_cached_companies: Load all companies from cached research JSON files

Example:
    >>> from pathlib import Path
    >>> companies = load_cached_companies(Path("data/research"))
    >>> print(f"Loaded {len(companies)} companies")
    >>> print(f"First company: {companies[0]}")
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Company:
    """
    Represents a company with its ground truth classification.

    Attributes:
        name: Company name
        location: Geographic location
        manual_classification: True if company uses grow lights, False otherwise

    """

    name: str
    location: str
    manual_classification: bool


def load_cached_companies(cache_dir: Path) -> list[dict[str, str | bool]]:
    """
    Load all company data from cached research JSON files.

    Reads all *.json files from the specified directory and extracts company
    name, location, and classification. Converts the classification_suggestion
    field ("POSITIVE (uses growlights)" / "NEGATIVE (no growlights)") to a
    boolean manual_classification field for use in evaluation.

    Args:
        cache_dir: Directory containing the *.json research files

    Returns:
        List of company dictionaries with keys:
            - 'company': Company name (str)
            - 'location': Geographic location (str)
            - 'manual_classification': Ground truth label (bool)
              True if classification_suggestion starts with "POSITIVE"

    Raises:
        FileNotFoundError: If no JSON files are found in the cache directory
        ValueError: If JSON files are missing required fields

    Example:
        >>> from pathlib import Path
        >>> companies = load_cached_companies(Path("data/research"))
        >>> assert len(companies) == 65
        >>> assert all(k in companies[0] for k in ['company', 'location', 'manual_classification'])
        >>> # True for companies with "POSITIVE (uses growlights)"
        >>> # False for companies with "NEGATIVE (no growlights)"

    """
    if not cache_dir.exists():
        msg = f"Cache directory does not exist: {cache_dir}"
        raise FileNotFoundError(msg)

    json_files = list(cache_dir.glob("*.json"))
    if not json_files:
        msg = f"No JSON files found in {cache_dir}. Ensure cached data exists."
        raise FileNotFoundError(msg)

    companies = []
    errors = []

    for json_file in sorted(json_files):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))

            # Validate required fields
            required_fields = ["company", "location", "classification_suggestion"]
            missing_fields = [field for field in required_fields if field not in data]

            if missing_fields:
                msg = f"File {json_file.name} is missing required fields: {missing_fields}"
                errors.append(msg)
                logger.warning(msg)
                continue

            # Parse classification_suggestion to boolean
            # "POSITIVE (uses growlights)" -> True
            # "NEGATIVE (no growlights)" -> False
            classification_suggestion = data["classification_suggestion"]
            manual_classification = classification_suggestion.startswith("POSITIVE")

            # Convert to dict format expected by evaluate_with_cv.py
            companies.append(
                {
                    "company": data["company"],
                    "location": data["location"],
                    "manual_classification": manual_classification,
                }
            )

        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in file {json_file.name}: {e}"
            errors.append(msg)
            logger.warning(msg)
            continue
        except Exception as e:
            msg = f"Error processing file {json_file.name}: {e}"
            errors.append(msg)
            logger.warning(msg)
            continue

    if not companies:
        msg = (
            f"No valid companies loaded from {cache_dir}. "
            f"Errors encountered: {len(errors)}"
        )
        raise ValueError(msg)

    if errors:
        logger.info(
            f"Loaded {len(companies)} companies with {len(errors)} errors/warnings"
        )

    return companies


def load_cached_companies_as_dataclass(cache_dir: Path) -> list[Company]:
    """
    Load all company data as Company dataclass instances.

    Alternative loading function that returns Company dataclass instances
    instead of dictionaries. Useful when type safety is preferred.

    Args:
        cache_dir: Directory containing the *.json research files

    Returns:
        List of Company dataclass instances

    Raises:
        FileNotFoundError: If no JSON files are found in the cache directory
        ValueError: If JSON files are missing required fields

    Example:
        >>> from pathlib import Path
        >>> companies = load_cached_companies_as_dataclass(Path("data/research"))
        >>> print(f"First company: {companies[0].name} in {companies[0].location}")

    """
    company_dicts = load_cached_companies(cache_dir)

    return [
        Company(
            name=str(c["company"]),
            location=str(c["location"]),
            manual_classification=bool(c["manual_classification"]),
        )
        for c in company_dicts
    ]
