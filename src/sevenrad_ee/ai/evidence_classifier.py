"""
Evidence classification for greenhouse grow light detection.

This module provides DSPy signatures for semantically classifying evidence
about horticultural grow light usage in Dutch greenhouse companies.
"""

from enum import Enum

import dspy


class EvidenceCategory(str, Enum):
    """Evidence categories for grow light usage classification."""

    POSITIVE = "POSITIVE"  # Company USES grow lights
    NEGATIVE = "NEGATIVE"  # Company explicitly DOES NOT use grow lights
    NEUTRAL = "NEUTRAL"  # Topic mentioned, no company-specific evidence
    AMBIGUOUS = "AMBIGUOUS"  # Unclear or contradictory signals


class GrowLightEvidenceClassifier(dspy.Signature):  # type: ignore[misc]
    """
    Analyzes text to determine evidence of horticultural grow light usage.

    This classifier must understand:
    - Dutch negation: "geen", "zonder", "niet", "kiest tegen"
    - Implicit signals: "overweegt" (considering) = not yet using
    - Context: "gebruikt geen SON-T maar LED" = still POSITIVE
    - Crop context: pot plants often don't need night lighting
    - Evidence types:
        * Direct: "installeert LED-belichting"
        * Indirect: supplier partnerships, research participation
        * Negative: "kiest voor onbelichte teelt"
        * Absence: "no evidence found" ≠ "does not use"

    POSITIVE Examples:
    - "Bedrijf X investeert in Signify LED-belichting"
    - "Tomaten worden geteeld met assimilatiebelichting"
    - "Deelname aan onderzoek naar energiezuinige groeilampen"

    NEGATIVE Examples:
    - "kiest bewust voor onbelichte teelt zonder kunstlicht"
    - "teelt potplanten uitsluitend met daglicht"
    - "geen assimilatiebelichting vanwege energiekosten"

    NEUTRAL Examples:
    - "In de sector wordt LED-belichting steeds gangbaarder" (sector trend)
    - "There is no evidence linking Company X to grow lights"

    AMBIGUOUS Examples:
    - "overweegt jaarrondteelt met belichting" (considering but not using yet)
    - Contradictory statements in same text
    """

    company_name: str = dspy.InputField(desc="Name of the company being researched")

    evidence_text: str = dspy.InputField(
        desc="Search result text content to classify for grow light evidence"
    )

    evidence_category: str = dspy.OutputField(
        desc="Must be exactly one of: POSITIVE, NEGATIVE, NEUTRAL, or AMBIGUOUS"
    )

    confidence: float = dspy.OutputField(
        desc=(
            "Confidence 0.0-1.0: 0.9+ for explicit statements, "
            "0.5-0.8 for indirect evidence, <0.5 for weak signals"
        )
    )

    reasoning: str = dspy.OutputField(
        desc=(
            "Concise explanation: what evidence led to this classification? "
            "Quote key phrases from the text."
        )
    )
