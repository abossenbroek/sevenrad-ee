"""
DSPy-based greenhouse detection with artificial lighting classification.

This module uses DSPy (Declarative Self-improving Python) to create
optimizable prompts for detecting greenhouse operations that use artificial
lighting. It prioritizes Dutch and German sources with domain-specific
terminology for enhanced accuracy, and uses WUR (Wageningen University)
research as an authoritative tie-breaker for uncertain classifications.
"""

import logging
from enum import Enum
from typing import Optional

try:
    import dspy
    from dspy import Signature
except ImportError as e:
    msg = (
        "dspy-ai package is required. Install with: "
        "uv pip install -e '.[dev]' or pip install dspy-ai~=2.5.0"
    )
    raise ImportError(msg) from e

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# Enums for structured outputs


class GrowlightUsage(str, Enum):
    """
    Explicit uncertainty modeling for growlight classification.

    This enum forces the model to explicitly indicate when evidence is
    inconclusive rather than providing a low-confidence boolean answer.
    """

    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"


# Pydantic models for structured output


class GreenhouseLightingAnalysis(BaseModel):
    """
    Hierarchical greenhouse classification with explicit uncertainty.

    This model enforces a logical classification hierarchy:
    1. is_greenhouse (primary gate)
    2. uses_growlight (only if is_greenhouse=True)
    3. species_grown (only if is_greenhouse=True)

    The model uses TriState (YES/NO/UNKNOWN) for growlight classification
    to explicitly model uncertainty when evidence is inconclusive.
    """

    is_greenhouse: bool = Field(
        ...,
        description=(
            "Is this location an actual greenhouse facility? "
            "False for auction houses, seed companies, transport, storage."
        ),
    )
    uses_growlight: Optional[GrowlightUsage] = Field(
        None,
        description=(
            "Does the greenhouse use artificial growlights? "
            "'UNKNOWN' if evidence is inconclusive. "
            "Null if is_greenhouse=False."
        ),
    )
    species_grown: Optional[list[str]] = Field(
        None,
        description=(
            "Species/crops grown (e.g., ['roses', 'gerberas', 'tomatoes']). "
            "Null if is_greenhouse=False."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence score (0.0-1.0) for the classification",
    )
    reasoning: str = Field(
        ...,
        description=(
            "Step-by-step reasoning: "
            "(1) Direct facility evidence, "
            "(2) WUR research if used, "
            "(3) Species validation, "
            "(4) Final decision. "
            "CITE ALL URLs."
        ),
    )
    lighting_type: Optional[str] = Field(
        None,
        description=(
            "Type of artificial lighting detected "
            "(e.g., 'LED', 'SON-T', 'HPS', 'Mixed'). "
            "Only if uses_growlight=YES."
        ),
    )
    sources: list[str] = Field(
        default_factory=list,
        description=(
            "URLs of sources used for analysis. "
            "Include WUR publications if consulted."
        ),
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is within valid range."""
        if not 0.0 <= v <= 1.0:
            msg = "Confidence must be between 0.0 and 1.0"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_hierarchy(self) -> "GreenhouseLightingAnalysis":
        """
        Validate hierarchical classification logic.

        If not a greenhouse, downstream fields must be None.
        If is a greenhouse, growlight status must be determined.
        """
        if not self.is_greenhouse:
            # Not a greenhouse → all downstream fields must be None
            if self.uses_growlight is not None or self.species_grown is not None:
                msg = (
                    "If is_greenhouse=False, uses_growlight and "
                    "species_grown must be None"
                )
                raise ValueError(msg)
        # Is a greenhouse → growlight status must be determined (even if UNKNOWN)
        elif self.uses_growlight is None:
            msg = (
                "If is_greenhouse=True, uses_growlight must be "
                "YES, NO, or UNKNOWN (not None)"
            )
            raise ValueError(msg)
        return self


# DSPy Signatures


class GreenhouseDetectionSignature(Signature):  # type: ignore[misc]
    """
    DSPy Signature for greenhouse detection with WUR academic tie-breaker.

    This signature implements a hierarchical classification workflow:
    1. Determine if location is an actual greenhouse
    2. Search for direct evidence of growlights
    3. If unclear, use WUR (Wageningen University) research as tie-breaker
    4. Cross-validate with species patterns

    The signature uses web search (Perplexity Sonar) to gather evidence
    and WUR academic research as an authoritative source for Dutch
    greenhouse agricultural practices.
    """

    # Input fields (minimal - let Perplexity search)
    location_name: str = dspy.InputField(desc="Company or facility name to research")
    location_area: str = dspy.InputField(
        desc="City or region (e.g., 'waddinxveen, Netherlands')"
    )

    # Output fields (hierarchical)
    is_greenhouse: bool = dspy.OutputField(
        desc=(
            "Is this an actual greenhouse facility? "
            "False for auction houses (bloemenveiling), seed companies, "
            "transport companies, caravan storage, etc."
        )
    )
    uses_growlight: str = dspy.OutputField(
        desc=(
            "'YES' if uses artificial growlights, "
            "'NO' if natural light only, "
            "'UNKNOWN' if evidence is inconclusive. "
            "Return empty string if is_greenhouse=False. "
            "\n\nCLASSIFICATION WORKFLOW:\n"
            "1. Search for DIRECT evidence at the facility\n"
            "2. If unclear, use TIE-BREAKER: Search WUR research\n"
            "   - 'assimilatieverlichting voor {species}' site:wur.nl\n"
            "   - 'supplemental lighting for {species}' site:wur.nl\n"
            "3. Cross-validate with species:\n"
            "   - Roses/Gerberas/Lilies/Chrysanthemums → almost always YES\n"
            "   - Tomatoes/Peppers → check WUR for Dutch climate\n"
            "4. If still unclear → UNKNOWN"
        )
    )
    species_grown: str = dspy.OutputField(
        desc=(
            "Comma-separated species/crops grown. "
            "Examples: 'roses', 'gerberas', 'tomatoes', 'peppers', "
            "'lilies', 'chrysanthemums', 'cucumbers'. "
            "Return empty string if is_greenhouse=False or unknown."
        )
    )
    confidence: float = dspy.OutputField(
        desc=(
            "Classification confidence (0.0-1.0). "
            "Direct facility evidence = highest confidence. "
            "WUR research support = medium-high confidence. "
            "No evidence = low confidence."
        )
    )
    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning with citations:\n"
            "1. Direct facility evidence (website, YouTube, news)\n"
            "2. WUR research if used (CITE wur.nl URLs)\n"
            "3. Species validation\n"
            "4. Final decision\n"
            "\nCITE ALL URLS. Include WUR publications if consulted."
        )
    )
    lighting_type: str = dspy.OutputField(
        desc=(
            "Type of lighting if uses_growlight=YES: "
            "'LED', 'SON-T', 'HPS', 'Mixed', or 'Unknown'. "
            "Return 'None' if uses_growlight=NO or UNKNOWN."
        )
    )
    sources: str = dspy.OutputField(
        desc=(
            "Source URLs separated by '|||'. "
            "\n\nSOURCE PRIORITY:\n"
            "1. Direct facility evidence (company website, YouTube)\n"
            "2. WUR academic research (wur.nl, edepot.wur.nl)\n"
            "3. Dutch agricultural sources (kasmagazine.nl, onderglas.nl)\n"
            "4. General sources\n"
            "\nInclude WUR URLs if used as tie-breaker."
        )
    )


# DSPy Module


class GreenhouseDetector(dspy.Module):  # type: ignore[misc]
    """
    DSPy Module for detecting greenhouse artificial lighting usage.

    This module uses Chain-of-Thought reasoning with Perplexity's search
    capabilities to determine if a greenhouse uses artificial lighting.
    It can be optimized using DSPy's teleprompters.

    Example:
        >>> import dspy
        >>> from dspy.adapters import Perplexity
        >>>
        >>> # Configure Perplexity as the LM
        >>> perplexity_lm = dspy.LM(
        ...     'perplexity/sonar',
        ...     api_key='your-api-key',
        ...     api_base='https://api.perplexity.ai'
        ... )
        >>> dspy.configure(lm=perplexity_lm)
        >>>
        >>> # Create detector
        >>> detector = GreenhouseDetector()
        >>>
        >>> # Analyze a company
        >>> result = detector(
        ...     company_name="Royal Van Zanten",
        ...     location="Rijsenhout, Nederland"
        ... )
        >>> print(result.uses_artificial_lighting)
        True

    """

    def __init__(self) -> None:
        """Initialize the greenhouse detector with CoT predictor."""
        super().__init__()
        # Use ChainOfThought for reasoning before output
        self.predictor = dspy.ChainOfThought(GreenhouseDetectionSignature)

    def forward(
        self,
        location_name: str,
        location_area: str,
    ) -> dspy.Prediction:
        """
        Analyze a location to detect greenhouse and artificial lighting usage.

        This method implements the hierarchical classification workflow with
        WUR academic tie-breaker strategy.

        Args:
            location_name: Name of the facility/company
            location_area: City or region (e.g., 'waddinxveen, Netherlands')

        Returns:
            dspy.Prediction with fields matching GreenhouseDetectionSignature

        """
        logger.info("Analyzing %s in %s", location_name, location_area)

        # Run prediction
        prediction = self.predictor(
            location_name=location_name,
            location_area=location_area,
        )

        logger.info(
            "Classification: is_greenhouse=%s, uses_growlight=%s, confidence=%.2f",
            prediction.is_greenhouse,
            prediction.uses_growlight,
            prediction.confidence,
        )

        return prediction

    def to_pydantic(self, prediction: dspy.Prediction) -> GreenhouseLightingAnalysis:
        """
        Convert DSPy Prediction to Pydantic model for validation.

        This method handles the hierarchical structure and converts
        string outputs to appropriate types (enums, lists, etc.).

        Args:
            prediction: DSPy Prediction from forward()

        Returns:
            GreenhouseLightingAnalysis validated Pydantic model

        Raises:
            ValueError: If prediction violates hierarchical logic

        """
        # Parse is_greenhouse
        is_greenhouse = bool(prediction.is_greenhouse)

        # Parse uses_growlight (TriState enum)
        uses_growlight_str = prediction.uses_growlight.strip().upper()
        if is_greenhouse:
            # Must be YES/NO/UNKNOWN
            if uses_growlight_str in ("YES", "NO", "UNKNOWN"):
                uses_growlight = GrowlightUsage(uses_growlight_str)
            else:
                # Default to UNKNOWN if unclear
                logger.warning(
                    "Invalid uses_growlight value '%s', defaulting to UNKNOWN",
                    prediction.uses_growlight,
                )
                uses_growlight = GrowlightUsage.UNKNOWN
        else:
            # Not a greenhouse → must be None
            uses_growlight = None

        # Parse species_grown
        if is_greenhouse and prediction.species_grown:
            species_list = [
                s.strip() for s in prediction.species_grown.split(",") if s.strip()
            ]
        else:
            species_list = None

        # Parse sources
        sources_list = (
            [s.strip() for s in prediction.sources.split("|||") if s.strip()]
            if prediction.sources
            else []
        )

        # Parse lighting_type
        lighting_type = (
            prediction.lighting_type
            if (
                is_greenhouse
                and uses_growlight == GrowlightUsage.YES
                and prediction.lighting_type
                and prediction.lighting_type.lower() != "none"
            )
            else None
        )

        # Build Pydantic model (will validate hierarchy)
        return GreenhouseLightingAnalysis(
            is_greenhouse=is_greenhouse,
            uses_growlight=uses_growlight,
            species_grown=species_list,
            confidence=float(prediction.confidence),
            reasoning=prediction.reasoning,
            lighting_type=lighting_type,
            sources=sources_list,
        )


# Helper function for convenience


def analyze_greenhouse(
    location_name: str,
    location_area: str,
    lm: Optional[dspy.LM] = None,
) -> GreenhouseLightingAnalysis:
    """
    Analyze a location for greenhouse and artificial lighting usage.

    Convenience function that creates a detector, runs prediction with
    WUR tie-breaker strategy, and returns a validated Pydantic model.

    Args:
        location_name: Name of the facility/company
        location_area: City or region (e.g., 'waddinxveen, Netherlands')
        lm: Optional DSPy LM instance. If None, uses dspy.settings.lm

    Returns:
        GreenhouseLightingAnalysis with hierarchical classification

    Raises:
        ValueError: If no LM is configured and none provided, or if
                    hierarchical validation fails

    Example:
        >>> import dspy
        >>> lm = dspy.LM('perplexity/sonar', api_key='...')
        >>> dspy.configure(lm=lm)
        >>>
        >>> result = analyze_greenhouse(
        ...     location_name="Marjoland",
        ...     location_area="waddinxveen, Netherlands"
        ... )
        >>> print(f"Is greenhouse: {result.is_greenhouse}")
        >>> print(f"Uses growlight: {result.uses_growlight}")
        >>> print(f"Species: {result.species_grown}")
        >>> print(f"Confidence: {result.confidence:.2%}")

    """
    if lm:
        with dspy.context(lm=lm):
            detector = GreenhouseDetector()
            prediction = detector(location_name, location_area)
            return detector.to_pydantic(prediction)

    # Use default LM from dspy.settings
    detector = GreenhouseDetector()
    prediction = detector(location_name, location_area)
    return detector.to_pydantic(prediction)
