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

from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

logger = logging.getLogger(__name__)


# Constants for Phase 2 validation
HIGH_CONFIDENCE_THRESHOLD = 0.8  # Min confidence for tier-2 + Dutch terms


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


# Phase 2: Enhanced Pydantic models for hierarchical classification with Dutch guidance


class EvidenceSource(BaseModel):
    """
    Evidence source with URL, quote, and quality tier.

    This model represents a single piece of evidence from web research,
    classified by source quality tier for confidence scoring.
    """

    url: str = Field(..., description="Source URL")
    quote: str = Field(..., description="Relevant quote from source")
    tier: Literal[
        "company_website",
        "supplier_case_study",
        "job_posting",
        "trade_media_nl",
        "general_web",
    ] = Field(..., description="Source quality tier classification")


class GreenhouseDetectionOutput(BaseModel):
    """
    Validated greenhouse detection output with hierarchical logic enforcement.

    This model enforces Phase 2 requirements:
    - Hierarchical gating (not greenhouse → growlight must be UNKNOWN)
    - Dutch terminology tracking
    - Evidence quality tiers
    - Confidence scoring based on evidence quality
    """

    is_greenhouse: Literal["YES", "NO"] = Field(
        ..., description="Is this a greenhouse/kwekerij/nursery?"
    )
    uses_growlight: Literal["YES", "NO", "UNKNOWN"] = Field(
        ...,
        description=(
            "Does this greenhouse use artificial lighting (assimilatiebelichting)? "
            "MUST be UNKNOWN if is_greenhouse=NO"
        ),
    )
    species_grown: list[str] = Field(
        default_factory=list,
        description="List of species/crops grown (e.g., ['roses', 'tomatoes'])",
    )
    dutch_terms_found: list[str] = Field(
        default_factory=list,
        description=(
            "Dutch horticultural terms found in sources "
            "(e.g., 'assimilatiebelichting', 'kwekerij', 'belichte teelt')"
        ),
    )
    evidence_sources: list[EvidenceSource] = Field(
        default_factory=list,
        description="List of source URLs with quotes and tier classification",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Classification confidence 0.0-1.0. "
            "High confidence (>0.8) requires Dutch sources + tier-2 evidence"
        ),
    )
    rationale: str = Field(
        ...,
        description=(
            "Brief explanation of classification decision mentioning "
            "key evidence, source quality, and uncertainty factors"
        ),
    )

    @field_validator("uses_growlight")
    @classmethod
    def validate_hierarchical_logic(
        cls, v: Literal["YES", "NO", "UNKNOWN"], info: ValidationInfo
    ) -> Literal["YES", "NO", "UNKNOWN"]:
        """Enforce hierarchical gating: not greenhouse → growlight must be UNKNOWN."""
        if info.data.get("is_greenhouse") == "NO" and v != "UNKNOWN":
            msg = "uses_growlight must be UNKNOWN when is_greenhouse=NO"
            raise ValueError(msg)
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence_matches_evidence(
        cls, v: float, info: ValidationInfo
    ) -> float:
        """Ensure confidence aligns with evidence quality."""
        sources = info.data.get("evidence_sources", [])

        # Count tier-2 sources
        tier2_count = sum(
            1
            for s in sources
            if s.tier in ["company_website", "supplier_case_study", "job_posting"]
        )

        if v > HIGH_CONFIDENCE_THRESHOLD and tier2_count == 0:
            msg = (
                f"High confidence (>{HIGH_CONFIDENCE_THRESHOLD}) "
                "requires at least one tier-2 source"
            )
            raise ValueError(msg)

        return v


class GreenhouseClassificationValidator:
    """
    Validate hierarchical classification logic and evidence quality.

    This validator implements Phase 2 business rules:
    1. Hierarchical gating (not greenhouse → growlight = UNKNOWN)
    2. Evidence requirement (uses_growlight=YES requires evidence)
    3. Confidence-evidence alignment (high confidence requires tier-2 sources)
    4. Dutch terminology requirement (high confidence requires Dutch terms)
    """

    @staticmethod
    def validate(
        prediction: GreenhouseDetectionOutput,
    ) -> tuple[bool, str]:
        """
        Validate prediction follows all hierarchical rules.

        Args:
            prediction: Greenhouse detection output to validate

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if all rules pass, False otherwise
            - error_message: Empty string if valid, error description if invalid

        """
        # Rule 1: Hierarchical gating
        if prediction.is_greenhouse == "NO" and prediction.uses_growlight != "UNKNOWN":
            return (
                False,
                f"uses_growlight must be UNKNOWN when is_greenhouse=NO, "
                f"got {prediction.uses_growlight}",
            )

        # Rule 2: Evidence requirement
        if prediction.uses_growlight == "YES" and not prediction.evidence_sources:
            return (
                False,
                "uses_growlight=YES requires evidence_sources to be non-empty",
            )

        # Rule 3: Confidence-evidence alignment
        tier2_count = sum(
            1
            for s in prediction.evidence_sources
            if s.tier in ["company_website", "supplier_case_study", "job_posting"]
        )

        if prediction.confidence > HIGH_CONFIDENCE_THRESHOLD and tier2_count == 0:
            return (
                False,
                f"High confidence ({prediction.confidence:.2f}) requires "
                f"at least one tier-2 source, found {tier2_count}",
            )

        # Rule 4: Dutch terminology requirement
        if (
            prediction.confidence > HIGH_CONFIDENCE_THRESHOLD
            and len(prediction.dutch_terms_found) == 0
        ):
            return (
                False,
                f"High confidence ({prediction.confidence:.2f}) requires "
                "Dutch terminology evidence",
            )

        # All rules passed
        return True, ""


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


# Phase 2: Enhanced DSPy Signature with Dutch-first guidance


class GreenhouseClassification(Signature):  # type: ignore[misc]
    """
    Classify Dutch greenhouse companies using Perplexity Sonar web search.

    CRITICAL: This is a web-search-only classifier. Use Perplexity Sonar to find
    evidence from Dutch horticultural sources.

    SEARCH STRATEGY (Dutch-First):
    1. Start with Dutch company websites and trade media
    2. Look for Dutch terminology:
       - Greenhouse: "kwekerij", "glastuinbouw", "teler", "kassen"
       - Lighting: "assimilatiebelichting", "assimilatieverlichting", "kunstlicht",
                  "belichte teelt", "LED-belichting", "SON-T lampen"
       - Suppliers: "Signify", "Philips Hortilux", "Priva", "Hoogendoorn"
    3. Prioritize these source types:
       - Tier 2 (highest): Company websites, supplier case studies, job postings
       - Tier 1: Dutch trade media (Groenten&Fruit, Floraldaily NL, KAS Magazine)
       - Tier 0.5: General web sources
    4. Check for negative indicators:
       - "onbelichte teelt" (unlit cultivation)
       - "daglichtkas" (daylight greenhouse only)
       - Summer-only crops with no winter operation

    HIERARCHICAL LOGIC:
    - If is_greenhouse = NO → uses_growlight MUST be UNKNOWN
    - If is_greenhouse = YES → determine uses_growlight based on evidence
    - If evidence insufficient → uses_growlight = UNKNOWN (prefer precision)
    - Species provides validation (roses/tomatoes often use lighting)
    """

    # Input fields
    location_name: str = dspy.InputField(desc="Company name and location to classify")
    location_area: str = dspy.InputField(desc="Geographic area (city, region)")

    # Hierarchical outputs with gating
    is_greenhouse: str = dspy.OutputField(
        desc=(
            "Is this a greenhouse/kwekerij/nursery? Answer 'YES' or 'NO'. "
            "Use web search to verify."
        )
    )

    uses_growlight: str = dspy.OutputField(
        desc=(
            "Does this greenhouse use artificial lighting (assimilatiebelichting)? "
            "Answer 'YES', 'NO', or 'UNKNOWN'. "
            "ONLY answer YES if is_greenhouse=YES AND you have evidence. "
            "Answer UNKNOWN if insufficient evidence. "
            "Answer NO only with explicit negative evidence (e.g., 'onbelichte teelt')."
        )
    )

    species_grown: str = dspy.OutputField(
        desc=(
            "Comma-separated list of species/crops grown (e.g., 'roses,tomatoes'). "
            "Empty string if unknown. Species helps validate lighting usage: "
            "roses/orchids/gerbera often require lighting; lettuce/herbs sometimes; "
            "cucumbers/peppers less common."
        )
    )

    # Evidence capture (critical for validation)
    dutch_terms_found: str = dspy.OutputField(
        desc=(
            "Comma-separated Dutch horticultural terms found in sources. "
            "Examples: 'assimilatiebelichting,kwekerij,belichte teelt,"
            "SON-T,LED-belichting'. More Dutch terms = higher confidence."
        )
    )

    evidence_sources: str = dspy.OutputField(
        desc=(
            "JSON list of source dictionaries with format: "
            "[{'url': '...', 'quote': '...', 'tier': 'company_website'}, ...]. "
            "Tiers: company_website, supplier_case_study, job_posting (tier 2); "
            "trade_media_nl (tier 1); general_web (tier 0.5)."
        )
    )

    confidence: float = dspy.OutputField(
        desc=(
            "Classification confidence 0.0-1.0. "
            "High confidence (>0.8) requires: Dutch sources + "
            "Dutch terminology + tier 2 evidence. "
            "Medium (0.5-0.8): Some Dutch evidence or tier 1 sources. "
            "Low (<0.5): Only general web sources or ambiguous evidence."
        )
    )

    rationale: str = dspy.OutputField(
        desc=(
            "Brief explanation of classification decision. "
            "Mention: key evidence found, source quality, Dutch terminology, "
            "and any uncertainty factors."
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
