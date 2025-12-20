"""
DSPy-based greenhouse detection with artificial lighting classification.

This module uses DSPy (Declarative Self-improving Python) to create
optimizable prompts for detecting greenhouse operations that use artificial
lighting. It prioritizes Dutch and German sources with domain-specific
terminology for enhanced accuracy, and uses WUR (Wageningen University)
research as an authoritative tie-breaker for uncertain classifications.
"""

import contextvars
import logging
from typing import TYPE_CHECKING, Literal

try:
    import dspy
    from dspy import Signature
except ImportError as e:
    msg = (
        "dspy-ai package is required. Install with: "
        "uv pip install -e '.[dev]' or pip install dspy-ai~=2.5.0"
    )
    raise ImportError(msg) from e

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

if TYPE_CHECKING:
    from sevenrad_ee.ai.retrievers import Retriever

logger = logging.getLogger(__name__)

# Context variables for request tracking (defined in logging_setup)
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="N/A"
)
phase_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "phase", default="setup"
)

# Constants for Phase 2 validation
HIGH_CONFIDENCE_THRESHOLD = 0.8  # Min confidence for tier-2 + Dutch terms


# Pydantic models for structured output
# Phase 2.5: Simplified model to avoid DSPy schema rendering issues


class SimplifiedGreenhouseAnalysis(BaseModel):
    """
    Simplified greenhouse classification optimized for DSPy + Perplexity.

    This model uses flat, primitive types to avoid DSPy's "messy and verbose"
    JSON schema rendering issues with complex nested Pydantic models.

    Key simplifications from GreenhouseLightingAnalysis:
    - Removed Optional/Enum complexity (uses_growlight is always required)
    - Removed list types (sources embedded in reasoning)
    - Removed validators (validation moved to application code)
    - Removed nice-to-have fields (species_grown, lighting_type)

    Design Philosophy:
    - Pure data transfer object (DTO) for LLM generation
    - Business logic validation happens after parsing
    - Optimized for reliable JSON schema generation
    """

    is_greenhouse: bool = Field(
        ...,
        description=(
            "Is this location an actual greenhouse facility? "
            "Must be true or false. "
            "False for auction houses, seed companies, transport, storage."
        ),
    )

    uses_growlight: Literal["YES", "NO", "UNKNOWN", "NOT_APPLICABLE"] = Field(
        ...,
        description=(
            "Classification of growlight usage. Must be one of: "
            "'YES' (confirmed artificial lighting), "
            "'NO' (confirmed no artificial lighting), "
            "'UNKNOWN' (evidence is inconclusive), or "
            "'NOT_APPLICABLE' (if is_greenhouse is false)."
        ),
    )

    confidence: float = Field(
        ...,
        description=(
            "Overall confidence score for the is_greenhouse classification, "
            "from 0.0 (no confidence) to 1.0 (complete confidence)."
        ),
    )

    reasoning: str = Field(
        ...,
        description=(
            "Step-by-step reasoning for the classification. "
            "Include: (1) Direct facility evidence, "
            "(2) WUR research if consulted, "
            "(3) Final decision with justification. "
            "IMPORTANT: Include all source URLs directly in this text."
        ),
    )


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


# Phase 1: Context-aware signature for dependency injection


class GreenhouseClassificationWithContext(Signature):  # type: ignore[misc]
    """
    DSPy Signature for greenhouse detection using pre-retrieved evidence.

    This signature enables dependency injection by accepting evidence as a
    context string rather than performing its own web search. Critical for
    deterministic optimization on cached data (Phase 1 architecture).

    The signature implements the same hierarchical classification workflow as
    GreenhouseDetectionSignature but operates on provided evidence:
    1. Determine if location is an actual greenhouse
    2. Search provided evidence for direct indicators of growlights
    3. If unclear, look for WUR research in evidence
    4. Cross-validate with species patterns

    This decouples retrieval from classification, enabling both:
    - Training: CachedRetriever provides frozen evidence (deterministic)
    - Production: PerplexityRetriever provides live search results
    """

    # Input fields
    location_name: str = dspy.InputField(desc="Company or facility name to classify")
    location_area: str = dspy.InputField(
        desc="City or region (e.g., 'waddinxveen, Netherlands')"
    )
    evidence_context: str = dspy.InputField(
        desc=(
            "Pre-retrieved search results and evidence regarding the location. "
            "Analyze this provided evidence rather than performing your own search. "
            "The evidence includes search queries, responses, and source citations."
        )
    )

    # Output fields (using str types to disable DSPy's automatic JSON schema mode)
    # This prevents JSONAdapter from sending response_format that Perplexity rejects
    is_greenhouse: str = dspy.OutputField(
        desc=(
            "Is this an actual greenhouse facility? "
            "Must be 'true' or 'false' (as a string). "
            "Use 'false' for auction houses (bloemenveiling), seed companies, "
            "transport companies, caravan storage, etc."
        )
    )
    uses_growlight: str = dspy.OutputField(
        desc=(
            "Classification of growlight usage. Must be one of: "
            "'YES' (confirmed artificial lighting), "
            "'NO' (confirmed no artificial lighting), "
            "'UNKNOWN' (evidence is inconclusive), or "
            "'NOT_APPLICABLE' (if is_greenhouse is false)."
            "\n\nCLASSIFICATION WORKFLOW:\n"
            "1. Analyze provided evidence for DIRECT facility indicators\n"
            "2. If unclear, check for WUR research in evidence\n"
            "   - Look for wur.nl URLs or WUR publications\n"
            "   - Check for 'assimilatieverlichting voor {species}'\n"
            "3. Cross-validate with species:\n"
            "   - Roses/Gerberas/Lilies/Chrysanthemums → almost always YES\n"
            "   - Tomatoes/Peppers → check WUR for Dutch climate\n"
            "4. If still unclear → UNKNOWN"
        )
    )
    confidence: float = dspy.OutputField(
        desc=(
            "Overall confidence score for the is_greenhouse classification, "
            "from 0.0 to 1.0 (e.g., 0.85). "
            "Direct facility evidence = highest confidence. "
            "WUR research support = medium-high confidence. "
            "No evidence = low confidence."
        )
    )
    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning for the classification. Include:\n"
            "1. Direct facility evidence from provided context\n"
            "2. WUR research if found in evidence (CITE wur.nl URLs)\n"
            "3. Species validation if relevant\n"
            "4. Final decision with justification\n"
            "\nIMPORTANT: Include all source URLs directly in this text. "
            "CITE ALL URLS from the provided evidence."
        )
    )


# DSPy Module


class GreenhouseDetector(dspy.Module):  # type: ignore[misc]
    """
    DSPy Module for greenhouse detection with dependency injection.

    This module decouples retrieval from classification by accepting a Retriever
    via dependency injection. This enables:
    - Deterministic optimization on cached data (CachedRetriever)
    - Production use with live search (PerplexityRetriever)

    The two-stage architecture (retrieve → classify) prevents temporal overfitting
    and enables reproducible DSPy optimization on frozen evidence.

    Example:
        >>> from sevenrad_ee.ai.retrievers import CachedRetriever
        >>> from sevenrad_ee.ai.dspy_perplexity import PerplexityLM
        >>> import dspy
        >>>
        >>> # Configure LM (use PerplexityLM for native structured outputs)
        >>> lm = PerplexityLM(model="sonar-pro")
        >>> dspy.configure(lm=lm)
        >>>
        >>> # Create detector with cached retriever
        >>> cache_dir = Path("data/research")
        >>> cached_retriever = CachedRetriever(cache_dir=cache_dir)
        >>> detector = GreenhouseDetector(retriever=cached_retriever)
        >>>
        >>> # Classify using cached evidence
        >>> result = detector(
        ...     location_name="Royal Van Zanten",
        ...     location_area="Rijsenhout"
        ... )
        >>> print(result.uses_growlight)

    """

    def __init__(self, retriever: "Retriever") -> None:
        """
        Initialize greenhouse detector with injected retriever.

        Args:
            retriever: Any object conforming to Retriever protocol
                      (must implement forward(company_name, location) -> str)

        """
        super().__init__()
        self.retriever = retriever
        # Use ChainOfThought - DSPy 3.x will use JSON adapter for structured output
        # The signature has proper type hints which triggers JSON mode automatically
        self.predictor = dspy.ChainOfThought(GreenhouseClassificationWithContext)

    def forward(
        self,
        location_name: str,
        location_area: str,
    ) -> dspy.Prediction:
        """
        Analyze a location using two-stage architecture.

        Stage 1: Retrieve evidence using injected retriever
        Stage 2: Classify based on retrieved evidence

        Args:
            location_name: Name of the facility/company
            location_area: City or region (e.g., 'waddinxveen, Netherlands')

        Returns:
            dspy.Prediction with fields matching GreenhouseClassificationWithContext

        """
        request_id = request_id_var.get()

        logger.info("=" * 60)
        logger.info("GreenhouseDetector.forward() called")
        logger.info(f"  Location: '{location_name}' in '{location_area}'")
        logger.debug(
            f"  location_name type: {type(location_name)}, length: {len(location_name)}"
        )
        logger.debug(
            f"  location_area type: {type(location_area)}, length: {len(location_area)}"
        )

        # ===== STAGE 1: RETRIEVE EVIDENCE =====
        logger.info("STAGE 1: Retrieving evidence...")
        try:
            evidence_context = self.retriever(  # type: ignore[operator]
                company_name=location_name,
                location=location_area,
            )
            logger.info(
                f"Evidence retrieval successful ({len(evidence_context)} chars)"
            )
            logger.debug(
                f"Evidence preview (first 300 chars): {evidence_context[:300]}"
            )
        except Exception as e:
            logger.error("STAGE 1 FAILED: Evidence retrieval failed")
            logger.error(f"Error type: {type(e).__name__}")
            logger.exception("Retrieval error details:")
            raise

        # ===== STAGE 2: CLASSIFY WITH EVIDENCE =====
        logger.info("STAGE 2: Classifying with retrieved evidence...")
        logger.debug(
            f"Calling predictor with {len(evidence_context)} chars of evidence"
        )

        try:
            prediction = self.predictor(
                location_name=location_name,
                location_area=location_area,
                evidence_context=evidence_context,
            )
            logger.info("Classification successful")
            logger.info(
                f"  Result: is_greenhouse={prediction.is_greenhouse}, "
                f"uses_growlight={prediction.uses_growlight}, "
                f"confidence={prediction.confidence}"
            )
            logger.debug(f"  Prediction type: {type(prediction)}")
            logger.debug(f"  Prediction fields: {list(prediction.__dict__.keys())}")

            # Log reasoning preview
            if hasattr(prediction, "reasoning"):
                reasoning_preview = (
                    prediction.reasoning[:200]
                    if len(prediction.reasoning) > 200
                    else prediction.reasoning
                )
                logger.debug(f"  Reasoning preview: {reasoning_preview}")

        except Exception as e:
            logger.error("STAGE 2 FAILED: Classification failed")
            logger.error(f"Error type: {type(e).__name__}")
            logger.exception("Classification error details:")
            raise

        logger.info("GreenhouseDetector.forward() completed successfully")
        logger.info("=" * 60)

        return prediction

    def to_pydantic(self, prediction: dspy.Prediction) -> SimplifiedGreenhouseAnalysis:
        """
        Convert DSPy Prediction to Pydantic model for validation.

        Args:
            prediction: DSPy Prediction from forward()

        Returns:
            SimplifiedGreenhouseAnalysis validated Pydantic model

        Raises:
            ValueError: If prediction violates Pydantic validation

        """
        # Parse uses_growlight - ensure it's a valid Literal value
        uses_growlight_str = prediction.uses_growlight.strip().upper()
        if uses_growlight_str not in ("YES", "NO", "UNKNOWN", "NOT_APPLICABLE"):
            logger.warning(
                "Invalid uses_growlight value '%s', defaulting to UNKNOWN",
                prediction.uses_growlight,
            )
            uses_growlight_str = "UNKNOWN"

        # Build simplified Pydantic model
        return SimplifiedGreenhouseAnalysis(
            is_greenhouse=bool(prediction.is_greenhouse),
            uses_growlight=uses_growlight_str,
            confidence=float(prediction.confidence),
            reasoning=prediction.reasoning,
        )


# Helper function for convenience


def analyze_greenhouse(
    location_name: str,
    location_area: str,
    lm: dspy.LM | None = None,
) -> SimplifiedGreenhouseAnalysis:
    """
    Analyze a location for greenhouse and artificial lighting usage.

    Convenience function that creates a detector with live Perplexity retrieval,
    runs prediction, and returns a validated Pydantic model. This is the primary
    entry point for production use with live web search.

    For cached/deterministic evaluation, use GreenhouseDetector directly with
    a CachedRetriever instance.

    Args:
        location_name: Name of the facility/company
        location_area: City or region (e.g., 'waddinxveen, Netherlands')
        lm: Optional DSPy LM instance. If None, uses dspy.settings.lm

    Returns:
        SimplifiedGreenhouseAnalysis with hierarchical classification

    Raises:
        ValueError: If no LM is configured and none provided, or if
                    Pydantic validation fails

    Example:
        >>> import dspy
        >>> from sevenrad_ee.ai.dspy_perplexity import PerplexityLM
        >>>
        >>> # Use PerplexityLM for native structured outputs (bypasses LiteLLM)
        >>> lm = PerplexityLM(model="sonar-pro")
        >>> dspy.configure(lm=lm)
        >>>
        >>> result = analyze_greenhouse(
        ...     location_name="Marjoland",
        ...     location_area="waddinxveen, Netherlands"
        ... )
        >>> print(f"Is greenhouse: {result.is_greenhouse}")
        >>> print(f"Uses growlight: {result.uses_growlight}")
        >>> print(f"Confidence: {result.confidence:.2%}")

    """
    # Import here to avoid circular dependency
    from sevenrad_ee.ai.retrievers import PerplexityRetriever

    # Create live retriever for production use
    live_retriever = PerplexityRetriever()

    if lm:
        with dspy.context(lm=lm):
            detector = GreenhouseDetector(retriever=live_retriever)
            prediction = detector(location_name, location_area)
            return detector.to_pydantic(prediction)

    # Use default LM from dspy.settings
    detector = GreenhouseDetector(retriever=live_retriever)
    prediction = detector(location_name, location_area)
    return detector.to_pydantic(prediction)
