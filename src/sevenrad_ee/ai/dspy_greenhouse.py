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
from enum import Enum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

if TYPE_CHECKING:
    from sevenrad_ee.ai.retrievers import Retriever

try:
    import dspy
    from dspy import Signature
except ImportError as e:
    msg = (
        "dspy-ai package is required. Install with: "
        "uv pip install -e '.[dev]' or pip install dspy-ai~=2.5.0"
    )
    raise ImportError(msg) from e


class GrowlightUsage(str, Enum):
    """Classification of growlight usage in greenhouses."""

    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class GroeilichtGebruik(str, Enum):
    """Dutch classification of growlight usage in greenhouses."""

    JA = "JA"
    NEE = "NEE"
    ONBEKEND = "ONBEKEND"

    def to_english(self) -> GrowlightUsage:
        """Convert Dutch classification to English equivalent."""
        mapping = {
            GroeilichtGebruik.JA: GrowlightUsage.YES,
            GroeilichtGebruik.NEE: GrowlightUsage.NO,
            GroeilichtGebruik.ONBEKEND: GrowlightUsage.UNKNOWN,
        }
        return mapping[self]

    @classmethod
    def from_string(cls, value: str) -> "GroeilichtGebruik":
        """Parse string to enum, case-insensitive."""
        value_upper = value.strip().upper()
        if value_upper in ("JA", "YES", "TRUE"):
            return cls.JA
        elif value_upper in ("NEE", "NO", "FALSE"):
            return cls.NEE
        else:
            return cls.ONBEKEND


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
        logger.info("  Location: '%s' in '%s'", location_name, location_area)
        logger.debug(
            "  location_name type: %s, length: %d",
            type(location_name),
            len(location_name),
        )
        logger.debug(
            "  location_area type: %s, length: %d",
            type(location_area),
            len(location_area),
        )

        # ===== STAGE 1: RETRIEVE EVIDENCE =====
        logger.info("STAGE 1: Retrieving evidence...")
        try:
            evidence_context = self.retriever(  # type: ignore[operator]
                company_name=location_name,
                location=location_area,
            )
            logger.info(
                "Evidence retrieval successful (%d chars)", len(evidence_context)
            )
            logger.debug(
                "Evidence preview (first 300 chars): %s", evidence_context[:300]
            )
        except Exception as e:
            logger.error("STAGE 1 FAILED: Evidence retrieval failed")
            logger.error("Error type: %s", type(e).__name__)
            logger.exception("Retrieval error details:")
            raise

        # ===== STAGE 2: CLASSIFY WITH EVIDENCE =====
        logger.info("STAGE 2: Classifying with retrieved evidence...")
        logger.debug(
            "Calling predictor with %d chars of evidence", len(evidence_context)
        )

        try:
            prediction = self.predictor(
                location_name=location_name,
                location_area=location_area,
                evidence_context=evidence_context,
            )
            logger.info("Classification successful")
            logger.info(
                "  Result: is_greenhouse=%s, uses_growlight=%s, confidence=%s",
                prediction.is_greenhouse,
                prediction.uses_growlight,
                prediction.confidence,
            )
            logger.debug("  Prediction type: %s", type(prediction))
            logger.debug("  Prediction fields: %s", list(prediction.__dict__.keys()))

            # Log reasoning preview
            if hasattr(prediction, "reasoning"):
                max_preview_len = 200
                reasoning_preview = (
                    prediction.reasoning[:max_preview_len]
                    if len(prediction.reasoning) > max_preview_len
                    else prediction.reasoning
                )
                logger.debug("  Reasoning preview: %s", reasoning_preview)

        except Exception as e:
            logger.error("STAGE 2 FAILED: Classification failed")
            logger.error("Error type: %s", type(e).__name__)
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


# Phase 3 GEPA Optimization Plan - Native Perplexity RAG
# ======================================================
#
# The following classes implement the new architecture where Perplexity
# performs search + reasoning + citation in a single call, rather than
# using it as a dumb search API with a separate classifier.


class ParseError(ValueError):
    """Raised when prediction parsing fails with invalid values."""

    pass


class PerplexityGreenhouseClassifier(Signature):  # type: ignore[misc]
    """
    Classify Dutch greenhouse companies using Perplexity's native web search.

    CRITICAL: Growers do NOT advertise lighting! Use CAUSAL REASONING:

    STEP 1: Find what crops they grow (search in Dutch AND English)
    - Search: "{company} kwekerij gewassen" / "{company} greenhouse crops"
    - Look for: rozen, tomaten, orchideeën, gerbera, chrysanten, paprika, etc.

    STEP 2: Infer lighting from crop type using this knowledge:

    ALMOST ALWAYS USE LIGHTING (say YES if found):
    - Roses (rozen): SON-T/Hybrid, 150-250 umol
    - Tomatoes (tomaten, year-round): SON-T/Hybrid, 180-350 umol
    - Gerbera: SON-T/Hybrid, 120-200 umol
    - Alstroemeria: SON-T/Hybrid, 120-200 umol
    - Freesia: SON-T, 100-180 umol
    - Phalaenopsis orchids: LED, 40-120 umol

    OFTEN USE LIGHTING (say YES if year-round production):
    - Bell peppers (paprika): SON-T/Hybrid if winter production
    - Cucumbers (komkommers): LED/Hybrid if winter production
    - Eggplant (aubergine): SON-T if year-round
    - Herbs (kruiden): LED if year-round supply

    SOMETIMES/RARELY (say UNKNOWN unless clear evidence):
    - Chrysanthemums: photoperiod control, not always assimilation
    - Lilies: temperature-driven forcing
    - Tulips: temperature-driven forcing
    - Cymbidium orchids: typically NO lighting
    - Anthurium: shade-loving, moderate light

    STEP 3: Check for negative indicators
    - "onbelichte teelt" = unlit cultivation -> NO
    - "daglichtkas" = daylight-only -> NO
    - Summer-only/seasonal production -> likely NO

    HIERARCHICAL LOGIC:
    - If is_greenhouse = 'false' -> uses_growlight MUST be 'UNKNOWN'
    - If is_greenhouse = 'true' -> infer uses_growlight from crop type
    - If crop type unknown -> uses_growlight = 'UNKNOWN'
    """

    # Input fields (minimal - let Perplexity search)
    location_name: str = dspy.InputField(desc="Company name to research")
    location_area: str = dspy.InputField(
        desc="City or region in Netherlands (e.g., 'Waddinxveen')"
    )

    # Core classification outputs
    is_greenhouse: str = dspy.OutputField(
        desc=(
            "Is this a greenhouse/kwekerij? 'true' or 'false'. "
            "Search for company info, look for terms like 'kwekerij', 'glastuinbouw'. "
            "'false' for auction houses, seed companies, transport, storage."
        )
    )

    uses_growlight: str = dspy.OutputField(
        desc=(
            "Infer from crop type: 'YES', 'NO', or 'UNKNOWN'. "
            "If roses/tomaten/gerbera/alstroemeria/freesia/Phalaenopsis -> YES. "
            "If paprika/komkommers/aubergine with year-round production -> YES. "
            "If chrysanten/lelies/tulpen/cymbidium/anthurium -> UNKNOWN. "
            "If 'onbelichte teelt' or 'daglichtkas' found -> NO. "
            "MUST be 'UNKNOWN' if is_greenhouse='false' or crop unknown."
        )
    )

    # Crop detection - critical for causal inference
    species_grown: str = dspy.OutputField(
        desc=(
            "Comma-separated crops found (Dutch or English): "
            "rozen, tomaten, gerbera, orchideeën, chrysanten, paprika, etc. "
            "This is CRITICAL for inferring lighting usage."
        )
    )

    # Evidence capture - critical for scoring
    dutch_terms_found: str = dspy.OutputField(
        desc=(
            "Comma-separated Dutch greenhouse/lighting terms found. "
            "Examples: assimilatiebelichting, belichte teelt, glastuinbouw, "
            "kunstlicht, kwekerij, SON-T, LED-belichting, groeilicht."
        )
    )

    evidence_sources: str = dspy.OutputField(
        desc=(
            'JSON array of sources: [{"url": "...", "quote": "...", "tier": "..."}]. '
            "Include the URLs from your search results with relevant quotes. "
            "Tiers: 'company_website', 'supplier_case_study', 'job_posting' (tier 2); "
            "'trade_media_nl' (tier 1); 'general_web' (tier 0.5)."
        )
    )

    confidence: float = dspy.OutputField(
        desc=(
            "Classification confidence 0.0-1.0. "
            "High (>0.8): Dutch sources + terminology + tier 2 evidence. "
            "Medium (0.5-0.8): Some Dutch evidence or tier 1 sources. "
            "Low (<0.5): General web sources or ambiguous evidence."
        )
    )

    # Reflection - forces model to verify causal reasoning
    verification_note: str = dspy.OutputField(
        desc=(
            "Verify your causal reasoning: "
            "1. What crop(s) did you find? "
            "2. According to the domain knowledge, does this crop use lighting? "
            "3. Is your uses_growlight consistent with the crop type? "
            "If crop is roses/tomaten/gerbera, uses_growlight should be YES."
        )
    )

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step causal reasoning:\n"
            "1. What crops does this company grow? (cite sources)\n"
            "2. Based on domain knowledge, what lighting do these crops need?\n"
            "3. Any negative indicators (onbelichte teelt, daglichtkas)?\n"
            "4. Final inference: crop type -> lighting usage.\n"
            "CITE ALL URLs from your search results."
        )
    )


# Phase 4: Full Dutch Signature with Causal Reasoning
# ====================================================
#
# Critical insight: Growers do NOT advertise lighting usage.
# Searching for "assimilatiebelichting" or "SON-T" is useless.
# Instead: Identify the CROP and infer lighting from domain knowledge.
#
# Full Dutch signature forces Perplexity to:
# - Search Dutch sources naturally
# - Perform chain-of-thought reasoning in Dutch
# - Surface Dutch horticultural terminology


class PerplexityKasClassificatie(Signature):  # type: ignore[misc]
    """
    Classificeer of een locatie een commerciële kas is met belichting.

    BELANGRIJK: Telers adverteren NIET met belichting! Gebruik causaal redeneren:

    STAP 1: Zoek informatie over dit bedrijf - wat telen ze?
    - Zoek naar: "{bedrijf} kwekerij" of "{bedrijf} glastuinbouw"
    - Identificeer het hoofdgewas (rozen, tomaten, orchideeën, etc.)

    STAP 2: Als het een kas is, bepaal het hoofdgewas en seizoen.

    STAP 3: Leid af of belichting waarschijnlijk is op basis van het gewas:

    GEWASSEN DIE BIJNA ALTIJD BELICHT WORDEN (zeg JA):
    - Rozen: SON-T/Hybrid, 150-250 umol (jaarrond productie)
    - Tomaten (jaarrond): SON-T/Hybrid, 180-350 umol
    - Gerbera: SON-T/Hybrid, 120-200 umol
    - Alstroemeria: SON-T/Hybrid, 120-200 umol
    - Freesia: SON-T, 100-180 umol
    - Phalaenopsis orchideeën: LED, 40-120 umol
    - Lisianthus: SON-T/LED voor jaarrond
    - Gypsophila (gipskruid): SON-T voor jaarrond
    - Ardisia: LED voor winterverkoop
    - Kalanchoe: LED/SON-T, fotoperiodesturing

    GEWASSEN DIE VAAK BELICHT WORDEN (zeg JA bij jaarrond):
    - Paprika (jaarrond productie): SON-T/Hybrid
    - Komkommers (jaarrond): LED/Hybrid
    - Aubergine (jaarrond): SON-T
    - Kruiden (jaarrond levering): LED
    - Aardbeien (winterteelt): LED
    - Cyclamen (winterproductie): LED
    - Begonia (winterproductie): LED/SON-T

    GEWASSEN DIE SOMS BELICHT WORDEN (zeg ONBEKEND tenzij bewijs):
    - Chrysanten: fotoperiodesturing, niet altijd assimilatie
    - Lelies: temperatuurgestuurde forcering
    - Tulpen: temperatuurgestuurde forcering
    - Ranunculus, anemoon: soms belichting

    GEWASSEN DIE ZELDEN BELICHT WORDEN (zeg NEE tenzij bewijs):
    - Cymbidium orchideeën: meestal geen belichting
    - Anthurium: schaduwminnend, matig licht
    - Bromelia: tropisch, veel daglicht
    - Perkplanten, tuinplanten: seizoensgebonden
    - Vetplanten, cactussen: weinig licht nodig
    - Zaadproductie, vermeerdering: meestal daglicht

    STAP 4: Controleer op negatieve indicatoren:
    - "onbelichte teelt" = geen kunstlicht -> NEE
    - "daglichtkas" = alleen daglicht -> NEE
    - Alleen zomerproductie/seizoensgebonden -> waarschijnlijk NEE

    STAP 5: Als gewas niet in bovenstaande lijsten:
    - Zoek aanvullende informatie over belichtingsgebruik bij dit gewas
    - Bij twijfel: ONBEKEND

    LET OP: Bedrijfstype ≠ faciliteitstype
    - Zaadveredelingsbedrijf MET kassen → is_kas = True
    - Onderzoekscentrum MET kassen → is_kas = True
    - Beoordeel of er KAS-faciliteiten zijn, niet alleen de hoofdactiviteit

    HIËRARCHISCHE LOGICA:
    - Als is_kas = False → gebruikt_groeilicht MOET ONBEKEND zijn
    - Als is_kas = True → leid gebruikt_groeilicht af van gewastype
    - Als gewastype onbekend → gebruikt_groeilicht = ONBEKEND
    """

    # Invoervelden
    bedrijfsnaam: str = dspy.InputField(
        desc="Naam van het bedrijf of de kwekerij om te onderzoeken"
    )
    locatie: str = dspy.InputField(
        desc="Plaats in Nederland (bijv. 'Waddinxveen', ''s-Gravenzande')"
    )

    # Kernclassificatie
    is_kas: str = dspy.OutputField(
        desc=(
            "Is dit een kas/kwekerij/glastuinbouwbedrijf? 'true' of 'false'. "
            "Zoek naar bedrijfsinformatie, let op 'kwekerij', 'glastuinbouw'. "
            "'false' voor veilingen, zaadhandel, transport, opslag. "
            "LET OP: Zaadveredelingsbedrijf MET kassen = 'true'."
        )
    )

    # Gewasidentificatie - cruciaal voor causale inferentie
    hoofdgewas: str = dspy.OutputField(
        desc=(
            "Wat teelt dit bedrijf? Belangrijkste gewas(sen). "
            "Voorbeelden: rozen, tomaten, gerbera, orchideeën, paprika, komkommers. "
            "Dit is CRUCIAAL voor het afleiden van belichtingsgebruik. "
            "Leeg als niet gevonden of geen kas."
        )
    )

    # Seizoensinformatie
    seizoen: str = dspy.OutputField(
        desc=(
            "Productieseizoen: 'jaarrond', 'seizoensgebonden', of 'onbekend'. "
            "Jaarrond productie = waarschijnlijk belichting. "
            "Alleen zomer/seizoensgebonden = waarschijnlijk geen belichting."
        )
    )

    # Kernclassificatie belichting
    gebruikt_groeilicht: str = dspy.OutputField(
        desc=(
            "Gebruikt dit bedrijf assimilatiebelichting? 'JA', 'NEE', of 'ONBEKEND'. "
            "Leid af van gewastype volgens de domeinkennis hierboven. "
            "MOET 'ONBEKEND' zijn als is_kas='false' of gewas onbekend."
        )
    )

    # Betrouwbaarheid
    zekerheid: float = dspy.OutputField(
        desc=(
            "Betrouwbaarheid van de classificatie 0.0-1.0. "
            "Hoog (>0.8): Gewas duidelijk geïdentificeerd + past in domeinkennis. "
            "Gemiddeld (0.5-0.8): Gewas gevonden maar niet in standaardlijst. "
            "Laag (<0.5): Gewas niet gevonden of onduidelijk."
        )
    )

    # Gevonden bronnen
    bronnen: str = dspy.OutputField(
        desc=(
            "Gevonden bronnen met URLs. Formaat: URL1 | URL2 | URL3. "
            "Vermeld de belangrijkste bronnen die je hebt gevonden."
        )
    )

    # Redenering
    redenering: str = dspy.OutputField(
        desc=(
            "Stapsgewijze causale redenering:\n"
            "1. Wat heb je gevonden over dit bedrijf?\n"
            "2. Welk gewas teelt dit bedrijf?\n"
            "3. Volgens de domeinkennis, gebruikt dit gewas belichting?\n"
            "4. Zijn er negatieve indicatoren (onbelichte teelt, daglichtkas)?\n"
            "5. Conclusie: gewastype → belichtingsgebruik.\n"
            "VERMELD alle gevonden URLs."
        )
    )


class PerplexityKasDetector(dspy.Module):  # type: ignore[misc]
    """
    Dutch single-stage detector using Perplexity's native RAG capabilities.

    This detector uses Perplexity with a full Dutch signature to:
    - Force Perplexity to search Dutch sources
    - Perform chain-of-thought reasoning in Dutch
    - Apply causal reasoning: crop type → lighting inference

    Key insight: Growers don't advertise lighting. Instead of searching for
    "assimilatiebelichting", we identify the CROP and infer lighting from
    domain knowledge embedded in the signature.

    Example:
        >>> from sevenrad_ee.ai.dspy_perplexity import PerplexityLM
        >>> import dspy
        >>>
        >>> lm = PerplexityLM(model="sonar-pro", temperature=0)
        >>> dspy.configure(lm=lm)
        >>>
        >>> detector = PerplexityKasDetector()
        >>> result = detector(
        ...     bedrijfsnaam="Porta Nova",
        ...     locatie="Waddinxveen"
        ... )
        >>> print(result.gebruikt_groeilicht)  # JA, NEE, or ONBEKEND

    """

    def __init__(self) -> None:
        """Initialize detector with ChainOfThought for Dutch reasoning."""
        super().__init__()
        self.classifier = dspy.ChainOfThought(PerplexityKasClassificatie)

    def forward(
        self,
        bedrijfsnaam: str,
        locatie: str,
    ) -> dspy.Prediction:
        """
        Classificeer een locatie met Perplexity's native RAG in het Nederlands.

        Args:
            bedrijfsnaam: Naam van het bedrijf/kwekerij
            locatie: Plaats in Nederland (bijv. 'Waddinxveen')

        Returns:
            dspy.Prediction met classificatieresultaten

        """
        logger.info("=" * 60)
        logger.info("PerplexityKasDetector.forward() called")
        logger.info("  Bedrijf: '%s' in '%s'", bedrijfsnaam, locatie)

        try:
            prediction = self.classifier(
                bedrijfsnaam=bedrijfsnaam,
                locatie=locatie,
            )
            logger.info("Classificatie succesvol")
            logger.info(
                "  Resultaat: is_kas=%s, gebruikt_groeilicht=%s, zekerheid=%s",
                prediction.is_kas,
                prediction.gebruikt_groeilicht,
                prediction.zekerheid,
            )
            logger.info("  Hoofdgewas: %s", prediction.hoofdgewas)
            logger.info("  Seizoen: %s", prediction.seizoen)
            logger.debug("  Redenering: %s", prediction.redenering[:200])

        except Exception as e:
            logger.error("Classificatie MISLUKT")
            logger.error("Fouttype: %s", type(e).__name__)
            logger.exception("Classificatie foutdetails:")
            raise

        logger.info("PerplexityKasDetector.forward() voltooid")
        logger.info("=" * 60)

        return prediction

    def to_english_prediction(self, prediction: dspy.Prediction) -> dspy.Prediction:
        """
        Convert Dutch prediction to English field names for compatibility.

        Args:
            prediction: Dutch prediction from forward()

        Returns:
            dspy.Prediction with English field names

        """
        # Parse is_kas to boolean
        is_kas_str = str(prediction.is_kas).strip().lower()
        is_greenhouse = is_kas_str in ("true", "ja", "yes", "1")

        # Convert Dutch growlight to English
        growlight_dutch = GroeilichtGebruik.from_string(prediction.gebruikt_groeilicht)
        uses_growlight = growlight_dutch.to_english().value

        return dspy.Prediction(
            is_greenhouse=is_greenhouse,
            uses_growlight=uses_growlight,
            species_grown=prediction.hoofdgewas,
            confidence=float(prediction.zekerheid),
            reasoning=prediction.redenering,
            # Preserve Dutch fields for debugging
            _dutch_prediction=prediction,
        )

    def to_pydantic(self, prediction: dspy.Prediction) -> SimplifiedGreenhouseAnalysis:
        """
        Convert Dutch prediction to Pydantic model for validation.

        Args:
            prediction: Dutch prediction from forward()

        Returns:
            SimplifiedGreenhouseAnalysis validated Pydantic model

        Raises:
            ParseError: If prediction contains invalid values

        """
        # Parse is_kas
        is_kas_str = str(prediction.is_kas).strip().lower()
        if is_kas_str in ("true", "ja", "yes", "1"):
            is_greenhouse = True
        elif is_kas_str in ("false", "nee", "no", "0"):
            is_greenhouse = False
        else:
            msg = f"Invalid is_kas value: '{prediction.is_kas}'"
            logger.error("Parse failure: %s", msg)
            raise ParseError(msg)

        # Convert Dutch growlight to English
        growlight_dutch = GroeilichtGebruik.from_string(prediction.gebruikt_groeilicht)
        growlight_english = growlight_dutch.to_english()

        return SimplifiedGreenhouseAnalysis(
            is_greenhouse=is_greenhouse,
            uses_growlight=growlight_english.value,
            confidence=float(prediction.zekerheid),
            reasoning=prediction.redenering,
        )


class PerplexityGreenhouseDetector(dspy.Module):  # type: ignore[misc]
    """
    Single-stage detector using Perplexity's native RAG capabilities.

    This detector uses Perplexity to perform search + reasoning + citation
    in a single API call, rather than using a separate retriever and classifier.

    Key benefits:
    - Perplexity searches the web FOR each classification (fresh, not cached)
    - Perplexity's native citations become our evidence_sources
    - Single LLM call instead of retriever + classifier + verifier
    - MIPROv2 optimizes the one prompt that does everything

    Example:
        >>> from sevenrad_ee.ai.dspy_perplexity import PerplexityLM
        >>> import dspy
        >>>
        >>> # Configure PerplexityLM directly
        >>> lm = PerplexityLM(model="sonar-pro", temperature=0)
        >>> dspy.configure(lm=lm)
        >>>
        >>> # Create detector (no retriever needed!)
        >>> detector = PerplexityGreenhouseDetector()
        >>>
        >>> # Classify - Perplexity searches and reasons in one call
        >>> result = detector(
        ...     location_name="Porta Nova",
        ...     location_area="Waddinxveen"
        ... )
        >>> print(result.uses_growlight)

    """

    def __init__(self) -> None:
        """Initialize detector with ChainOfThought for reasoning."""
        super().__init__()
        # Perplexity does search + reasoning in one call via ChainOfThought
        self.classifier = dspy.ChainOfThought(PerplexityGreenhouseClassifier)

    def forward(
        self,
        location_name: str,
        location_area: str,
    ) -> dspy.Prediction:
        """
        Classify a location using Perplexity's native RAG.

        Perplexity performs web search, reasoning, and citation in a single call.
        No separate retriever is needed.

        Args:
            location_name: Name of the company/facility
            location_area: City or region (e.g., 'Waddinxveen, Netherlands')

        Returns:
            dspy.Prediction with classification results and evidence

        """
        request_id = request_id_var.get()

        logger.info("=" * 60)
        logger.info("PerplexityGreenhouseDetector.forward() called")
        logger.info("  Location: '%s' in '%s'", location_name, location_area)
        logger.debug(
            "  location_name type: %s, length: %d",
            type(location_name),
            len(location_name),
        )

        # Single call - Perplexity searches, reasons, and cites
        try:
            prediction = self.classifier(
                location_name=location_name,
                location_area=location_area,
            )
            logger.info("Classification successful")
            logger.info(
                "  Result: is_greenhouse=%s, uses_growlight=%s, confidence=%s",
                prediction.is_greenhouse,
                prediction.uses_growlight,
                prediction.confidence,
            )
            logger.info("  Species grown: %s", prediction.species_grown)
            logger.debug("  Dutch terms found: %s", prediction.dutch_terms_found)
            logger.debug("  Verification note: %s", prediction.verification_note)

        except Exception as e:
            logger.error("Classification FAILED")
            logger.error("Error type: %s", type(e).__name__)
            logger.exception("Classification error details:")
            raise

        logger.info("PerplexityGreenhouseDetector.forward() completed successfully")
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
            ParseError: If prediction contains invalid values (fail-fast)

        """
        # Parse is_greenhouse - handle various string formats
        is_greenhouse_str = str(prediction.is_greenhouse).strip().lower()
        if is_greenhouse_str in ("true", "yes", "1"):
            is_greenhouse = True
        elif is_greenhouse_str in ("false", "no", "0"):
            is_greenhouse = False
        else:
            msg = f"Invalid is_greenhouse value: '{prediction.is_greenhouse}'"
            logger.error("Parse failure: %s", msg)
            raise ParseError(msg)

        # Parse uses_growlight - fail-fast instead of silent UNKNOWN default
        uses_growlight_str = str(prediction.uses_growlight).strip().upper()
        valid_growlight_values: tuple[
            Literal["YES", "NO", "UNKNOWN", "NOT_APPLICABLE"], ...
        ] = ("YES", "NO", "UNKNOWN", "NOT_APPLICABLE")
        if uses_growlight_str not in valid_growlight_values:
            msg = (
                f"Invalid uses_growlight value: '{prediction.uses_growlight}'. "
                f"Must be one of: {valid_growlight_values}"
            )
            logger.error("Parse failure: %s", msg)
            raise ParseError(msg)

        # Cast to the Literal type after validation
        uses_growlight: Literal["YES", "NO", "UNKNOWN", "NOT_APPLICABLE"] = (
            uses_growlight_str  # type: ignore[assignment]
        )

        # Build simplified Pydantic model
        return SimplifiedGreenhouseAnalysis(
            is_greenhouse=is_greenhouse,
            uses_growlight=uses_growlight,
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
