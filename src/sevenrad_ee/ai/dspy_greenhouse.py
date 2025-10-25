"""
DSPy-based greenhouse detection with artificial lighting classification.

This module uses DSPy (Declarative Self-improving Python) to create
optimizable prompts for detecting greenhouse operations that use artificial
lighting. It prioritizes Dutch and German sources with domain-specific
terminology for enhanced accuracy.
"""

import logging
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

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# Pydantic models for structured output


class GreenhouseLightingAnalysis(BaseModel):
    """
    Structured analysis of greenhouse artificial lighting usage.

    This model represents the output from DSPy's analysis, designed to
    be parseable and validated with Pydantic for downstream processing.
    """

    uses_artificial_lighting: bool = Field(
        ...,
        description=(
            "Whether the greenhouse uses artificial lighting "
            "(LED, HPS, SON-T, assimilatieverlichting)"
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0) for the classification",
    )
    lighting_type: Optional[str] = Field(
        None,
        description=(
            "Type of artificial lighting detected "
            "(e.g., 'LED', 'SON-T', 'HPS', 'Mixed')"
        ),
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence excerpts from sources supporting the classification",
    )
    primary_crops: list[str] = Field(
        default_factory=list,
        description="Primary crops grown (e.g., tomaten, rozen, gerbera)",
    )
    size_hectares: Optional[float] = Field(
        None,
        ge=0.0,
        description="Greenhouse size in hectares if available",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="URLs of sources used for analysis (prioritize .nl domains)",
    )
    reasoning: Optional[str] = Field(
        None,
        description="Brief explanation of the classification decision",
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is within valid range."""
        if not 0.0 <= v <= 1.0:
            msg = "Confidence must be between 0.0 and 1.0"
            raise ValueError(msg)
        return v


# DSPy Signatures


class GreenhouseDetectionSignature(Signature):  # type: ignore[misc]
    """
    DSPy Signature for greenhouse artificial lighting detection.

    This signature defines the input/output interface for the LLM,
    with optimizable instructions that DSPy can refine through
    demonstrations and teleprompters.
    """

    # Input fields
    company_name: str = dspy.InputField(
        desc="Name of the company or greenhouse operation to analyze"
    )
    location: str = dspy.InputField(
        desc="Geographic location (city, region, country) of the operation"
    )
    additional_context: str = dspy.InputField(
        desc="Additional context or hints (e.g., business type, known crops)",
        default="",
    )

    # Output fields
    uses_artificial_lighting: bool = dspy.OutputField(
        desc=(
            "True if the greenhouse uses artificial lighting "
            "(LED, HPS, SON-T, assimilatieverlichting, groeilicht), False otherwise"
        )
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score between 0.0 and 1.0 for the classification"
    )
    lighting_type: str = dspy.OutputField(
        desc=(
            "Type of artificial lighting: 'LED', 'SON-T', 'HPS', 'Mixed', "
            "'Unknown', or 'None' if no artificial lighting"
        )
    )
    evidence: str = dspy.OutputField(
        desc=(
            "Key evidence excerpts from sources, separated by '|||'. "
            "Include Dutch/German terms like 'assimilatieverlichting', "
            "'groeilicht', 'LED-verlichting'"
        )
    )
    primary_crops: str = dspy.OutputField(
        desc=(
            "Primary crops grown, comma-separated. "
            "Use Dutch names: tomaten, rozen, gerbera, paprika, komkommer"
        )
    )
    size_hectares: float = dspy.OutputField(
        desc="Greenhouse size in hectares, or 0.0 if unknown"
    )
    sources: str = dspy.OutputField(
        desc=(
            "Source URLs separated by '|||'. "
            "Prioritize .nl domains, Dutch/German trade publications"
        )
    )
    reasoning: str = dspy.OutputField(
        desc=(
            "Brief explanation (2-3 sentences) of why the classification was made, "
            "referencing key evidence"
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
        company_name: str,
        location: str,
        additional_context: str = "",
    ) -> dspy.Prediction:
        """
        Analyze a company to detect artificial lighting usage.

        Args:
            company_name: Name of the greenhouse/company
            location: Geographic location
            additional_context: Optional additional context

        Returns:
            dspy.Prediction with fields matching GreenhouseDetectionSignature

        """
        logger.info("Analyzing %s in %s", company_name, location)

        # Enhance the query with Dutch/German source priorities
        enhanced_context = self._enhance_context(additional_context)

        # Run prediction
        prediction = self.predictor(
            company_name=company_name,
            location=location,
            additional_context=enhanced_context,
        )

        logger.info(
            "Classification: uses_lighting=%s, confidence=%.2f",
            prediction.uses_artificial_lighting,
            prediction.confidence,
        )

        return prediction

    def _enhance_context(self, base_context: str) -> str:
        """
        Enhance context with Dutch/German source priorities and terminology.

        Args:
            base_context: Original context string

        Returns:
            Enhanced context with search guidance

        """
        dutch_guidance = (
            "\n\nSOURCE PRIORITIES:\n"
            "1. Prioritize Dutch (.nl) domains and Dutch trade publications\n"
            "2. German sources acceptable for technical greenhouse information\n"
            "3. Key Dutch sources: royalvanzanten.com, floraldaily.com, "
            "vakbladvoordebloemisterij.nl, kasmagazine.nl, onderglas.nl\n"
            "\nDUTCH TERMINOLOGY TO SEARCH:\n"
            "- assimilatieverlichting (assimilation lighting)\n"
            "- groeilicht (grow light)\n"
            "- kunstlicht (artificial light)\n"
            "- kas (greenhouse)\n"
            "- teelt (cultivation)\n"
            "- LED-verlichting (LED lighting)\n"
            "- SON-T (high-pressure sodium)\n"
            "- assimilatiebelichting (assimilation lighting)\n"
            "- lichtspectrum (light spectrum)\n"
            "- Intensiteit verlichting (lighting intensity)\n"
        )

        if base_context:
            return f"{base_context}{dutch_guidance}"
        return dutch_guidance.strip()

    def to_pydantic(self, prediction: dspy.Prediction) -> GreenhouseLightingAnalysis:
        """
        Convert DSPy Prediction to Pydantic model for validation.

        Args:
            prediction: DSPy Prediction from forward()

        Returns:
            GreenhouseLightingAnalysis validated Pydantic model

        """
        # Parse delimited strings into lists
        evidence_list = (
            [e.strip() for e in prediction.evidence.split("|||") if e.strip()]
            if prediction.evidence
            else []
        )

        crops_list = (
            [c.strip() for c in prediction.primary_crops.split(",") if c.strip()]
            if prediction.primary_crops
            else []
        )

        sources_list = (
            [s.strip() for s in prediction.sources.split("|||") if s.strip()]
            if prediction.sources
            else []
        )

        # Build Pydantic model
        return GreenhouseLightingAnalysis(
            uses_artificial_lighting=bool(prediction.uses_artificial_lighting),
            confidence=float(prediction.confidence),
            lighting_type=prediction.lighting_type or None,
            evidence=evidence_list,
            primary_crops=crops_list,
            size_hectares=(
                float(prediction.size_hectares)
                if prediction.size_hectares and float(prediction.size_hectares) > 0
                else None
            ),
            sources=sources_list,
            reasoning=prediction.reasoning or None,
        )


# Helper function for convenience


def analyze_greenhouse(
    company_name: str,
    location: str,
    additional_context: str = "",
    lm: Optional[dspy.LM] = None,
) -> GreenhouseLightingAnalysis:
    """
    Analyze a greenhouse operation for artificial lighting usage.

    Convenience function that creates a detector, runs prediction,
    and returns a validated Pydantic model.

    Args:
        company_name: Name of the greenhouse/company
        location: Geographic location
        additional_context: Optional additional context
        lm: Optional DSPy LM instance. If None, uses dspy.settings.lm

    Returns:
        GreenhouseLightingAnalysis with validated results

    Raises:
        ValueError: If no LM is configured and none provided

    Example:
        >>> import dspy
        >>> lm = dspy.LM('perplexity/sonar', api_key='...')
        >>> dspy.configure(lm=lm)
        >>>
        >>> result = analyze_greenhouse(
        ...     company_name="Gerbera Breeding BV",
        ...     location="Ridderkerk, Nederland"
        ... )
        >>> print(f"Uses lighting: {result.uses_artificial_lighting}")
        >>> print(f"Confidence: {result.confidence:.2%}")

    """
    if lm:
        with dspy.context(lm=lm):
            detector = GreenhouseDetector()
            prediction = detector(company_name, location, additional_context)
            return detector.to_pydantic(prediction)

    # Use default LM from dspy.settings
    detector = GreenhouseDetector()
    prediction = detector(company_name, location, additional_context)
    return detector.to_pydantic(prediction)
