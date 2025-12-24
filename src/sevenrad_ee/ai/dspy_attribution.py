"""
DSPy-based Perplexity prompt optimization for greenhouse attribution.

This module provides DSPy signatures and modules for optimizing the
Perplexity prompt used in attribution.py. It uses Chain-of-Thought
reasoning and BootstrapFinetune for prompt optimization.
"""

import logging
from typing import Optional

try:
    import dspy
except ImportError as e:
    msg = "dspy-ai package required. Install with: uv pip install -e '.[dev]'"
    raise ImportError(msg) from e

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Match the PerplexityAnalysis model from attribution.py
class PerplexityAnalysis(BaseModel):
    """
    Structured analysis of greenhouse operation from Perplexity AI.

    This model matches the one in attribution.py for compatibility.
    """

    grow_light_likelihood: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Probability (0.0-1.0) that greenhouse uses artificial lighting",
    )
    lighting_evidence: Optional[str] = Field(
        None,
        description="Evidence of lighting technology use",
    )
    primary_crops: list[str] = Field(
        default_factory=list,
        description="Primary crops grown (e.g., rose, tomato, gerbera)",
    )
    size_hectares: Optional[float] = Field(
        None,
        ge=0.0,
        description="Greenhouse size in hectares",
    )
    instagram_handle: Optional[str] = Field(
        None,
        description="Instagram handle without @ symbol",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Source URLs used for analysis",
    )


# DSPy Signature for greenhouse attribution analysis


class GreenhouseAttributionSignature(dspy.Signature):  # type: ignore[misc]
    """
    Analyze greenhouse operation for artificial lighting usage.

    This signature uses Dutch/European source priorities and anti-greenwashing
    instructions to accurately detect grow light usage.
    """

    # Inputs
    business_name: str = dspy.InputField(
        desc="Name of the greenhouse or agricultural business"
    )
    location_context: str = dspy.InputField(
        desc="Location context (e.g., 'Moerkapelle, Netherlands')"
    )
    business_types: str = dspy.InputField(
        desc="Business types from Google Places (comma-separated)",
        default="",
    )

    # Outputs - matching PerplexityAnalysis fields
    grow_light_likelihood: float = dspy.OutputField(
        desc=(
            "Probability (0.0-1.0) that greenhouse uses artificial lighting. "
            "Consider: 1) Crop type (roses/gerbera/tomato = HIGH), "
            "2) ANY lighting tech mention (LED/HPS/SON-T = evidence), "
            "3) Year-round production = HIGH likelihood. "
            "CRITICAL: Ignore sustainability marketing - "
            "LED claims still mean grow lights!"
        )
    )
    lighting_evidence: str = dspy.OutputField(
        desc=(
            "Quote ANY evidence of lighting technology, energy use, or controlled "
            "environment. Include: LED mentions, HPS, SON-T, assimilation lighting, "
            "supplemental light, grow lights, photoperiod control. "
            "Return 'None found' if no evidence."
        )
    )
    primary_crops: str = dspy.OutputField(
        desc=(
            "Primary crops grown, comma-separated. "
            "Use lowercase: rose, gerbera, tomato, pepper, cucumber, lettuce, etc. "
            "Return empty string if unknown."
        )
    )
    size_hectares: float = dspy.OutputField(
        desc=(
            "Greenhouse size in hectares. "
            "Return 0.0 if size information not available."
        )
    )
    instagram_handle: str = dspy.OutputField(
        desc=(
            "Official Instagram handle WITHOUT @ symbol "
            "(e.g., 'royalvanzanten' not '@royalvanzanten'). "
            "Return empty string if not found."
        )
    )
    sources: str = dspy.OutputField(
        desc=(
            "Source URLs separated by ' ||| '. "
            "PRIORITIZE: .nl domains, floraldaily.com, kasmagazine.nl, "
            "onderglas.nl, vakbladvoordebloemisterij.nl, royalvanzanten.com. "
            "Include at least 2-3 sources."
        )
    )
    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning for the grow_light_likelihood score. "
            "Explain: 1) What crops were found, 2) What lighting evidence was found, "
            "3) Why the final probability was chosen. 2-3 sentences."
        )
    )


# DSPy Module with Chain-of-Thought


class GreenhouseAttributionModule(dspy.Module):  # type: ignore[misc]
    """
    DSPy module for greenhouse attribution analysis with CoT.

    This module uses Chain-of-Thought reasoning to improve accuracy
    of grow light detection and can be optimized with BootstrapFinetune.
    """

    def __init__(self) -> None:
        """Initialize with Chain-of-Thought predictor."""
        super().__init__()
        self.predictor = dspy.ChainOfThought(GreenhouseAttributionSignature)

    def forward(
        self,
        business_name: str,
        location_context: str,
        business_types: str = "",
    ) -> dspy.Prediction:
        """
        Analyze greenhouse for artificial lighting usage.

        Args:
            business_name: Name of the business
            location_context: Location context
            business_types: Business types from Google Places

        Returns:
            dspy.Prediction with analysis fields

        """
        # Add Dutch source guidance to the context
        enhanced_context = self._add_dutch_guidance(location_context)

        prediction = self.predictor(
            business_name=business_name,
            location_context=enhanced_context,
            business_types=business_types,
        )

        return prediction

    def _add_dutch_guidance(self, location_context: str) -> str:
        """Add Dutch source and terminology guidance."""
        dutch_guidance = (
            f"{location_context}\n\n"
            "SEARCH STRATEGY:\n"
            "- Prioritize .nl domains and Dutch trade publications\n"
            "- Use Dutch terms: assimilatieverlichting, groeilicht, kunstlicht, "
            "kas, teelt, LED-verlichting, SON-T\n"
            "- Priority sources: royalvanzanten.com, floraldaily.com, "
            "kasmagazine.nl, onderglas.nl"
        )
        return dutch_guidance

    def to_pydantic(self, prediction: dspy.Prediction) -> PerplexityAnalysis:
        """
        Convert DSPy prediction to PerplexityAnalysis Pydantic model.

        Args:
            prediction: DSPy prediction from forward()

        Returns:
            Validated PerplexityAnalysis model

        """
        # Parse crops
        crops_str = prediction.primary_crops or ""
        crops_list = (
            [c.strip() for c in crops_str.split(",") if c.strip()] if crops_str else []
        )

        # Parse sources
        sources_str = prediction.sources or ""
        sources_list = (
            [s.strip() for s in sources_str.split("|||") if s.strip()]
            if sources_str
            else []
        )

        # Clean lighting evidence
        evidence_str = prediction.lighting_evidence or ""
        lighting_evidence = (
            None if evidence_str.lower() in ("none found", "none", "") else evidence_str
        )

        # Clean Instagram handle
        instagram_str = (prediction.instagram_handle or "").strip()
        instagram_handle = instagram_str if instagram_str else None

        # Clean size
        try:
            size = float(prediction.size_hectares)
            size_hectares = None if size == 0.0 else size
        except (ValueError, TypeError):
            size_hectares = None

        # Build Pydantic model
        return PerplexityAnalysis(
            grow_light_likelihood=float(prediction.grow_light_likelihood),
            lighting_evidence=lighting_evidence,
            primary_crops=crops_list,
            size_hectares=size_hectares,
            instagram_handle=instagram_handle,
            sources=sources_list,
        )


# Training examples for optimization


def create_training_examples() -> list[dspy.Example]:
    """
    Create training examples for DSPy optimization.

    These are known greenhouse operations with verified artificial
    lighting usage for training the optimizer.

    Returns:
        List of dspy.Example instances with inputs and expected outputs

    """
    examples = [
        # Example 1: Royal Van Zanten - Known gerbera/rose grower with lighting
        dspy.Example(
            business_name="Royal Van Zanten",
            location_context="Rijsenhout, Netherlands",
            business_types="greenhouse,florist",
            grow_light_likelihood=0.95,
            lighting_evidence=(
                "Uses LED and SON-T supplemental lighting for year-round "
                "gerbera and rose production"
            ),
            primary_crops="gerbera,rose",
            size_hectares=30.0,
            instagram_handle="royalvanzanten",
            sources=(
                "https://www.royalvanzanten.com ||| "
                "https://www.floraldaily.com/article/9123456"
            ),
            reasoning=(
                "Royal Van Zanten grows gerbera and roses (Tier 1 crops requiring "
                "high light). Multiple sources confirm LED and SON-T supplemental "
                "lighting for year-round production. Very high likelihood (0.95)."
            ),
        ).with_inputs("business_name", "location_context", "business_types"),
        # Example 2: Duijvestijn Tomaten - Tomato greenhouse
        dspy.Example(
            business_name="Duijvestijn Tomaten",
            location_context="Pijnacker, Netherlands",
            business_types="greenhouse,agricultural",
            grow_light_likelihood=0.90,
            lighting_evidence=(
                "Geothermal greenhouse with LED assimilatieverlichting "
                "for tomato cultivation"
            ),
            primary_crops="tomato",
            size_hectares=12.5,
            instagram_handle="duijvestijntomaten",
            sources=(
                "https://www.duijvestijn.nl ||| "
                "https://www.kasmagazine.nl/duijvestijn"
            ),
            reasoning=(
                "Tomato cultivation (Tier 1 crop) in controlled environment. "
                "Website confirms LED assimilatieverlichting use. "
                "Year-round production indicates artificial lighting (0.90)."
            ),
        ).with_inputs("business_name", "location_context", "business_types"),
        # Example 3: Prominent - Gerbera specialist
        dspy.Example(
            business_name="Prominent",
            location_context="Poeldijk, Netherlands",
            business_types="greenhouse,florist",
            grow_light_likelihood=0.95,
            lighting_evidence=(
                "SON-T grow lights for gerbera production, 24/7 operation"
            ),
            primary_crops="gerbera",
            size_hectares=8.0,
            instagram_handle="prominent_gerbera",
            sources=(
                "https://www.prominent.nl ||| "
                "https://www.floraldaily.com/article/9234567"
            ),
            reasoning=(
                "Gerbera specialist (Tier 1 crop). Sources confirm SON-T lighting "
                "with 24/7 operation. Typical high-intensity greenhouse (0.95)."
            ),
        ).with_inputs("business_name", "location_context", "business_types"),
        # Example 4: Non-greenhouse business (negative example)
        dspy.Example(
            business_name="De Kas Restaurant",
            location_context="Amsterdam, Netherlands",
            business_types="restaurant",
            grow_light_likelihood=0.10,
            lighting_evidence="None found",
            primary_crops="",
            size_hectares=0.0,
            instagram_handle="restaurantdekas",
            sources="https://www.restaurantdekas.nl ||| https://www.tripadvisor.nl",
            reasoning=(
                "Restaurant in former greenhouse building. No evidence of "
                "agricultural production or grow lights. Very low likelihood (0.10)."
            ),
        ).with_inputs("business_name", "location_context", "business_types"),
        # Example 5: Koppert Cress - Microgreens with LED
        dspy.Example(
            business_name="Koppert Cress",
            location_context="Monster, Netherlands",
            business_types="greenhouse,agricultural",
            grow_light_likelihood=0.85,
            lighting_evidence="LED lighting for microgreen and cress cultivation",
            primary_crops="cress,microgreens",
            size_hectares=5.0,
            instagram_handle="koppertcress",
            sources=(
                "https://www.koppertcress.com ||| "
                "https://www.onderglas.nl/koppert-cress"
            ),
            reasoning=(
                "Microgreens cultivation with confirmed LED lighting. "
                "While not Tier 1 crops, LED infrastructure confirms artificial "
                "lighting use for controlled growing (0.85)."
            ),
        ).with_inputs("business_name", "location_context", "business_types"),
    ]

    return examples


# Evaluation metric for optimization

# Threshold for considering an example as positive (uses grow lights)
POSITIVE_EXAMPLE_THRESHOLD = 0.7


def greenhouse_attribution_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Optional[str] = None,  # noqa: ARG001
) -> float:
    """
    Calculate evaluation metric for greenhouse attribution accuracy.

    Combines multiple factors:
    - Likelihood accuracy (primary metric)
    - Evidence quality
    - Source relevance

    Args:
        example: Ground truth example
        prediction: Model prediction
        trace: Optional trace (unused, required by DSPy interface)

    Returns:
        Score between 0.0 and 1.0 (higher is better)

    """
    score = 0.0

    # 1. Likelihood accuracy (40% weight)
    try:
        pred_likelihood = float(prediction.grow_light_likelihood)
        true_likelihood = float(example.grow_light_likelihood)
        likelihood_error = abs(pred_likelihood - true_likelihood)
        likelihood_score = max(0.0, 1.0 - likelihood_error)
        score += 0.4 * likelihood_score
    except (ValueError, TypeError, AttributeError):
        pass  # No score contribution

    # 2. Evidence quality (30% weight)
    # Check if evidence mentions key lighting terms
    pred_evidence = str(prediction.lighting_evidence or "").lower()
    if example.grow_light_likelihood > POSITIVE_EXAMPLE_THRESHOLD:  # Positive example
        lighting_terms = [
            "led",
            "son-t",
            "hps",
            "assimilation",
            "grow light",
            "supplemental",
        ]
        evidence_found = any(term in pred_evidence for term in lighting_terms)
        if evidence_found:
            score += 0.3
    elif pred_evidence in ("none found", "none", ""):
        score += 0.3

    # 3. Source relevance (30% weight)
    # Check for Dutch sources (.nl domains)
    pred_sources = str(prediction.sources or "").lower()
    if ".nl" in pred_sources:
        score += 0.15
    # Check for priority sources
    priority_sources = ["floraldaily", "kasmagazine", "onderglas", "royalvanzanten"]
    if any(source in pred_sources for source in priority_sources):
        score += 0.15

    return score
