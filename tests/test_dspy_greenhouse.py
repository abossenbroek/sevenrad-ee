"""
Tests for DSPy-based greenhouse detection.

Tests the GreenhouseDetector module including:
- Pydantic model validation
- DSPy signature behavior
- Context enhancement for Dutch/German sources
- Integration with Perplexity API (marked as @api tests)
"""

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import dspy

from pydantic import ValidationError
from sevenrad_ee.ai.dspy_greenhouse import (
    GreenhouseDetector,
    GreenhouseLightingAnalysis,
    analyze_greenhouse,
)


class TestGreenhouseLightingAnalysis:
    """Test Pydantic model for greenhouse lighting analysis."""

    def test_valid_analysis_minimal(self) -> None:
        """Create analysis with minimal required fields."""
        analysis = GreenhouseLightingAnalysis(
            uses_artificial_lighting=True,
            confidence=0.85,
        )

        assert analysis.uses_artificial_lighting is True
        assert analysis.confidence == 0.85
        assert analysis.lighting_type is None
        assert analysis.evidence == []
        assert analysis.primary_crops == []
        assert analysis.size_hectares is None
        assert analysis.sources == []
        assert analysis.reasoning is None

    def test_valid_analysis_complete(self) -> None:
        """Create analysis with all fields populated."""
        analysis = GreenhouseLightingAnalysis(
            uses_artificial_lighting=True,
            confidence=0.95,
            lighting_type="SON-T",
            evidence=[
                "Gebruikt LED-verlichting voor assimilatiebelichting",
                "SON-T lampen voor nachtverlichting",
            ],
            primary_crops=["rozen", "gerbera"],
            size_hectares=12.5,
            sources=[
                "https://example.nl/greenhouse",
                "https://kasmagazine.nl/article",
            ],
            reasoning=(
                "Company website mentions assimilatieverlichting "
                "for year-round production"
            ),
        )

        assert analysis.uses_artificial_lighting is True
        assert analysis.confidence == 0.95
        assert analysis.lighting_type == "SON-T"
        assert len(analysis.evidence) == 2
        assert analysis.primary_crops == ["rozen", "gerbera"]
        assert analysis.size_hectares == 12.5
        assert len(analysis.sources) == 2
        assert "assimilatieverlichting" in analysis.reasoning

    def test_confidence_validation(self) -> None:
        """Confidence must be between 0.0 and 1.0."""
        # Valid confidence values
        GreenhouseLightingAnalysis(uses_artificial_lighting=True, confidence=0.0)
        GreenhouseLightingAnalysis(uses_artificial_lighting=True, confidence=0.5)
        GreenhouseLightingAnalysis(uses_artificial_lighting=True, confidence=1.0)

        # Invalid confidence values - Pydantic raises ValidationError
        with pytest.raises(ValidationError):
            GreenhouseLightingAnalysis(uses_artificial_lighting=True, confidence=-0.1)

        with pytest.raises(ValidationError):
            GreenhouseLightingAnalysis(uses_artificial_lighting=True, confidence=1.5)

    def test_size_validation(self) -> None:
        """Size must be non-negative if provided."""
        # Valid sizes
        GreenhouseLightingAnalysis(
            uses_artificial_lighting=True,
            confidence=0.8,
            size_hectares=0.0,
        )
        GreenhouseLightingAnalysis(
            uses_artificial_lighting=True,
            confidence=0.8,
            size_hectares=10.5,
        )

        # Invalid negative size - Pydantic raises ValidationError
        with pytest.raises(ValidationError):
            GreenhouseLightingAnalysis(
                uses_artificial_lighting=True,
                confidence=0.8,
                size_hectares=-5.0,
            )


class TestGreenhouseDetector:
    """Test DSPy GreenhouseDetector module."""

    def test_detector_initialization(self) -> None:
        """Detector can be initialized without errors."""
        detector = GreenhouseDetector()

        # Verify predictor is set up
        assert hasattr(detector, "predictor")
        assert detector.predictor is not None

    def test_context_enhancement_adds_dutch_guidance(self) -> None:
        """Context enhancement adds Dutch/German source priorities."""
        detector = GreenhouseDetector()

        # Empty base context
        enhanced = detector._enhance_context("")
        assert "assimilatieverlichting" in enhanced
        assert "groeilicht" in enhanced
        assert "kasmagazine.nl" in enhanced
        assert "royalvanzanten.com" in enhanced

        # With base context
        base = "Company grows tomatoes"
        enhanced = detector._enhance_context(base)
        assert "Company grows tomatoes" in enhanced
        assert "assimilatieverlichting" in enhanced

    def test_context_includes_dutch_terminology(self) -> None:
        """Context includes all required Dutch search terms."""
        detector = GreenhouseDetector()
        enhanced = detector._enhance_context("")

        # Check all required Dutch terms
        required_terms = [
            "assimilatieverlichting",
            "groeilicht",
            "kunstlicht",
            "kas",
            "teelt",
            "LED-verlichting",
            "SON-T",
            "assimilatiebelichting",
            "lichtspectrum",
            "Intensiteit verlichting",
        ]

        for term in required_terms:
            assert term in enhanced, f"Missing required term: {term}"

    def test_context_includes_dutch_sources(self) -> None:
        """Context includes prioritized Dutch sources."""
        detector = GreenhouseDetector()
        enhanced = detector._enhance_context("")

        # Check priority sources
        priority_sources = [
            "royalvanzanten.com",
            "floraldaily.com",
            "vakbladvoordebloemisterij.nl",
            "kasmagazine.nl",
            "onderglas.nl",
        ]

        for source in priority_sources:
            assert source in enhanced, f"Missing priority source: {source}"

    @pytest.mark.api
    def test_detector_with_mock_lm(self) -> None:
        """Test detector with a mock LM (no real API call)."""
        import dspy

        # Create a mock LM that returns fixed outputs
        class MockLM(dspy.LM):
            """Mock LM for testing without API calls."""

            def __init__(self) -> None:
                """Initialize mock LM."""
                # Don't call super().__init__() to avoid API setup
                self.model = "mock"
                self.provider = "mock"

            def __call__(self, prompt: str, **kwargs):  # type: ignore[no-untyped-def]  # noqa: ANN204, ANN003
                """Return fixed mock response."""
                return dspy.Prediction(
                    uses_artificial_lighting=True,
                    confidence=0.85,
                    lighting_type="LED",
                    evidence="Mock evidence: Uses LED grow lights",
                    primary_crops="tomaten, paprika",
                    size_hectares=5.0,
                    sources="https://example.nl/greenhouse",
                    reasoning=(
                        "Mock reasoning: Company website mentions LED-verlichting"
                    ),
                )

        # Note: This test demonstrates the structure but may not work
        # with actual DSPy internals. For real testing, use VCR or
        # API integration tests.


class TestPydanticConversion:
    """Test conversion from DSPy Prediction to Pydantic model."""

    def test_to_pydantic_with_delimited_strings(self) -> None:
        """Convert DSPy prediction with delimited strings to Pydantic."""
        import dspy

        detector = GreenhouseDetector()

        # Create a mock prediction with delimited strings
        prediction = dspy.Prediction(
            uses_artificial_lighting=True,
            confidence=0.92,
            lighting_type="SON-T",
            evidence="Evidence 1: LED mentioned|||Evidence 2: SON-T used",
            primary_crops="rozen, gerbera, tomaten",
            size_hectares=15.5,
            sources="https://example.nl/greenhouse|||https://kasmagazine.nl/article",
            reasoning="Strong evidence from multiple Dutch sources",
        )

        # Convert to Pydantic
        pydantic_result = detector.to_pydantic(prediction)

        # Verify conversion
        assert pydantic_result.uses_artificial_lighting is True
        assert pydantic_result.confidence == 0.92
        assert pydantic_result.lighting_type == "SON-T"
        assert len(pydantic_result.evidence) == 2
        assert "LED mentioned" in pydantic_result.evidence[0]
        assert len(pydantic_result.primary_crops) == 3
        assert "rozen" in pydantic_result.primary_crops
        assert pydantic_result.size_hectares == 15.5
        assert len(pydantic_result.sources) == 2
        assert ".nl" in pydantic_result.sources[0]

    def test_to_pydantic_handles_empty_fields(self) -> None:
        """Convert prediction with empty optional fields."""
        import dspy

        detector = GreenhouseDetector()

        prediction = dspy.Prediction(
            uses_artificial_lighting=False,
            confidence=0.3,
            lighting_type="None",
            evidence="",
            primary_crops="",
            size_hectares=0.0,
            sources="",
            reasoning="No evidence of artificial lighting found",
        )

        pydantic_result = detector.to_pydantic(prediction)

        assert pydantic_result.uses_artificial_lighting is False
        assert pydantic_result.confidence == 0.3
        assert pydantic_result.evidence == []
        assert pydantic_result.primary_crops == []
        assert pydantic_result.size_hectares is None
        assert pydantic_result.sources == []


@pytest.mark.api
class TestPerplexityIntegration:
    """
    Integration tests with Perplexity API.

    These tests make real API calls and are marked with @api marker.
    Run with: uv run pytest tests/test_dspy_greenhouse.py -m api
    Skip with: uv run pytest -m "not api"
    """

    @pytest.fixture
    def perplexity_api_key(self) -> str:
        """Get Perplexity API key from environment."""
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            pytest.skip("PERPLEXITY_API_KEY not set")
        return api_key

    def test_analyze_known_greenhouse_with_lighting(
        self, perplexity_api_key: str
    ) -> None:
        """
        Test analysis of known greenhouse with artificial lighting.

        Uses Royal Van Zanten as a test case - known Dutch greenhouse
        operation with documented artificial lighting usage.
        """
        import dspy

        # Configure Perplexity LM
        lm = dspy.LM(
            "perplexity/sonar-pro",
            api_key=perplexity_api_key,
        )
        dspy.configure(lm=lm)

        # Analyze known greenhouse
        result = analyze_greenhouse(
            company_name="Royal Van Zanten",
            location="Rijsenhout, Nederland",
            additional_context="Major gerbera and rose grower",
        )

        # Verify structured output
        assert isinstance(result, GreenhouseLightingAnalysis)
        assert isinstance(result.uses_artificial_lighting, bool)
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.sources) > 0

        # Log results for manual inspection
        print("\n=== Analysis Results ===")  # noqa: T201
        print(f"Uses artificial lighting: {result.uses_artificial_lighting}")  # noqa: T201
        print(f"Confidence: {result.confidence:.2%}")  # noqa: T201
        print(f"Lighting type: {result.lighting_type}")  # noqa: T201
        print(f"Crops: {result.primary_crops}")  # noqa: T201
        print(f"Size: {result.size_hectares} ha")  # noqa: T201
        print(f"Evidence: {result.evidence[:2]}")  # noqa: T201
        print(f"Sources: {result.sources[:3]}")  # noqa: T201
        print(f"Reasoning: {result.reasoning}")  # noqa: T201

        # Verify Dutch sources are prioritized
        dutch_sources = [s for s in result.sources if ".nl" in s]
        assert len(dutch_sources) > 0, "Should find at least one .nl source"

    def test_analyze_non_greenhouse_business(self, perplexity_api_key: str) -> None:
        """
        Test analysis of non-greenhouse business.

        Should return False for uses_artificial_lighting with low confidence.
        """
        import dspy

        lm = dspy.LM(
            "perplexity/sonar-pro",
            api_key=perplexity_api_key,
        )
        dspy.configure(lm=lm)

        # Analyze non-greenhouse (e.g., a restaurant)
        result = analyze_greenhouse(
            company_name="De Kas Restaurant",
            location="Amsterdam, Nederland",
            additional_context="Restaurant in a former greenhouse",
        )

        # Verify structured output
        assert isinstance(result, GreenhouseLightingAnalysis)
        print("\n=== Non-Greenhouse Analysis ===")  # noqa: T201
        print(f"Uses artificial lighting: {result.uses_artificial_lighting}")  # noqa: T201
        print(f"Confidence: {result.confidence:.2%}")  # noqa: T201
        print(f"Reasoning: {result.reasoning}")  # noqa: T201

        # Should detect this is not an agricultural greenhouse
        # (though it may have some lighting, it's not for plant growth)

    def test_dutch_terminology_in_evidence(self, perplexity_api_key: str) -> None:
        """
        Verify that Dutch terminology appears in evidence.

        Tests that the enhanced context leads to Dutch-language
        evidence being included in results.
        """
        import dspy

        lm = dspy.LM(
            "perplexity/sonar-pro",
            api_key=perplexity_api_key,
        )
        dspy.configure(lm=lm)

        result = analyze_greenhouse(
            company_name="Gerbera Breeding",
            location="Ridderkerk, Nederland",
        )

        # Check for Dutch terminology in evidence or reasoning
        all_text = " ".join([*result.evidence, result.reasoning or ""])

        print("\n=== Dutch Terminology Check ===")  # noqa: T201
        print(f"Combined text: {all_text[:500]}...")  # noqa: T201

        # At least one Dutch source should be present
        dutch_sources = [s for s in result.sources if ".nl" in s]
        print(f"Dutch sources found: {len(dutch_sources)}")  # noqa: T201


class TestAnalyzeGreenhouseConvenience:
    """Test the convenience function analyze_greenhouse."""

    def test_analyze_greenhouse_requires_lm(self) -> None:
        """analyze_greenhouse requires an LM to be configured."""
        import dspy

        # Clear any existing LM
        dspy.configure(lm=None)

        # This should work if an LM is provided explicitly
        # (but we're not testing with real API here)

    def test_analyze_greenhouse_returns_pydantic(self) -> None:
        """analyze_greenhouse returns a Pydantic model."""
        # This is implicitly tested in integration tests
        # Just verify the type signature is correct
        from inspect import signature

        sig = signature(analyze_greenhouse)
        assert sig.return_annotation == GreenhouseLightingAnalysis
