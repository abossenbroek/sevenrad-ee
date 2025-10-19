"""
Tests for attribution module.

Tests greenhouse attribution logic including confidence scoring,
Perplexity analysis, and attribution results.
"""

import pytest
from sevenrad_ee.top_polluters.attribution import (
    AGRICULTURE_TYPE_SCORE,
    ENERGY_CROP_MULTIPLIER,
    GREENHOUSE_TYPE_SCORE,
    LARGE_GREENHOUSE_THRESHOLD_HA,
    LARGE_OPERATION_MULTIPLIER,
    MAX_DISTANCE_SCORE_WEIGHT,
    MAX_LIKELIHOOD_SCORE_WEIGHT,
    PerplexityAnalysis,
    calculate_confidence,
)


class TestPerplexityAnalysis:
    """Test Perplexity analysis model."""

    def test_perplexity_analysis_minimal(self) -> None:
        """Create Perplexity analysis with minimal fields."""
        analysis = PerplexityAnalysis()

        assert analysis.grow_light_likelihood is None
        assert analysis.lighting_evidence is None
        assert analysis.primary_crops == []
        assert analysis.size_hectares is None
        assert analysis.instagram_handle is None
        assert analysis.sources == []

    def test_perplexity_analysis_complete(self) -> None:
        """Create Perplexity analysis with all fields populated."""
        analysis = PerplexityAnalysis(
            grow_light_likelihood=0.95,
            lighting_evidence="Uses LED grow lights year-round",
            primary_crops=["gerbera", "rose"],
            size_hectares=7.5,
            instagram_handle="summitgerbera",
            sources=[
                "https://example.com/greenhouse1",
                "https://example.com/greenhouse2",
            ],
        )

        assert analysis.grow_light_likelihood == 0.95
        assert analysis.lighting_evidence == "Uses LED grow lights year-round"
        assert analysis.primary_crops == ["gerbera", "rose"]
        assert analysis.size_hectares == 7.5
        assert analysis.instagram_handle == "summitgerbera"
        assert len(analysis.sources) == 2


class TestCalculateConfidence:
    """Test confidence calculation algorithm."""

    def test_confidence_no_perplexity_analysis(self) -> None:
        """Confidence is zero when Perplexity analysis is None."""
        confidence, factors = calculate_confidence(
            perplexity=None,
            distance_m=100,
            business_types=["greenhouse"],
            max_distance=1000,
        )

        assert confidence == 0.0
        assert "analysis_failed" in factors

    def test_confidence_distance_scoring(self) -> None:
        """Distance contributes to confidence score (closer is better)."""
        analysis = PerplexityAnalysis(grow_light_likelihood=0.5)

        # Close business
        conf_close, factors_close = calculate_confidence(
            perplexity=analysis,
            distance_m=50,
            business_types=[],
            max_distance=1000,
        )

        # Far business
        conf_far, factors_far = calculate_confidence(
            perplexity=analysis,
            distance_m=900,
            business_types=[],
            max_distance=1000,
        )

        # Closer business should have higher confidence
        assert conf_close > conf_far
        assert factors_close["distance"] > factors_far["distance"]

    def test_confidence_greenhouse_type_bonus(self) -> None:
        """Greenhouse business types get higher score."""
        analysis = PerplexityAnalysis(grow_light_likelihood=0.5)

        # Greenhouse type
        conf_greenhouse, factors_greenhouse = calculate_confidence(
            perplexity=analysis,
            distance_m=100,
            business_types=["greenhouse", "nursery"],
            max_distance=1000,
        )

        # Generic type
        conf_generic, factors_generic = calculate_confidence(
            perplexity=analysis,
            distance_m=100,
            business_types=["restaurant"],
            max_distance=1000,
        )

        assert conf_greenhouse > conf_generic
        assert factors_greenhouse["business_type"] == GREENHOUSE_TYPE_SCORE
        assert factors_generic["business_type"] == 0.0

    def test_confidence_agriculture_type_bonus(self) -> None:
        """Agricultural business types get moderate score."""
        analysis = PerplexityAnalysis(grow_light_likelihood=0.5)

        # Agriculture type (need to match one of: "agricultural", "farming", "agri")
        conf_agri, factors_agri = calculate_confidence(
            perplexity=analysis,
            distance_m=100,
            business_types=["farming"],
            max_distance=1000,
        )

        # Generic type
        conf_generic, factors_generic = calculate_confidence(
            perplexity=analysis,
            distance_m=100,
            business_types=["restaurant"],
            max_distance=1000,
        )

        assert conf_agri > conf_generic
        assert factors_agri["business_type"] == AGRICULTURE_TYPE_SCORE
        assert factors_generic["business_type"] == 0.0

    def test_confidence_grow_light_likelihood_score(self) -> None:
        """Grow light likelihood contributes to confidence."""
        # High likelihood
        analysis_high = PerplexityAnalysis(grow_light_likelihood=0.9)
        conf_high, _ = calculate_confidence(
            perplexity=analysis_high,
            distance_m=100,
            business_types=[],
            max_distance=1000,
        )

        # Low likelihood
        analysis_low = PerplexityAnalysis(grow_light_likelihood=0.1)
        conf_low, _ = calculate_confidence(
            perplexity=analysis_low,
            distance_m=100,
            business_types=[],
            max_distance=1000,
        )

        # No likelihood
        analysis_none = PerplexityAnalysis(grow_light_likelihood=None)
        conf_none, factors_none = calculate_confidence(
            perplexity=analysis_none,
            distance_m=100,
            business_types=[],
            max_distance=1000,
        )

        assert conf_high > conf_low > conf_none
        assert "no_lighting_info" in factors_none

    def test_confidence_energy_crop_multiplier(self) -> None:
        """Energy-intensive crops increase confidence."""
        # Energy crop
        analysis_energy = PerplexityAnalysis(
            grow_light_likelihood=0.5, primary_crops=["gerbera", "rose"]
        )
        conf_energy, factors_energy = calculate_confidence(
            perplexity=analysis_energy,
            distance_m=100,
            business_types=[],
            max_distance=1000,
        )

        # Non-energy crop
        analysis_regular = PerplexityAnalysis(
            grow_light_likelihood=0.5, primary_crops=["lettuce"]
        )
        conf_regular, factors_regular = calculate_confidence(
            perplexity=analysis_regular,
            distance_m=100,
            business_types=[],
            max_distance=1000,
        )

        assert conf_energy > conf_regular
        assert "energy_crops" in factors_energy
        assert factors_energy["energy_crops"] == pytest.approx(
            ENERGY_CROP_MULTIPLIER - 1.0
        )

    def test_confidence_large_operation_multiplier(self) -> None:
        """Large operations (>5 ha) increase confidence."""
        # Large operation
        analysis_large = PerplexityAnalysis(
            grow_light_likelihood=0.5, size_hectares=10.0
        )
        conf_large, factors_large = calculate_confidence(
            perplexity=analysis_large,
            distance_m=100,
            business_types=[],
            max_distance=1000,
        )

        # Small operation
        analysis_small = PerplexityAnalysis(
            grow_light_likelihood=0.5, size_hectares=2.0
        )
        conf_small, factors_small = calculate_confidence(
            perplexity=analysis_small,
            distance_m=100,
            business_types=[],
            max_distance=1000,
        )

        assert conf_large > conf_small
        assert "large_operation" in factors_large
        assert factors_large["large_operation"] == pytest.approx(
            LARGE_OPERATION_MULTIPLIER - 1.0
        )

    def test_confidence_combined_factors(self) -> None:
        """Test confidence with multiple factors combined."""
        analysis = PerplexityAnalysis(
            grow_light_likelihood=0.9,
            primary_crops=["rose"],  # Energy crop
            size_hectares=7.5,  # Large operation
        )

        confidence, factors = calculate_confidence(
            perplexity=analysis,
            distance_m=50,
            business_types=["greenhouse"],
            max_distance=1000,
        )

        # Should have high confidence with all positive factors
        assert confidence > 0.5
        assert "distance" in factors
        assert "business_type" in factors
        assert "grow_light_likelihood" in factors
        # These only appear when conditions are met
        if "rose" in ["tomato", "pepper", "cucumber", "flower", "rose"]:
            assert "energy_crops" in factors
        if LARGE_GREENHOUSE_THRESHOLD_HA < 7.5:
            assert "large_operation" in factors

    def test_confidence_capped_at_1_0(self) -> None:
        """Confidence score never exceeds 1.0."""
        # Create perfect scenario
        analysis = PerplexityAnalysis(
            grow_light_likelihood=1.0,
            primary_crops=["gerbera", "rose", "tomato"],
            size_hectares=50.0,
        )

        confidence, _ = calculate_confidence(
            perplexity=analysis,
            distance_m=0,
            business_types=["greenhouse", "nursery"],
            max_distance=1000,
        )

        assert confidence <= 1.0

    def test_confidence_factors_dict(self) -> None:
        """Confidence factors dictionary contains expected keys."""
        analysis = PerplexityAnalysis(
            grow_light_likelihood=0.8,
            primary_crops=["pepper"],
            size_hectares=6.0,
        )

        _, factors = calculate_confidence(
            perplexity=analysis,
            distance_m=100,
            business_types=["greenhouse"],
            max_distance=1000,
        )

        # Check core expected keys
        assert "distance" in factors
        assert "business_type" in factors
        assert "grow_light_likelihood" in factors
        assert "grow_light_score" in factors
        assert "total" in factors

        # Multipliers only appear when conditions are met
        # pepper is energy crop, 6.0 > 5.0 so should have both
        assert "energy_crops" in factors
        assert "large_operation" in factors
