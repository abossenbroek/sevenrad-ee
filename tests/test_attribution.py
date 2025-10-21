"""
Tests for attribution module.

Tests greenhouse attribution logic including confidence scoring,
Perplexity analysis, and attribution results with log-odds based Bayesian scoring.
"""

import math

import pytest
from sevenrad_ee.top_polluters.attribution import (
    EVIDENCE_TEXT_SCORE,
    LIGHTING_KEYWORDS,
    TIER_1_CROPS,
    TIER_1_SCORE,
    TIER_2_CROPS,
    TIER_2_SCORE,
    TIER_3_CROPS,
    TIER_3_SCORE,
    PerplexityAnalysis,
    _get_distance_adjustment,
    _get_evidence_based_likelihood,
    _get_size_adjustment,
    _get_type_adjustment,
    _logit,
    _sigmoid,
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


class TestLogOddsHelpers:
    """Test log-odds conversion helper functions."""

    def test_logit_converts_probability_to_log_odds(self) -> None:
        """Logit function converts probability to log-odds."""
        # p=0.5 → logit=0
        assert _logit(0.5) == pytest.approx(0.0, abs=0.01)

        # p=0.75 → logit≈1.099
        assert _logit(0.75) == pytest.approx(1.099, abs=0.01)

        # p=0.95 → logit≈2.944
        assert _logit(0.95) == pytest.approx(2.944, abs=0.01)

    def test_logit_handles_edge_cases(self) -> None:
        """Logit clamps extreme probabilities to prevent inf."""
        # Very close to 0 - should not produce inf
        result_low = _logit(1e-10)
        assert math.isfinite(result_low)
        assert result_low < 0

        # Very close to 1 - should not produce inf
        result_high = _logit(1 - 1e-10)
        assert math.isfinite(result_high)
        assert result_high > 0

    def test_sigmoid_converts_log_odds_to_probability(self) -> None:
        """Sigmoid function converts log-odds to probability."""
        # logit=0 → p=0.5
        assert _sigmoid(0.0) == pytest.approx(0.5, abs=0.01)

        # logit=1.099 → p≈0.75
        assert _sigmoid(1.099) == pytest.approx(0.75, abs=0.01)

        # logit=2.944 → p≈0.95
        assert _sigmoid(2.944) == pytest.approx(0.95, abs=0.01)

    def test_logit_sigmoid_roundtrip(self) -> None:
        """Logit and sigmoid are inverse functions."""
        test_probs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        for p in test_probs:
            assert _sigmoid(_logit(p)) == pytest.approx(p, abs=0.001)


class TestAdjustmentFunctions:
    """Test log-odds adjustment functions."""

    def test_distance_adjustment_linear_interpolation(self) -> None:
        """Distance adjustment varies linearly from +1.5 to -1.5."""
        # At 0m: +1.5
        assert _get_distance_adjustment(0, max_distance=1000) == pytest.approx(1.5)

        # At 500m (midpoint): 0.0
        assert _get_distance_adjustment(500, max_distance=1000) == pytest.approx(0.0)

        # At 1000m: -1.5
        assert _get_distance_adjustment(1000, max_distance=1000) == pytest.approx(-1.5)

        # Beyond max: -1.5 (capped)
        assert _get_distance_adjustment(1500, max_distance=1000) == pytest.approx(-1.5)

    def test_size_adjustment_log_scale(self) -> None:
        """Size adjustment uses log scale."""
        # No size info: 0.0
        assert _get_size_adjustment(None) == 0.0
        assert _get_size_adjustment(0.0) == 0.0
        assert _get_size_adjustment(-1.0) == 0.0

        # 1 hectare: 0.5 * ln(1) = 0.0
        assert _get_size_adjustment(1.0) == pytest.approx(0.0)

        # 2.718 hectares (e): 0.5 * ln(e) = 0.5
        assert _get_size_adjustment(math.e) == pytest.approx(0.5, abs=0.01)

        # Larger sizes get higher adjustments (log scale)
        assert _get_size_adjustment(10.0) > _get_size_adjustment(5.0)

    def test_type_adjustment_greenhouse_bonus(self) -> None:
        """Greenhouse types get +1.1 adjustment."""
        # Greenhouse types
        assert _get_type_adjustment(["greenhouse"]) == 1.1
        assert _get_type_adjustment(["nursery"]) == 1.1
        assert _get_type_adjustment(["horticulture"]) == 1.1
        assert _get_type_adjustment(["greenhouse", "other"]) == 1.1

        # Non-greenhouse types
        assert _get_type_adjustment(["restaurant"]) == 0.0
        assert _get_type_adjustment(["store"]) == 0.0
        assert _get_type_adjustment([]) == 0.0


class TestEvidenceBasedLikelihood:
    """Test multi-signal evidence-based likelihood function."""

    def test_tier1_crops_highest_priority(self) -> None:
        """Tier 1 crops (rose, tomato, etc.) override other signals."""
        analysis = PerplexityAnalysis(
            grow_light_likelihood=0.1,  # AI says low
            primary_crops=["rose"],  # But it's a tier 1 crop
        )

        likelihood, reason = _get_evidence_based_likelihood(analysis)

        assert likelihood == TIER_1_SCORE  # 0.95
        assert reason == "tier1_crop"

    def test_tier2_crops_high_priority(self) -> None:
        """Tier 2 crops (gerbera, lettuce, etc.) have high likelihood."""
        analysis = PerplexityAnalysis(
            grow_light_likelihood=0.2,
            primary_crops=["gerbera"],
        )

        likelihood, reason = _get_evidence_based_likelihood(analysis)

        assert likelihood == TIER_2_SCORE  # 0.85
        assert reason == "tier2_crop"

    def test_tier3_crops_moderate_priority(self) -> None:
        """Tier 3 crops have moderate likelihood."""
        analysis = PerplexityAnalysis(
            grow_light_likelihood=0.3,
            primary_crops=["alstroemeria"],
        )

        likelihood, reason = _get_evidence_based_likelihood(analysis)

        assert likelihood == TIER_3_SCORE  # 0.65
        assert reason == "tier3_crop"

    def test_evidence_keywords_override_ai(self) -> None:
        """Lighting keywords in evidence override low AI score."""
        analysis = PerplexityAnalysis(
            grow_light_likelihood=0.1,
            lighting_evidence="Uses LED grow lights for year-round production",
            primary_crops=[],
        )

        likelihood, reason = _get_evidence_based_likelihood(analysis)

        assert likelihood == EVIDENCE_TEXT_SCORE  # 0.85
        assert reason == "evidence_keyword"

    def test_ai_likelihood_fallback(self) -> None:
        """AI likelihood used when no other signals available."""
        analysis = PerplexityAnalysis(
            grow_light_likelihood=0.6,
            primary_crops=[],
        )

        likelihood, reason = _get_evidence_based_likelihood(analysis)

        assert likelihood == 0.6
        assert reason == "ai_likelihood"

    def test_max_signal_wins(self) -> None:
        """Highest signal value wins."""
        # Tier 1 crop (0.95) should beat evidence keywords (0.85)
        analysis = PerplexityAnalysis(
            grow_light_likelihood=0.1,
            lighting_evidence="LED lighting mentioned",
            primary_crops=["tomato"],  # Tier 1
        )

        likelihood, reason = _get_evidence_based_likelihood(analysis)

        assert likelihood == TIER_1_SCORE
        assert reason == "tier1_crop"

    def test_no_signals_returns_zero(self) -> None:
        """Returns 0.0 when no signals available."""
        analysis = PerplexityAnalysis()

        likelihood, reason = _get_evidence_based_likelihood(analysis)

        assert likelihood == 0.0
        assert reason == "no_signals"


class TestCalculateConfidence:
    """Test log-odds based confidence calculation."""

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

    def test_confidence_uses_log_odds_scaling(self) -> None:
        """Confidence calculation uses log-odds space."""
        analysis = PerplexityAnalysis(
            grow_light_likelihood=0.5,
            primary_crops=["rose"],  # Tier 1: base=0.95
        )

        confidence, factors = calculate_confidence(
            perplexity=analysis,
            distance_m=100,
            business_types=[],
            max_distance=1000,
        )

        # Check that log-odds keys exist
        assert "base_logit" in factors
        assert "final_logit" in factors
        assert "base_likelihood" in factors
        assert "evidence_reason" in factors

        # Base likelihood should be from tier1_crop
        assert factors["base_likelihood"] == TIER_1_SCORE
        assert factors["evidence_reason"] == "tier1_crop"

    def test_confidence_distance_affects_score(self) -> None:
        """Closer businesses get higher confidence."""
        analysis = PerplexityAnalysis(primary_crops=["rose"])

        # Close business
        conf_close, _ = calculate_confidence(
            perplexity=analysis,
            distance_m=50,
            business_types=[],
            max_distance=1000,
        )

        # Far business
        conf_far, _ = calculate_confidence(
            perplexity=analysis,
            distance_m=900,
            business_types=[],
            max_distance=1000,
        )

        # Closer should have higher confidence
        assert conf_close > conf_far

    def test_confidence_greenhouse_type_increases_score(self) -> None:
        """Greenhouse business types increase confidence."""
        analysis = PerplexityAnalysis(primary_crops=["rose"])

        # Greenhouse type
        conf_greenhouse, factors_greenhouse = calculate_confidence(
            perplexity=analysis,
            distance_m=100,
            business_types=["greenhouse"],
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
        assert factors_greenhouse["type_adj"] == 1.1
        assert factors_generic["type_adj"] == 0.0

    def test_confidence_large_size_increases_score(self) -> None:
        """Larger greenhouses get higher confidence."""
        # Large greenhouse
        analysis_large = PerplexityAnalysis(
            primary_crops=["rose"],
            size_hectares=10.0,
        )
        conf_large, factors_large = calculate_confidence(
            perplexity=analysis_large,
            distance_m=100,
            business_types=[],
            max_distance=1000,
        )

        # Small greenhouse
        analysis_small = PerplexityAnalysis(
            primary_crops=["rose"],
            size_hectares=2.0,
        )
        conf_small, factors_small = calculate_confidence(
            perplexity=analysis_small,
            distance_m=100,
            business_types=[],
            max_distance=1000,
        )

        assert conf_large > conf_small
        assert factors_large["size_adj"] > factors_small["size_adj"]

    def test_confidence_naturally_bounded(self) -> None:
        """Confidence never exceeds 1.0 without clamping (sigmoid does this)."""
        # Perfect scenario
        analysis = PerplexityAnalysis(
            grow_light_likelihood=1.0,
            primary_crops=["rose", "tomato"],  # Tier 1
            size_hectares=50.0,
        )

        confidence, _ = calculate_confidence(
            perplexity=analysis,
            distance_m=0,
            business_types=["greenhouse"],
            max_distance=1000,
        )

        # Sigmoid ensures this is bounded
        assert 0.0 <= confidence <= 1.0

    def test_confidence_factors_transparency(self) -> None:
        """Factors dict includes all scoring components for transparency."""
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

        # Check new log-odds based keys
        assert "base_likelihood" in factors
        assert "evidence_reason" in factors
        assert "ai_grow_light_likelihood" in factors
        assert "distance_m" in factors
        assert "distance_adj" in factors
        assert "size_hectares" in factors
        assert "size_adj" in factors
        assert "type_adj" in factors
        assert "base_logit" in factors
        assert "final_logit" in factors
        assert "final_confidence" in factors
