"""
Tests for Phase 3 Dutch-aware hierarchical F1 metric.

Tests the enhanced evaluation metric system with:
- Dutch terminology scoring
- Evidence quality tier weighting
- Hierarchical classification scoring
- Feedback generation for GEPA reflection
- Backward compatibility wrapper
"""

import pytest

try:
    import dspy
except ImportError:
    pytest.skip("dspy-ai not installed", allow_module_level=True)

from sevenrad_ee.ai.dspy_evaluation import (
    DUTCH_TERMS,
    SOURCE_TIER_SCORES,
    dutch_aware_f1_score_only,
    dutch_aware_hierarchical_f1,
)
from sevenrad_ee.ai.dspy_greenhouse import EvidenceSource


class MockPrediction:
    """Mock prediction object for testing."""

    def __init__(
        self,
        is_greenhouse: str = "YES",
        uses_growlight: str = "YES",
        dutch_terms_found: list[str] | str | None = None,
        evidence_sources: list[dict | EvidenceSource] | str | None = None,
        confidence: float = 0.9,
    ) -> None:
        """Initialize mock prediction."""
        self.is_greenhouse = is_greenhouse
        self.uses_growlight = uses_growlight
        self.dutch_terms_found = dutch_terms_found or []
        self.evidence_sources = evidence_sources or []
        self.confidence = confidence


class TestDutchTerminologyScoring:
    """Test Dutch terminology detection and scoring."""

    def test_no_dutch_terms_found(self) -> None:
        """Prediction with no Dutch terms gets low Dutch score."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
            location_name="Test Company",
        )

        prediction = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="YES",
            dutch_terms_found=[],  # No Dutch terms
            evidence_sources=[
                {
                    "url": "https://example.com",
                    "quote": "test",
                    "tier": "company_website",
                }
            ],
        )

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        # Should still get classification score (70%) + evidence score (15%)
        # But Dutch score (15%) will be 0
        assert "NO Dutch terminology found" in feedback
        assert score < 1.0  # Not perfect due to missing Dutch terms

    def test_many_dutch_terms_found(self) -> None:
        """Prediction with many Dutch terms gets positive feedback."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
            location_name="Test Kwekerij",
        )

        prediction = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="YES",
            dutch_terms_found=[
                "assimilatiebelichting",
                "kwekerij",
                "led-belichting",
                "glastuinbouw",
            ],
            evidence_sources=[
                {
                    "url": "https://example.com",
                    "quote": "test",
                    "tier": "company_website",
                }
            ],
        )

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        assert "GOOD: Found" in feedback
        assert "Dutch terms" in feedback
        assert score > 0.8  # High score with Dutch terms

    def test_dutch_terms_as_comma_separated_string(self) -> None:
        """Handle dutch_terms_found as comma-separated string."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
        )

        prediction = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="YES",
            dutch_terms_found="kwekerij,assimilatiebelichting,led-belichting",
            evidence_sources=[
                {
                    "url": "https://example.com",
                    "quote": "test",
                    "tier": "company_website",
                }
            ],
        )

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        assert "GOOD: Found" in feedback
        assert score > 0.8

    def test_dutch_terms_validation_against_known_set(self) -> None:
        """Only terms in DUTCH_TERMS set should be counted."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
        )

        prediction = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="YES",
            dutch_terms_found=[
                "kwekerij",  # Valid
                "random_term",  # Invalid
                "assimilatiebelichting",  # Valid
                "another_random",  # Invalid
            ],
            evidence_sources=[
                {
                    "url": "https://example.com",
                    "quote": "test",
                    "tier": "company_website",
                }
            ],
        )

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        # Should only count the 2 valid terms
        assert "Found 2 Dutch terms" in feedback


class TestEvidenceQualityScoring:
    """Test evidence source tier-based quality scoring."""

    def test_no_evidence_sources(self) -> None:
        """Prediction with no evidence gets low evidence score."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
        )

        prediction = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="YES",
            dutch_terms_found=["kwekerij"],
            evidence_sources=[],  # No evidence
        )

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        assert "NO evidence sources provided" in feedback
        assert score < 1.0

    def test_tier2_evidence_gets_high_score(self) -> None:
        """Tier-2 evidence (company_website) gets highest quality score."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
        )

        prediction = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="YES",
            dutch_terms_found=["kwekerij"],
            evidence_sources=[
                {
                    "url": "https://company.com",
                    "quote": "LED lighting",
                    "tier": "company_website",
                },
                {
                    "url": "https://signify.com",
                    "quote": "case study",
                    "tier": "supplier_case_study",
                },
            ],
        )

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        assert "tier-2 sources found" in feedback
        assert score > 0.8

    def test_general_web_gets_lower_score(self) -> None:
        """General web sources get lower quality score than tier-2."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
        )

        prediction_tier2 = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="YES",
            dutch_terms_found=["kwekerij"],
            evidence_sources=[
                {
                    "url": "https://company.com",
                    "quote": "test",
                    "tier": "company_website",
                }
            ],
        )

        prediction_general = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="YES",
            dutch_terms_found=["kwekerij"],
            evidence_sources=[
                {"url": "https://random.com", "quote": "test", "tier": "general_web"}
            ],
        )

        score_tier2, _ = dutch_aware_hierarchical_f1(example, prediction_tier2)
        score_general, feedback_general = dutch_aware_hierarchical_f1(
            example, prediction_general
        )

        assert score_tier2 > score_general
        assert "LOW-QUALITY sources" in feedback_general

    def test_evidence_sources_as_pydantic_objects(self) -> None:
        """Handle evidence_sources as Pydantic EvidenceSource objects."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
        )

        prediction = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="YES",
            dutch_terms_found=["kwekerij"],
            evidence_sources=[
                EvidenceSource(
                    url="https://company.com",
                    quote="LED lighting",
                    tier="company_website",
                )
            ],
        )

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        assert "tier-2 sources found" in feedback
        assert score > 0.8

    def test_source_tier_weights(self) -> None:
        """Verify SOURCE_TIER_SCORES are used correctly."""
        assert SOURCE_TIER_SCORES["company_website"] == 2.0
        assert SOURCE_TIER_SCORES["supplier_case_study"] == 2.0
        assert SOURCE_TIER_SCORES["job_posting"] == 2.0
        assert SOURCE_TIER_SCORES["trade_media_nl"] == 1.0
        assert SOURCE_TIER_SCORES["general_web"] == 0.5


class TestHierarchicalClassificationScoring:
    """Test hierarchical classification (is_greenhouse + uses_growlight) scoring."""

    def test_perfect_prediction(self) -> None:
        """Perfect classification with good evidence gets high score."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
        )

        prediction = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="YES",
            dutch_terms_found=["kwekerij", "assimilatiebelichting"],
            evidence_sources=[
                {
                    "url": "https://company.com",
                    "quote": "test",
                    "tier": "company_website",
                }
            ],
        )

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        # Perfect classification (0.70) + some Dutch terms (0.01875) + tier-2 evidence (0.15)
        assert score > 0.85  # High score, but not 1.0 due to limited Dutch terms
        # No misclassification errors in feedback
        assert "MISCLASSIFIED" not in feedback
        assert "GOOD: 1 tier-2 sources" in feedback

    def test_is_greenhouse_misclassification(self) -> None:
        """Misclassified is_greenhouse gets low score and feedback."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
        )

        prediction = MockPrediction(
            is_greenhouse="NO",  # Wrong!
            uses_growlight="UNKNOWN",
            dutch_terms_found=["kwekerij"],
            evidence_sources=[
                {"url": "https://example.com", "quote": "test", "tier": "general_web"}
            ],
        )

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        assert "MISCLASSIFIED is_greenhouse" in feedback
        assert "predicted NO, expected YES" in feedback
        assert score < 0.5  # Classification score is 0, only Dutch+evidence remain

    def test_uses_growlight_misclassification(self) -> None:
        """Misclassified uses_growlight gets feedback."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
        )

        prediction = MockPrediction(
            is_greenhouse="YES",  # Correct
            uses_growlight="NO",  # Wrong!
            dutch_terms_found=["kwekerij"],
            evidence_sources=[
                {
                    "url": "https://company.com",
                    "quote": "test",
                    "tier": "company_website",
                }
            ],
        )

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        assert "MISCLASSIFIED uses_growlight" in feedback
        assert "predicted NO, expected YES" in feedback
        assert 0.3 < score < 0.7  # Gets partial credit for is_greenhouse + evidence

    def test_hierarchical_gating_enforcement(self) -> None:
        """Not greenhouse → uses_growlight must be UNKNOWN."""
        example = dspy.Example(
            is_greenhouse="NO",
            uses_growlight="UNKNOWN",
        )

        prediction = MockPrediction(
            is_greenhouse="NO",
            uses_growlight="YES",  # Logic error!
            dutch_terms_found=[],
            evidence_sources=[],
        )

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        assert "LOGIC ERROR" in feedback
        assert "should be UNKNOWN" in feedback

    def test_unknown_growlight_classification(self) -> None:
        """UNKNOWN growlight classification is valid when uncertain."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="UNKNOWN",
        )

        prediction = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="UNKNOWN",
            dutch_terms_found=["kwekerij"],
            evidence_sources=[
                {
                    "url": "https://company.com",
                    "quote": "test",
                    "tier": "company_website",
                }
            ],
        )

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        # Perfect classification + tier-2 evidence, limited Dutch terms
        assert score > 0.85  # High score but not 1.0 due to limited Dutch terms
        # No misclassification errors in feedback
        assert "MISCLASSIFIED" not in feedback
        assert "GOOD: 1 tier-2 sources" in feedback


class TestFeedbackGeneration:
    """Test feedback message generation for GEPA reflection."""

    def test_feedback_suggests_dutch_search_terms(self) -> None:
        """Feedback suggests Dutch terminology searches when missing terms."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
            location_name="Test Kwekerij",
        )

        prediction = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="NO",  # Wrong
            dutch_terms_found=[],  # No Dutch terms
            evidence_sources=[
                {"url": "https://example.com", "quote": "test", "tier": "general_web"}
            ],
        )

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        assert "assimilatiebelichting" in feedback.lower()
        assert "search" in feedback.lower()

    def test_feedback_warns_about_high_confidence_without_tier2(self) -> None:
        """Warn when confidence is high but no tier-2 evidence."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
        )

        prediction = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="YES",
            dutch_terms_found=["kwekerij"],
            evidence_sources=[
                {"url": "https://random.com", "quote": "test", "tier": "general_web"}
            ],
            confidence=0.95,  # High confidence
        )

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        assert "WARNING" in feedback
        assert "tier-2 evidence" in feedback

    def test_feedback_identifies_missed_company_name_terms(self) -> None:
        """Feedback identifies Dutch terms in company name that were missed."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
            location_name="Assimilatiebelichting Specialists BV",
        )

        prediction = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="NO",  # Wrong
            dutch_terms_found=[],  # Missed "assimilatie" in company name!
            evidence_sources=[
                {
                    "url": "https://company.com",
                    "quote": "test",
                    "tier": "company_website",
                }
            ],
        )

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        assert "MISSED 'assimilatie' terminology" in feedback
        assert "company name" in feedback

    def test_feedback_format_is_pipe_separated(self) -> None:
        """Feedback parts are separated by ' | ' for readability."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
        )

        prediction = MockPrediction(
            is_greenhouse="NO",  # Wrong
            uses_growlight="UNKNOWN",
            dutch_terms_found=[],
            evidence_sources=[],
        )

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        assert " | " in feedback  # Multiple feedback parts separated


class TestBackwardCompatibility:
    """Test backward compatibility wrapper for non-GEPA optimizers."""

    def test_score_only_wrapper_returns_float(self) -> None:
        """dutch_aware_f1_score_only returns only float, not tuple."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
        )

        prediction = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="YES",
            dutch_terms_found=["kwekerij"],
            evidence_sources=[
                {
                    "url": "https://company.com",
                    "quote": "test",
                    "tier": "company_website",
                }
            ],
        )

        score = dutch_aware_f1_score_only(example, prediction)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_only_matches_full_metric_score(self) -> None:
        """Score from wrapper should match score from full metric."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="NO",
        )

        prediction = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="NO",
            dutch_terms_found=["kwekerij", "onbelichte teelt"],
            evidence_sources=[
                {
                    "url": "https://company.com",
                    "quote": "test",
                    "tier": "company_website",
                }
            ],
        )

        score_only = dutch_aware_f1_score_only(example, prediction)
        score_full, _ = dutch_aware_hierarchical_f1(example, prediction)

        assert score_only == score_full


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_is_greenhouse_field(self) -> None:
        """Missing is_greenhouse field returns error feedback."""

        class BadPrediction:
            """Prediction missing is_greenhouse."""

            def __init__(self) -> None:
                """Initialize with uses_growlight only."""
                self.uses_growlight = "YES"

        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
        )

        prediction = BadPrediction()

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        assert score == 0.0
        assert "FATAL" in feedback
        assert "missing is_greenhouse" in feedback

    def test_missing_uses_growlight_field(self) -> None:
        """Missing uses_growlight field returns error feedback."""

        class BadPrediction:
            """Prediction missing uses_growlight."""

            def __init__(self) -> None:
                """Initialize with is_greenhouse only."""
                self.is_greenhouse = "YES"

        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
        )

        prediction = BadPrediction()

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        assert score == 0.0
        assert "FATAL" in feedback
        assert "missing uses_growlight" in feedback

    def test_malformed_json_evidence_sources(self) -> None:
        """Malformed JSON in evidence_sources string is handled gracefully."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
        )

        prediction = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="YES",
            dutch_terms_found=["kwekerij"],
            evidence_sources="not valid json {{{",  # type: ignore[arg-type]
        )

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        # Should treat as no evidence
        assert "NO evidence sources" in feedback
        assert score < 1.0

    def test_empty_strings_normalized_to_uppercase(self) -> None:
        """Empty strings and whitespace are normalized correctly."""
        example = dspy.Example(
            is_greenhouse="yes",  # Lowercase
            uses_growlight=" YES ",  # Whitespace
        )

        prediction = MockPrediction(
            is_greenhouse=" Yes ",  # Mixed case + whitespace
            uses_growlight="yes",  # Lowercase
            dutch_terms_found=["kwekerij"],
            evidence_sources=[
                {
                    "url": "https://company.com",
                    "quote": "test",
                    "tier": "company_website",
                }
            ],
        )

        score, feedback = dutch_aware_hierarchical_f1(example, prediction)

        # Should match due to normalization (perfect classification + tier-2 evidence)
        assert score > 0.85  # High score but not 1.0 due to limited Dutch terms
        # No misclassification errors in feedback - normalization worked
        assert "MISCLASSIFIED" not in feedback
        assert "GOOD: 1 tier-2 sources" in feedback


class TestScoringWeights:
    """Test that scoring weights are applied correctly."""

    def test_classification_weight_is_70_percent(self) -> None:
        """Hierarchical classification contributes 70% to total score."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
        )

        # Perfect classification, no Dutch terms, no evidence
        prediction = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="YES",
            dutch_terms_found=[],
            evidence_sources=[],
        )

        score, _ = dutch_aware_hierarchical_f1(example, prediction)

        # Should be 0.70 (perfect classification) + 0.0 + 0.0
        assert abs(score - 0.70) < 0.01

    def test_dutch_weight_is_15_percent(self) -> None:
        """Dutch terminology contributes 15% to total score."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
        )

        # Perfect classification, all Dutch terms, no evidence
        prediction = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="YES",
            dutch_terms_found=list(DUTCH_TERMS),  # All Dutch terms
            evidence_sources=[],
        )

        score, _ = dutch_aware_hierarchical_f1(example, prediction)

        # Should be 0.70 (classification) + 0.15 (all Dutch terms) + 0.0 (no evidence)
        assert abs(score - 0.85) < 0.01

    def test_evidence_weight_is_15_percent(self) -> None:
        """Evidence quality contributes 15% to total score."""
        example = dspy.Example(
            is_greenhouse="YES",
            uses_growlight="YES",
        )

        # Perfect classification, no Dutch terms, perfect tier-2 evidence
        prediction = MockPrediction(
            is_greenhouse="YES",
            uses_growlight="YES",
            dutch_terms_found=[],
            evidence_sources=[
                {
                    "url": "https://company.com",
                    "quote": "test",
                    "tier": "company_website",
                }  # tier 2 = 2.0/2.0 = 1.0 score
            ],
        )

        score, _ = dutch_aware_hierarchical_f1(example, prediction)

        # Should be 0.70 (classification) + 0.0 (no Dutch) + 0.15 (perfect evidence)
        assert abs(score - 0.85) < 0.01
