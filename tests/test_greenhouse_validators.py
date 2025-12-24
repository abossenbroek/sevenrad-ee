"""
Tests for Phase 2 GreenhouseClassificationValidator.

Tests the 4 critical validation rules:
1. Hierarchical gating (not greenhouse → growlight = UNKNOWN)
2. Evidence requirement (uses_growlight=YES requires evidence)
3. Confidence-evidence alignment (high confidence requires tier-2)
4. Dutch terminology requirement (high confidence requires Dutch terms)
"""

import pytest
from sevenrad_ee.ai.dspy_greenhouse import (
    HIGH_CONFIDENCE_THRESHOLD,
    EvidenceSource,
    GreenhouseClassificationValidator,
    GreenhouseDetectionOutput,
)


class TestGreenhouseClassificationValidator:
    """Test GreenhouseClassificationValidator business rules."""

    def test_valid_prediction_passes_all_rules(self) -> None:
        """A valid prediction should pass all validation rules."""
        prediction = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="YES",
            species_grown=["roses"],
            dutch_terms_found=["assimilatiebelichting", "kwekerij"],
            evidence_sources=[
                EvidenceSource(
                    url="https://porta-nova.com",
                    quote="LED assimilatiebelichting voor rozen",
                    tier="company_website",
                )
            ],
            confidence=0.95,
            rationale="Company website confirms LED lighting",
        )

        is_valid, error_message = GreenhouseClassificationValidator.validate(prediction)

        assert is_valid is True
        assert error_message == ""

    def test_rule1_hierarchical_gating_not_greenhouse_with_yes(self) -> None:
        """Rule 1: is_greenhouse=NO + uses_growlight=YES should fail."""
        prediction = GreenhouseDetectionOutput(
            is_greenhouse="NO",
            uses_growlight="UNKNOWN",  # Must use UNKNOWN to pass Pydantic validation
            species_grown=[],
            dutch_terms_found=[],
            evidence_sources=[],
            confidence=0.7,
            rationale="Not a greenhouse",
        )

        # Manually set invalid state to test validator (bypassing Pydantic)
        prediction.uses_growlight = "YES"  # type: ignore[assignment]

        is_valid, error_message = GreenhouseClassificationValidator.validate(prediction)

        assert is_valid is False
        assert "uses_growlight must be UNKNOWN when is_greenhouse=NO" in error_message
        assert "got YES" in error_message

    def test_rule1_hierarchical_gating_not_greenhouse_with_no(self) -> None:
        """Rule 1: is_greenhouse=NO + uses_growlight=NO should fail."""
        prediction = GreenhouseDetectionOutput(
            is_greenhouse="NO",
            uses_growlight="UNKNOWN",
            species_grown=[],
            dutch_terms_found=[],
            evidence_sources=[],
            confidence=0.7,
            rationale="Not a greenhouse",
        )

        # Manually set invalid state
        prediction.uses_growlight = "NO"  # type: ignore[assignment]

        is_valid, error_message = GreenhouseClassificationValidator.validate(prediction)

        assert is_valid is False
        assert "uses_growlight must be UNKNOWN when is_greenhouse=NO" in error_message
        assert "got NO" in error_message

    def test_rule1_hierarchical_gating_not_greenhouse_with_unknown_passes(
        self,
    ) -> None:
        """Rule 1: is_greenhouse=NO + uses_growlight=UNKNOWN should pass."""
        prediction = GreenhouseDetectionOutput(
            is_greenhouse="NO",
            uses_growlight="UNKNOWN",
            species_grown=[],
            dutch_terms_found=[],
            evidence_sources=[],
            confidence=0.7,
            rationale="Not a greenhouse",
        )

        is_valid, error_message = GreenhouseClassificationValidator.validate(prediction)

        assert is_valid is True
        assert error_message == ""

    def test_rule2_evidence_requirement_yes_without_evidence(self) -> None:
        """Rule 2: uses_growlight=YES without evidence should fail."""
        prediction = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="UNKNOWN",  # Use UNKNOWN to bypass Pydantic validation
            species_grown=["roses"],
            dutch_terms_found=["kwekerij"],
            evidence_sources=[],  # No evidence
            confidence=0.6,
            rationale="Test",
        )

        # Manually set to YES after construction
        prediction.uses_growlight = "YES"  # type: ignore[assignment]

        is_valid, error_message = GreenhouseClassificationValidator.validate(prediction)

        assert is_valid is False
        assert "uses_growlight=YES requires evidence_sources" in error_message

    def test_rule2_evidence_requirement_yes_with_evidence_passes(self) -> None:
        """Rule 2: uses_growlight=YES with evidence should pass."""
        prediction = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="YES",
            species_grown=["roses"],
            dutch_terms_found=["assimilatiebelichting"],
            evidence_sources=[
                EvidenceSource(
                    url="https://example.com",
                    quote="Uses LED lighting",
                    tier="company_website",
                )
            ],
            confidence=0.9,
            rationale="Evidence found",
        )

        is_valid, error_message = GreenhouseClassificationValidator.validate(prediction)

        assert is_valid is True
        assert error_message == ""

    def test_rule2_evidence_not_required_for_no_or_unknown(self) -> None:
        """Rule 2: Evidence not required for uses_growlight=NO or UNKNOWN."""
        # NO without evidence
        prediction_no = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="NO",
            species_grown=[],
            dutch_terms_found=["onbelichte teelt"],
            evidence_sources=[],
            confidence=0.7,
            rationale="Explicitly unlit",
        )

        is_valid, _ = GreenhouseClassificationValidator.validate(prediction_no)
        assert is_valid is True

        # UNKNOWN without evidence
        prediction_unknown = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="UNKNOWN",
            species_grown=[],
            dutch_terms_found=[],
            evidence_sources=[],
            confidence=0.4,
            rationale="Insufficient evidence",
        )

        is_valid, _ = GreenhouseClassificationValidator.validate(prediction_unknown)
        assert is_valid is True

    def test_rule3_confidence_evidence_alignment_high_without_tier2(self) -> None:
        """Rule 3: High confidence without tier-2 evidence should fail."""
        prediction = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="YES",
            species_grown=["roses"],
            dutch_terms_found=["assimilatiebelichting"],
            evidence_sources=[
                EvidenceSource(
                    url="https://example.com",
                    quote="General web mention",
                    tier="general_web",  # Not tier-2
                )
            ],
            confidence=0.6,  # Start with low confidence
            rationale="Test",
        )

        # Manually set high confidence to test validator
        prediction.confidence = 0.85

        is_valid, error_message = GreenhouseClassificationValidator.validate(prediction)

        assert is_valid is False
        assert "High confidence" in error_message
        assert "tier-2 source" in error_message
        assert "found 0" in error_message

    def test_rule3_confidence_evidence_alignment_high_with_tier2_passes(
        self,
    ) -> None:
        """Rule 3: High confidence with tier-2 evidence should pass."""
        prediction = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="YES",
            species_grown=["roses"],
            dutch_terms_found=["assimilatiebelichting"],
            evidence_sources=[
                EvidenceSource(
                    url="https://porta-nova.com",
                    quote="Company website",
                    tier="company_website",  # Tier-2
                )
            ],
            confidence=0.92,
            rationale="Company website confirms",
        )

        is_valid, error_message = GreenhouseClassificationValidator.validate(prediction)

        assert is_valid is True
        assert error_message == ""

    def test_rule3_confidence_at_threshold_boundary(self) -> None:
        """Rule 3: Test behavior at HIGH_CONFIDENCE_THRESHOLD boundary."""
        # Just above threshold without tier-2 (should fail)
        prediction_above = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="UNKNOWN",
            species_grown=[],
            dutch_terms_found=["kwekerij"],
            evidence_sources=[
                EvidenceSource(
                    url="https://example.com", quote="test", tier="general_web"
                )
            ],
            confidence=0.7,  # Start low
            rationale="Test",
        )
        prediction_above.confidence = HIGH_CONFIDENCE_THRESHOLD + 0.01

        is_valid, _ = GreenhouseClassificationValidator.validate(prediction_above)
        assert is_valid is False

        # Just at threshold without tier-2 (should fail)
        prediction_at = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="UNKNOWN",
            species_grown=[],
            dutch_terms_found=["kwekerij"],
            evidence_sources=[
                EvidenceSource(
                    url="https://example.com", quote="test", tier="general_web"
                )
            ],
            confidence=0.7,
            rationale="Test",
        )
        prediction_at.confidence = HIGH_CONFIDENCE_THRESHOLD + 0.0001

        is_valid, _ = GreenhouseClassificationValidator.validate(prediction_at)
        assert is_valid is False

        # Just below threshold without tier-2 (should pass)
        prediction_below = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="UNKNOWN",
            species_grown=[],
            dutch_terms_found=[],  # No Dutch terms OK below threshold
            evidence_sources=[
                EvidenceSource(
                    url="https://example.com", quote="test", tier="general_web"
                )
            ],
            confidence=HIGH_CONFIDENCE_THRESHOLD - 0.01,
            rationale="Test",
        )

        is_valid, _ = GreenhouseClassificationValidator.validate(prediction_below)
        assert is_valid is True

    def test_rule3_all_tier2_types_count(self) -> None:
        """Rule 3: All tier-2 types should count for confidence validation."""
        tier2_types = ["company_website", "supplier_case_study", "job_posting"]

        for tier in tier2_types:
            prediction = GreenhouseDetectionOutput(
                is_greenhouse="YES",
                uses_growlight="YES",
                species_grown=["roses"],
                dutch_terms_found=["assimilatiebelichting"],
                evidence_sources=[
                    EvidenceSource(
                        url="https://example.com", quote="Test quote", tier=tier
                    )
                ],
                confidence=0.9,
                rationale=f"Tier-2 evidence: {tier}",
            )

            is_valid, error_message = GreenhouseClassificationValidator.validate(
                prediction
            )
            assert is_valid is True, f"Tier {tier} should count as tier-2"

    def test_rule4_dutch_terminology_requirement_high_without_terms(self) -> None:
        """Rule 4: High confidence without Dutch terms should fail."""
        prediction = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="YES",
            species_grown=["roses"],
            dutch_terms_found=[],  # No Dutch terms
            evidence_sources=[
                EvidenceSource(
                    url="https://example.com",
                    quote="Test",
                    tier="company_website",  # Have tier-2
                )
            ],
            confidence=0.6,
            rationale="Test",
        )

        # Manually set high confidence
        prediction.confidence = 0.85

        is_valid, error_message = GreenhouseClassificationValidator.validate(prediction)

        assert is_valid is False
        assert "High confidence" in error_message
        assert "Dutch terminology" in error_message

    def test_rule4_dutch_terminology_requirement_high_with_terms_passes(self) -> None:
        """Rule 4: High confidence with Dutch terms should pass."""
        prediction = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="YES",
            species_grown=["roses"],
            dutch_terms_found=["assimilatiebelichting", "kwekerij"],
            evidence_sources=[
                EvidenceSource(
                    url="https://example.com", quote="Test", tier="company_website"
                )
            ],
            confidence=0.93,
            rationale="Dutch terminology found",
        )

        is_valid, error_message = GreenhouseClassificationValidator.validate(prediction)

        assert is_valid is True
        assert error_message == ""

    def test_rule4_dutch_terms_not_required_for_medium_confidence(self) -> None:
        """Rule 4: Dutch terms not required for medium confidence."""
        prediction = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="UNKNOWN",
            species_grown=[],
            dutch_terms_found=[],  # No Dutch terms
            evidence_sources=[
                EvidenceSource(
                    url="https://example.com", quote="Test", tier="general_web"
                )
            ],
            confidence=0.65,  # Medium confidence
            rationale="Limited evidence",
        )

        is_valid, error_message = GreenhouseClassificationValidator.validate(prediction)

        assert is_valid is True
        assert error_message == ""


class TestValidatorReturnType:
    """Test validator return type consistency."""

    def test_valid_prediction_returns_true_empty_string(self) -> None:
        """Valid predictions should return (True, '')."""
        prediction = GreenhouseDetectionOutput(
            is_greenhouse="NO",
            uses_growlight="UNKNOWN",
            species_grown=[],
            dutch_terms_found=[],
            evidence_sources=[],
            confidence=0.7,
            rationale="Not a greenhouse",
        )

        result = GreenhouseClassificationValidator.validate(prediction)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is True
        assert result[1] == ""

    def test_invalid_prediction_returns_false_with_message(self) -> None:
        """Invalid predictions should return (False, 'error message')."""
        prediction = GreenhouseDetectionOutput(
            is_greenhouse="NO",
            uses_growlight="UNKNOWN",
            species_grown=[],
            dutch_terms_found=[],
            evidence_sources=[],
            confidence=0.7,
            rationale="Test",
        )

        # Manually violate a rule
        prediction.uses_growlight = "YES"  # type: ignore[assignment]

        result = GreenhouseClassificationValidator.validate(prediction)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is False
        assert isinstance(result[1], str)
        assert len(result[1]) > 0  # Should have an error message


class TestMultipleRuleViolations:
    """Test behavior when multiple rules are violated."""

    def test_validator_checks_rules_in_order(self) -> None:
        """Validator checks rules in order and reports first violation."""
        # Create a prediction that passes Pydantic but fails validator
        prediction = GreenhouseDetectionOutput(
            is_greenhouse="NO",
            uses_growlight="UNKNOWN",
            species_grown=[],
            dutch_terms_found=[],
            evidence_sources=[],
            confidence=0.7,
            rationale="Test",
        )

        # Manually violate multiple rules (bypassing Pydantic)
        prediction.uses_growlight = "YES"  # type: ignore[assignment]  # Violates Rule 1
        # Also violates Rule 2 (no evidence for YES)

        is_valid, error_message = GreenhouseClassificationValidator.validate(prediction)

        assert is_valid is False
        # Should report hierarchical gating first (Rule 1 checked before Rule 2)
        assert "uses_growlight must be UNKNOWN when is_greenhouse=NO" in error_message
