"""
Tests for Phase 2 DSPy greenhouse detection models.

Tests the enhanced hierarchical classification system with:
- EvidenceSource Pydantic v2 model
- GreenhouseDetectionOutput with field validators
- Hierarchical logic enforcement
- Dutch terminology tracking
- Evidence quality tiers
"""

import pytest
from pydantic import ValidationError
from sevenrad_ee.ai.dspy_greenhouse import (
    HIGH_CONFIDENCE_THRESHOLD,
    EvidenceSource,
    GreenhouseDetectionOutput,
)


class TestEvidenceSource:
    """Test EvidenceSource Pydantic model."""

    def test_valid_evidence_source(self) -> None:
        """Create evidence source with all valid fields."""
        evidence = EvidenceSource(
            url="https://signify.com/nl-nl/case-study/porta-nova",
            quote="Porta Nova installeert 5000 Philips GreenPower LED modules",
            tier="supplier_case_study",
        )

        assert evidence.url == "https://signify.com/nl-nl/case-study/porta-nova"
        assert "LED" in evidence.quote
        assert evidence.tier == "supplier_case_study"

    def test_all_tier_types(self) -> None:
        """Test all valid tier classifications."""
        valid_tiers = [
            "company_website",
            "supplier_case_study",
            "job_posting",
            "trade_media_nl",
            "general_web",
        ]

        for tier in valid_tiers:
            evidence = EvidenceSource(
                url="https://example.com",
                quote="Test quote",
                tier=tier,
            )
            assert evidence.tier == tier

    def test_invalid_tier_rejected(self) -> None:
        """Invalid tier classifications should be rejected."""
        with pytest.raises(ValidationError):
            EvidenceSource(
                url="https://example.com",
                quote="Test quote",
                tier="invalid_tier",  # type: ignore[arg-type]
            )


class TestGreenhouseDetectionOutput:
    """Test GreenhouseDetectionOutput Pydantic model with validators."""

    def test_valid_greenhouse_with_lighting(self) -> None:
        """Valid greenhouse with lighting and evidence."""
        output = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="YES",
            species_grown=["roses", "gerbera"],
            dutch_terms_found=["assimilatiebelichting", "kwekerij"],
            evidence_sources=[
                EvidenceSource(
                    url="https://porta-nova.com",
                    quote="LED assimilatiebelichting",
                    tier="company_website",
                )
            ],
            confidence=0.9,
            rationale="Company website confirms LED lighting usage",
        )

        assert output.is_greenhouse == "YES"
        assert output.uses_growlight == "YES"
        assert len(output.species_grown) == 2
        assert len(output.dutch_terms_found) == 2
        assert len(output.evidence_sources) == 1
        assert output.confidence == 0.9

    def test_valid_greenhouse_no_lighting(self) -> None:
        """Valid greenhouse without lighting evidence."""
        output = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="NO",
            species_grown=["lettuce"],
            dutch_terms_found=["onbelichte teelt"],
            evidence_sources=[
                EvidenceSource(
                    url="https://example.nl",
                    quote="onbelichte teelt, alleen daglicht",
                    tier="trade_media_nl",
                )
            ],
            confidence=0.75,
            rationale="Explicitly states unlit cultivation",
        )

        assert output.is_greenhouse == "YES"
        assert output.uses_growlight == "NO"
        assert "onbelichte teelt" in output.dutch_terms_found

    def test_valid_not_greenhouse(self) -> None:
        """Valid non-greenhouse classification."""
        output = GreenhouseDetectionOutput(
            is_greenhouse="NO",
            uses_growlight="UNKNOWN",
            species_grown=[],
            dutch_terms_found=[],
            evidence_sources=[],
            confidence=0.6,
            rationale="Company is a flower auction house, not a greenhouse",
        )

        assert output.is_greenhouse == "NO"
        assert output.uses_growlight == "UNKNOWN"

    def test_hierarchical_gating_enforced(self) -> None:
        """If not greenhouse, growlight must be UNKNOWN."""
        with pytest.raises(ValidationError, match="uses_growlight must be UNKNOWN"):
            GreenhouseDetectionOutput(
                is_greenhouse="NO",
                uses_growlight="YES",  # INVALID - should be UNKNOWN
                species_grown=[],
                dutch_terms_found=[],
                evidence_sources=[],
                confidence=0.5,
                rationale="Invalid",
            )

        with pytest.raises(ValidationError, match="uses_growlight must be UNKNOWN"):
            GreenhouseDetectionOutput(
                is_greenhouse="NO",
                uses_growlight="NO",  # INVALID - should be UNKNOWN
                species_grown=[],
                dutch_terms_found=[],
                evidence_sources=[],
                confidence=0.5,
                rationale="Invalid",
            )

    def test_high_confidence_requires_tier2_evidence(self) -> None:
        """High confidence (>0.8) requires tier-2 evidence."""
        # Should fail: high confidence without tier-2 evidence
        with pytest.raises(
            ValidationError, match="High confidence.*requires.*tier-2 source"
        ):
            GreenhouseDetectionOutput(
                is_greenhouse="YES",
                uses_growlight="YES",
                species_grown=["roses"],
                dutch_terms_found=["assimilatiebelichting"],
                evidence_sources=[
                    EvidenceSource(
                        url="https://example.com",
                        quote="general web mention",
                        tier="general_web",  # Only tier 0.5
                    )
                ],
                confidence=0.9,  # High confidence
                rationale="Test",
            )

    def test_high_confidence_with_tier2_evidence_passes(self) -> None:
        """High confidence with tier-2 evidence should pass."""
        output = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="YES",
            species_grown=["roses"],
            dutch_terms_found=["assimilatiebelichting"],
            evidence_sources=[
                EvidenceSource(
                    url="https://porta-nova.com",
                    quote="LED lighting",
                    tier="company_website",  # Tier 2
                )
            ],
            confidence=0.9,
            rationale="Company website confirms",
        )

        assert output.confidence == 0.9

    def test_medium_confidence_allows_any_evidence(self) -> None:
        """Medium confidence (<0.8) doesn't require tier-2 evidence."""
        output = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="UNKNOWN",
            species_grown=[],
            dutch_terms_found=["kwekerij"],
            evidence_sources=[
                EvidenceSource(
                    url="https://example.com",
                    quote="general mention",
                    tier="general_web",
                )
            ],
            confidence=0.6,  # Medium confidence
            rationale="Insufficient evidence",
        )

        assert output.confidence == 0.6

    def test_confidence_bounds(self) -> None:
        """Confidence must be between 0.0 and 1.0."""
        # Valid bounds
        GreenhouseDetectionOutput(
            is_greenhouse="NO",
            uses_growlight="UNKNOWN",
            species_grown=[],
            dutch_terms_found=[],
            evidence_sources=[],
            confidence=0.0,
            rationale="No confidence",
        )

        # For confidence=1.0, need to satisfy high confidence requirements
        GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="YES",
            species_grown=[],
            dutch_terms_found=["kwekerij"],  # Need Dutch terms for high confidence
            evidence_sources=[
                EvidenceSource(
                    url="https://example.com",
                    quote="test",
                    tier="company_website",  # Need tier-2 for high confidence
                )
            ],
            confidence=1.0,
            rationale="Full confidence",
        )

        # Invalid bounds
        with pytest.raises(ValidationError):
            GreenhouseDetectionOutput(
                is_greenhouse="NO",
                uses_growlight="UNKNOWN",
                species_grown=[],
                dutch_terms_found=[],
                evidence_sources=[],
                confidence=-0.1,  # Too low
                rationale="Invalid",
            )

        with pytest.raises(ValidationError):
            GreenhouseDetectionOutput(
                is_greenhouse="NO",
                uses_growlight="UNKNOWN",
                species_grown=[],
                dutch_terms_found=[],
                evidence_sources=[],
                confidence=1.5,  # Too high
                rationale="Invalid",
            )

    def test_empty_lists_allowed(self) -> None:
        """Empty lists for species, terms, and sources are valid."""
        output = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="UNKNOWN",
            species_grown=[],  # Empty OK
            dutch_terms_found=[],  # Empty OK
            evidence_sources=[],  # Empty OK
            confidence=0.4,
            rationale="Insufficient information found",
        )

        assert output.species_grown == []
        assert output.dutch_terms_found == []
        assert output.evidence_sources == []


class TestHierarchicalLogicValidation:
    """Test hierarchical logic enforcement across scenarios."""

    def test_greenhouse_yes_with_all_growlight_states(self) -> None:
        """When is_greenhouse=YES, any growlight state is valid."""
        # YES + YES
        GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="YES",
            species_grown=[],
            dutch_terms_found=[],
            evidence_sources=[
                EvidenceSource(
                    url="https://example.com", quote="test", tier="company_website"
                )
            ],
            confidence=0.9,
            rationale="Valid",
        )

        # YES + NO
        GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="NO",
            species_grown=[],
            dutch_terms_found=[],
            evidence_sources=[],
            confidence=0.7,
            rationale="Valid",
        )

        # YES + UNKNOWN
        GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="UNKNOWN",
            species_grown=[],
            dutch_terms_found=[],
            evidence_sources=[],
            confidence=0.5,
            rationale="Valid",
        )

    def test_greenhouse_no_only_allows_unknown(self) -> None:
        """When is_greenhouse=NO, only UNKNOWN is valid for growlight."""
        # NO + UNKNOWN (valid)
        GreenhouseDetectionOutput(
            is_greenhouse="NO",
            uses_growlight="UNKNOWN",
            species_grown=[],
            dutch_terms_found=[],
            evidence_sources=[],
            confidence=0.8,
            rationale="Not a greenhouse",
        )

        # NO + YES (invalid)
        with pytest.raises(ValidationError):
            GreenhouseDetectionOutput(
                is_greenhouse="NO",
                uses_growlight="YES",
                species_grown=[],
                dutch_terms_found=[],
                evidence_sources=[],
                confidence=0.5,
                rationale="Invalid",
            )

        # NO + NO (invalid)
        with pytest.raises(ValidationError):
            GreenhouseDetectionOutput(
                is_greenhouse="NO",
                uses_growlight="NO",
                species_grown=[],
                dutch_terms_found=[],
                evidence_sources=[],
                confidence=0.5,
                rationale="Invalid",
            )


class TestDutchTerminologyTracking:
    """Test Dutch terminology tracking functionality."""

    def test_multiple_dutch_terms_tracked(self) -> None:
        """Multiple Dutch terms can be tracked."""
        terms = [
            "assimilatiebelichting",
            "kwekerij",
            "belichte teelt",
            "LED-belichting",
            "SON-T",
        ]

        output = GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="YES",
            species_grown=["rozen"],
            dutch_terms_found=terms,
            evidence_sources=[
                EvidenceSource(
                    url="https://example.nl", quote="test", tier="company_website"
                )
            ],
            confidence=0.95,
            rationale="Rich Dutch terminology found",
        )

        assert len(output.dutch_terms_found) == 5
        assert "assimilatiebelichting" in output.dutch_terms_found
        assert "SON-T" in output.dutch_terms_found


class TestConfidenceThreshold:
    """Test HIGH_CONFIDENCE_THRESHOLD constant."""

    def test_threshold_value(self) -> None:
        """Verify HIGH_CONFIDENCE_THRESHOLD is set correctly."""
        assert HIGH_CONFIDENCE_THRESHOLD == 0.8

    def test_threshold_used_in_validation(self) -> None:
        """Verify threshold is actually enforced."""
        # Just above threshold
        with pytest.raises(ValidationError):
            GreenhouseDetectionOutput(
                is_greenhouse="YES",
                uses_growlight="YES",
                species_grown=[],
                dutch_terms_found=["kwekerij"],  # Have Dutch terms
                evidence_sources=[
                    EvidenceSource(
                        url="https://example.com",
                        quote="test",
                        tier="general_web",  # Not tier-2
                    )
                ],
                confidence=0.81,  # Above threshold
                rationale="Test",
            )

        # Just below threshold (should pass)
        GreenhouseDetectionOutput(
            is_greenhouse="YES",
            uses_growlight="UNKNOWN",
            species_grown=[],
            dutch_terms_found=[],  # No Dutch terms OK for medium confidence
            evidence_sources=[
                EvidenceSource(
                    url="https://example.com", quote="test", tier="general_web"
                )
            ],
            confidence=0.79,  # Below threshold
            rationale="Test",
        )
