"""
Pydantic v2 models for structured company greenhouse research results.

This module defines data models for storing company research results with
categorized evidence, source attribution, and classification suggestions.
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class EvidenceSourceTier(str, Enum):
    """Quality tier for evidence sources."""

    COMPANY_WEBSITE = "company_website"
    SUPPLIER_CASE_STUDY = "supplier_case_study"
    JOB_POSTING = "job_posting"
    TRADE_MEDIA_NL = "trade_media_nl"
    GENERAL_WEB = "general_web"


class QueryResult(BaseModel):
    """Result from a single Perplexity query."""

    query: str = Field(..., description="Search query executed")
    response_id: str = Field(..., description="Perplexity response ID")
    model: str = Field(..., description="Model used for response")
    content: str = Field(..., description="Response content")
    citations: list[str] = Field(
        default_factory=list, description="Citation URLs from response"
    )


class EvidenceItem(BaseModel):
    """Single piece of evidence with source attribution."""

    url: str = Field(..., description="Source URL")
    snippet: str = Field(..., description="Relevant text snippet")
    source: str = Field(
        ...,
        description="Query source (e.g., 'query_1_positive', 'query_2_supplier')",
    )
    tier: EvidenceSourceTier = Field(
        default=EvidenceSourceTier.GENERAL_WEB,
        description="Source quality tier",
    )


class Evidence(BaseModel):
    """Categorized evidence for company classification."""

    positive: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence supporting growlight usage",
    )
    negative: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence against growlight usage (e.g., 'onbelichte teelt')",
    )
    ambiguous: list[EvidenceItem] = Field(
        default_factory=list,
        description="Unclear or conflicting evidence",
    )


class ClassificationSuggestion(str, Enum):
    """Classification suggestion for human review."""

    POSITIVE = "POSITIVE (uses growlights)"
    NEGATIVE = "NEGATIVE (no growlights)"
    NEEDS_MANUAL_REVIEW = "NEEDS_MANUAL_REVIEW"


class CompanyResearchResult(BaseModel):
    """Complete research result for a single company."""

    company: str = Field(..., description="Company name")
    location: str = Field(..., description="Geographic location")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Research timestamp",
    )
    queries: list[QueryResult] = Field(
        default_factory=list,
        description="All queries executed for this company",
    )
    evidence: Evidence = Field(
        default_factory=Evidence,
        description="Categorized evidence",
    )
    classification_suggestion: ClassificationSuggestion = Field(
        ...,
        description="Automated classification suggestion for human validation",
    )
    dutch_terms_found: list[str] = Field(
        default_factory=list,
        description="Dutch horticultural terms found in sources",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in classification (0.0-1.0)",
    )
    notes: str = Field(
        default="",
        description="Additional notes or observations",
    )

    def tier2_evidence_count(self) -> int:
        """
        Count tier-2 evidence sources (company websites, case studies, job postings).

        Returns:
            Number of tier-2 evidence items

        """
        tier2_sources = {
            EvidenceSourceTier.COMPANY_WEBSITE,
            EvidenceSourceTier.SUPPLIER_CASE_STUDY,
            EvidenceSourceTier.JOB_POSTING,
        }

        count = 0
        for item in self.evidence.positive + self.evidence.negative:
            if item.tier in tier2_sources:
                count += 1

        return count

    def tier2_percentage(self) -> float:
        """
        Calculate percentage of evidence from tier-2 sources.

        Returns:
            Percentage (0.0-100.0) of tier-2 evidence

        """
        total_evidence = (
            len(self.evidence.positive)
            + len(self.evidence.negative)
            + len(self.evidence.ambiguous)
        )

        if total_evidence == 0:
            return 0.0

        return (self.tier2_evidence_count() / total_evidence) * 100

    def has_dutch_sources(self) -> bool:
        """
        Check if research includes Dutch-language sources.

        Returns:
            True if Dutch terms were found

        """
        return len(self.dutch_terms_found) > 0


class ValidationDecision(BaseModel):
    """Human validation decision for a company research result."""

    company: str = Field(..., description="Company name")
    classification: Literal["POSITIVE", "NEGATIVE", "UNKNOWN"] = Field(
        ...,
        description="Final human classification",
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        ...,
        description="Human confidence in classification",
    )
    evidence_tier: Literal[1, 2] = Field(
        ...,
        description="Highest evidence tier supporting decision",
    )
    primary_sources: list[str] = Field(
        default_factory=list,
        description="Key source URLs supporting decision",
    )
    validation_notes: str = Field(
        default="",
        description="Human reviewer notes",
    )
    verified: bool = Field(
        default=True,
        description="Whether decision has been verified",
    )
    validation_date: datetime = Field(
        default_factory=datetime.now,
        description="Date of validation",
    )
