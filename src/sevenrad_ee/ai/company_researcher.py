"""
Company greenhouse research using 3-query Perplexity strategy.

This module implements the systematic 3-query strategy for researching Dutch
greenhouse companies using Perplexity Sonar API with evidence categorization.
"""

from pathlib import Path

import dspy
from rich.console import Console

from sevenrad_ee.ai.company_research_models import (
    ClassificationSuggestion,
    CompanyResearchResult,
    Evidence,
    EvidenceItem,
    EvidenceSourceTier,
    QueryResult,
)
from sevenrad_ee.ai.evidence_classifier import (
    EvidenceCategory,
    GrowLightEvidenceClassifier,
)
from sevenrad_ee.ai.perplexity_cache import PerplexityResponse
from sevenrad_ee.ai.perplexity_client import PerplexityAPIError, PerplexityClient

console = Console()

# Dutch terminology patterns for evidence analysis
DUTCH_POSITIVE_TERMS = {
    "assimilatiebelichting",
    "assimilatieverlichting",
    "kunstlicht",
    "groeilicht",
    "belichte teelt",
    "led-belichting",
    "son-t",
    "assimilatielampen",
}

DUTCH_NEGATIVE_TERMS = {
    "onbelichte teelt",
    "zonder kunstlicht",
    "daglichtkas",
}

SUPPLIER_NAMES = {
    "signify",
    "hortilux",
    "philips",
    "gavita",
    "fluence",
    "oreon",
}

# Classification thresholds
MIN_POSITIVE_EVIDENCE_FOR_CLASSIFICATION = 2
MIN_NEGATIVE_EVIDENCE_FOR_CLASSIFICATION = 1


class CompanyResearcher:
    """Research Dutch greenhouse companies using Perplexity 3-query strategy."""

    def __init__(self, client: PerplexityClient) -> None:
        """
        Initialize company researcher.

        Args:
            client: Perplexity API client instance

        """
        self.client = client
        # Initialize LLM-based evidence classifier for semantic understanding
        self.evidence_classifier = dspy.Predict(GrowLightEvidenceClassifier)

    def research_company(
        self,
        company_name: str,
        location: str,
    ) -> CompanyResearchResult:
        """
        Research a single company using 3-query strategy.

        Strategy:
        - Query 1: Positive signals (assimilatiebelichting, groeilicht, etc.)
        - Query 2: Supplier associations (Signify, Hortilux, Philips LED, etc.)
        - Query 3: Negative signals (onbelichte teelt, daglichtkas)

        Args:
            company_name: Company name to research
            location: Geographic location

        Returns:
            Structured research result with categorized evidence

        Raises:
            PerplexityAPIError: If API queries fail

        """
        console.print(f"\n[cyan]Researching:[/cyan] {company_name}, {location}")

        queries = self._build_queries(company_name, location)
        query_results: list[QueryResult] = []
        evidence = Evidence()
        dutch_terms: set[str] = set()

        # Execute queries
        for i, query in enumerate(queries, 1):
            console.print(f"  [dim]Query {i}/3:[/dim] {query[:60]}...")

            try:
                response = self.client.query(query)

                # Store query result
                query_results.append(
                    QueryResult(
                        query=query,
                        response_id=response.id,
                        model=response.model,
                        content=response.content,
                        citations=[c.url for c in response.citations],
                    )
                )

                # Categorize evidence from this query
                self._categorize_evidence(
                    response,
                    query_source=f"query_{i}",
                    evidence=evidence,
                    dutch_terms=dutch_terms,
                    company_name=company_name,
                )

            except PerplexityAPIError as e:
                console.print(f"  [red]✗[/red] Query {i} failed: {e}")
                continue

        # Suggest classification
        suggestion = self._suggest_classification(evidence)

        # Calculate confidence
        confidence = self._calculate_confidence(evidence, dutch_terms)

        return CompanyResearchResult(
            company=company_name,
            location=location,
            queries=query_results,
            evidence=evidence,
            classification_suggestion=suggestion,
            dutch_terms_found=list(dutch_terms),
            confidence_score=confidence,
        )

    def _build_queries(self, company_name: str, location: str) -> list[str]:
        """
        Build 3-query strategy for company research.

        Args:
            company_name: Company name
            location: Geographic location

        Returns:
            List of 3 query strings

        """
        return [
            # Query 1: Positive signals
            f'"{company_name}" {location} AND '
            f'(assimilatiebelichting OR groeilicht OR "belichte teelt")',
            # Query 2: Supplier associations
            f'"{company_name}" AND '
            f'(Signify OR Hortilux OR "Philips LED" OR Gavita)',
            # Query 3: Negative signals
            f'"{company_name}" AND '
            f'("onbelichte teelt" OR "daglichtkas" OR "zonder kunstlicht")',
        ]

    def _categorize_evidence(
        self,
        response: PerplexityResponse,
        query_source: str,
        evidence: Evidence,
        dutch_terms: set[str],
        company_name: str,
    ) -> None:
        """
        Categorize evidence using LLM semantic classification.

        Args:
            response: Perplexity API response
            query_source: Source identifier (e.g., 'query_1')
            evidence: Evidence container to populate
            dutch_terms: Set to collect Dutch terms found
            company_name: Company name for context

        """
        content_lower = response.content.lower()

        # Extract Dutch terms (keep for statistics/debugging)
        for term in DUTCH_POSITIVE_TERMS | DUTCH_NEGATIVE_TERMS:
            if term in content_lower:
                dutch_terms.add(term)

        # Use LLM to semantically classify the evidence
        classification = self.evidence_classifier(
            company_name=company_name, evidence_text=response.content
        )

        category = EvidenceCategory(classification.evidence_category)

        # Create evidence items for all citations
        for citation in response.citations:
            tier = self._classify_source_tier(citation.url, response.content)

            item = EvidenceItem(
                url=citation.url,
                snippet=response.content[:200],
                source=query_source,
                tier=tier,
            )

            # Categorize based on LLM's semantic understanding
            if category == EvidenceCategory.POSITIVE:
                evidence.positive.append(item)
            elif category == EvidenceCategory.NEGATIVE:
                evidence.negative.append(item)
            else:  # NEUTRAL or AMBIGUOUS
                evidence.ambiguous.append(item)

    def _classify_source_tier(self, url: str, content: str) -> EvidenceSourceTier:
        """
        Classify source quality tier based on URL and content.

        Args:
            url: Source URL
            content: Page content

        Returns:
            Source quality tier

        """
        url_lower = url.lower()
        content_lower = content.lower()

        # Tier 2: Company website
        if any(
            domain in url_lower
            for domain in [".nl/", ".com/", ".eu/"]
            if "nieuws" not in url_lower
            and "news" not in url_lower
            and "blog" not in url_lower
        ):
            return EvidenceSourceTier.COMPANY_WEBSITE

        # Tier 2: Supplier case studies
        if any(
            domain in url_lower
            for domain in [
                "signify.com",
                "philips.com",
                "hortilux.com",
                "priva.com",
            ]
        ):
            return EvidenceSourceTier.SUPPLIER_CASE_STUDY

        # Tier 2: Job postings
        if any(domain in url_lower for domain in ["indeed.nl", "linkedin.com"]) and any(
            keyword in content_lower for keyword in ["vacature", "belichting"]
        ):
            return EvidenceSourceTier.JOB_POSTING

        # Tier 1: Dutch trade media
        if any(
            domain in url_lower
            for domain in [
                "groentennieuws.nl",
                "floraldaily",
                "agf.nl",
                "tuinbouw.nl",
            ]
        ):
            return EvidenceSourceTier.TRADE_MEDIA_NL

        # Tier 0.5: General web
        return EvidenceSourceTier.GENERAL_WEB

    def _suggest_classification(self, evidence: Evidence) -> ClassificationSuggestion:
        """
        Suggest classification, prioritizing negative evidence.

        Args:
            evidence: Categorized evidence

        Returns:
            Classification suggestion

        """
        pos_count = len(evidence.positive)
        neg_count = len(evidence.negative)

        # Strong negative evidence is definitive
        if neg_count >= MIN_NEGATIVE_EVIDENCE_FOR_CLASSIFICATION:
            # Contradictory evidence needs manual review
            if pos_count > 0:
                return ClassificationSuggestion.NEEDS_MANUAL_REVIEW
            return ClassificationSuggestion.NEGATIVE

        # Positive evidence
        elif pos_count >= MIN_POSITIVE_EVIDENCE_FOR_CLASSIFICATION:
            return ClassificationSuggestion.POSITIVE

        # Insufficient evidence either way
        else:
            return ClassificationSuggestion.NEEDS_MANUAL_REVIEW

    def _calculate_confidence(self, evidence: Evidence, dutch_terms: set[str]) -> float:
        """
        Calculate confidence score for classification.

        Args:
            evidence: Categorized evidence
            dutch_terms: Dutch terms found

        Returns:
            Confidence score (0.0-1.0)

        """
        score = 0.0

        # Evidence count (40% weight)
        total_evidence = len(evidence.positive) + len(evidence.negative)
        if total_evidence > 0:
            score += min(total_evidence / 5.0, 1.0) * 0.4

        # Tier-2 evidence (30% weight)
        tier2_sources = {
            EvidenceSourceTier.COMPANY_WEBSITE,
            EvidenceSourceTier.SUPPLIER_CASE_STUDY,
            EvidenceSourceTier.JOB_POSTING,
        }
        tier2_count = sum(
            1
            for item in evidence.positive + evidence.negative
            if item.tier in tier2_sources
        )
        if tier2_count > 0:
            score += min(tier2_count / 3.0, 1.0) * 0.3

        # Dutch terminology (30% weight)
        if dutch_terms:
            score += min(len(dutch_terms) / 5.0, 1.0) * 0.3

        return round(score, 2)

    def save_result(
        self, result: CompanyResearchResult, output_dir: Path | None = None
    ) -> Path:
        """
        Save research result to JSON file.

        Args:
            result: Research result to save
            output_dir: Output directory (default: data/research)

        Returns:
            Path to saved file

        """
        if output_dir is None:
            output_dir = Path("data/research")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"{result.company.replace(' ', '_')}_{result.location}.json"
        output_path = output_dir / filename

        # Save as JSON
        output_path.write_text(result.model_dump_json(indent=2))

        return output_path
