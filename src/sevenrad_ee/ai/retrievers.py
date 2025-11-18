"""
Retriever abstraction for dependency injection in DSPy greenhouse detection.

This module provides a Protocol for retriever modules and concrete implementations
for both cached (deterministic) and live (Perplexity) retrieval. This architecture
enables systematic optimization on frozen data while maintaining production capability
with live web search.

Key Components:
    - Retriever: Protocol defining the interface for evidence retrieval
    - CachedRetriever: Deterministic retriever using cached research data
    - PerplexityRetriever: Production retriever using live Perplexity search
"""

import json
import logging
from pathlib import Path
from typing import Protocol

try:
    import dspy
except ImportError as e:
    msg = (
        "dspy-ai package is required. Install with: "
        "uv pip install -e '.[dev]' or pip install dspy-ai~=2.5.0"
    )
    raise ImportError(msg) from e

from sevenrad_ee.ai.perplexity_cache import PerplexityAPIConfig
from sevenrad_ee.ai.perplexity_client import PerplexityClient

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    """
    Protocol for retriever modules.

    Any retriever must implement forward() to return evidence as a formatted string.
    This enables dependency injection for both cached (training) and live (production)
    evidence retrieval.
    """

    def forward(self, company_name: str, location: str) -> str:
        """
        Retrieve evidence for a company.

        Args:
            company_name: Company name
            location: Geographic location

        Returns:
            Formatted evidence string ready for classification

        """
        ...


class CachedRetriever(dspy.Module):  # type: ignore[misc]
    """
    Deterministic retriever using cached research data.

    Loads evidence from data/research/*.json files at initialization (eager loading)
    for fail-fast behavior and optimal performance. Critical for stable, reproducible
    optimization that prevents temporal overfitting.

    The cached data is stored in JSON files with this structure:
    {
        "company": "Company Name",
        "location": "Location",
        "queries": [
            {"query": "...", "content": "...", "citations": ["..."]}
        ]
    }
    """

    def __init__(self, cache_dir: Path | str) -> None:
        """
        Initialize cached retriever with eager loading.

        Loads all cached research files into memory for fast lookup and
        fail-fast error detection.

        Args:
            cache_dir: Directory containing cached research JSON files

        Raises:
            FileNotFoundError: If cache directory doesn't exist or contains no files
            ValueError: If JSON files are malformed

        """
        super().__init__()
        cache_path = Path(cache_dir)

        if not cache_path.exists():
            msg = f"Cache directory not found: {cache_path}"
            raise FileNotFoundError(msg)

        self.cache = self._load_cache(cache_path)

        if not self.cache:
            msg = f"No cached research files found in {cache_path}"
            raise FileNotFoundError(msg)

        logger.info("Loaded %d cached companies from %s", len(self.cache), cache_path)

    def _load_cache(self, cache_dir: Path) -> dict[str, str]:
        """
        Load all cached research files into memory.

        Args:
            cache_dir: Directory containing JSON cache files

        Returns:
            Dictionary mapping "company_location" keys to formatted evidence strings

        Raises:
            ValueError: If any JSON file is malformed

        """
        cache = {}

        for json_file in cache_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text())
            except json.JSONDecodeError as e:
                msg = f"Failed to decode JSON from {json_file}: {e}"
                raise ValueError(msg) from e

            # Extract company and location
            company = data.get("company", "")
            location = data.get("location", "")

            if not company or not location:
                logger.warning("Skipping %s: missing company or location", json_file)
                continue

            # Create cache key
            key = f"{company}_{location}"

            # Format cached evidence as passages (matching Perplexity structure)
            passages = []
            for query_result in data.get("queries", []):
                query = query_result.get("query", "")
                content = query_result.get("content", "")
                citations = query_result.get("citations", [])

                passage = (
                    f"Query: {query}\n"
                    f"Response: {content}\n"
                    f"Sources: {', '.join(citations)}"
                )
                passages.append(passage)

            # Store concatenated passages
            cache[key] = "\n\n".join(passages)

        return cache

    def forward(self, company_name: str, location: str) -> str:
        """
        Retrieve cached evidence for a company.

        Args:
            company_name: Company name
            location: Geographic location

        Returns:
            Formatted evidence string from cache

        """
        key = f"{company_name}_{location}"
        evidence = self.cache.get(key)

        if evidence is None:
            logger.warning("No cached data for %s (key: %s)", company_name, key)
            return "No cached data available for this company."

        logger.debug("Retrieved cached evidence for %s (%d chars)", company_name, len(evidence))
        return evidence


class PerplexityRetriever(dspy.Module):  # type: ignore[misc]
    """
    Production retriever using live Perplexity search.

    Executes a 3-query strategy for comprehensive evidence gathering:
    1. Positive signals (assimilatiebelichting, groeilicht, belichte teelt)
    2. Supplier associations (Signify, Hortilux, Philips LED, Gavita)
    3. Negative signals (onbelichte teelt, daglichtkas, zonder kunstlicht)

    Used in production and for generalization validation (Phase 4).
    """

    def __init__(self, client: PerplexityClient | None = None) -> None:
        """
        Initialize Perplexity retriever.

        Args:
            client: PerplexityClient instance. If None, creates default client
                   with API key from PERPLEXITY_API_KEY environment variable

        """
        super().__init__()
        self.client = client or PerplexityClient()

    def forward(self, company_name: str, location: str) -> str:
        """
        Execute live Perplexity search with 3-query strategy.

        Args:
            company_name: Company name
            location: Geographic location

        Returns:
            Formatted evidence string matching cached data structure

        """
        logger.info("Executing live Perplexity search for %s in %s", company_name, location)

        # Build 3-query strategy (matching cached research pattern)
        queries = [
            # Query 1: Positive signals for growlight usage
            (
                f'"{company_name}" {location} AND '
                f'(assimilatiebelichting OR groeilicht OR "belichte teelt")'
            ),
            # Query 2: Supplier associations
            (
                f'"{company_name}" AND '
                f'(Signify OR Hortilux OR "Philips LED" OR Gavita)'
            ),
            # Query 3: Negative signals (unlit cultivation)
            (
                f'"{company_name}" AND '
                f'("onbelichte teelt" OR "daglichtkas" OR "zonder kunstlicht")'
            ),
        ]

        # Execute queries
        passages = []
        config = PerplexityAPIConfig()  # Use default config

        for query in queries:
            try:
                response = self.client.query(query, config=config)

                passage = (
                    f"Query: {query}\n"
                    f"Response: {response.content}\n"
                    f"Sources: {', '.join([c.url for c in response.citations])}"
                )
                passages.append(passage)

            except Exception as e:
                logger.warning("Query failed for %s: %s", company_name, e)
                # Continue with other queries even if one fails
                passages.append(
                    f"Query: {query}\n"
                    f"Response: [Query failed: {e}]\n"
                    f"Sources: "
                )

        evidence = "\n\n".join(passages)
        logger.debug("Retrieved live evidence for %s (%d chars)", company_name, len(evidence))
        return evidence
