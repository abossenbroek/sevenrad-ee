"""
SHA-256 caching infrastructure for Perplexity API queries.

This module provides efficient caching for Perplexity Sonar API requests to minimize
costs and enable iterative data collection strategy refinement.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PerplexityAPIConfig(BaseModel):
    """Configuration for Perplexity API requests."""

    model: str = Field(
        default="sonar-pro",
        description="Perplexity model to use (sonar-pro is the primary online model)",
    )
    temperature: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Sampling temperature"
    )
    max_tokens: int = Field(default=1000, ge=1, description="Maximum response tokens")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling")
    search_recency_filter: str | None = Field(
        default=None,
        description="Recency filter (month, week, day) - optional",
    )


class Citation(BaseModel):
    """Citation from a Perplexity API response."""

    url: str = Field(..., description="Source URL")
    text: str = Field(..., description="Relevant text snippet from source")


class PerplexityResponse(BaseModel):
    """Structured Perplexity API response."""

    id: str = Field(..., description="Response ID")
    model: str = Field(..., description="Model used for response")
    content: str = Field(..., description="Generated response content")
    citations: list[Citation] = Field(
        default_factory=list, description="Source citations"
    )
    usage: dict[str, Any] = Field(
        default_factory=dict,
        description="Token usage statistics (can contain ints, floats, or strings)",
    )


class CacheManager:
    """Manage SHA-256 cached queries for Perplexity API."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        results_dir: Path | None = None,
    ) -> None:
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for cached API responses
                (default: cache/perplexity_searches)
            results_dir: Directory for research results
                (default: data/research)

        """
        self.cache_dir = cache_dir or Path("cache/perplexity_searches")
        self.results_dir = results_dir or Path("data/research")

        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def generate_cache_key(self, query: str, config: dict[str, Any]) -> str:
        """
        Generate SHA-256 cache key from query and configuration.

        Args:
            query: Search query string
            config: API configuration dictionary

        Returns:
            SHA-256 hash as hexadecimal string

        """
        # Combine query and config into deterministic string
        cache_input = query + json.dumps(config, sort_keys=True)
        return hashlib.sha256(cache_input.encode()).hexdigest()

    def get_cached_response(self, cache_key: str) -> PerplexityResponse | None:
        """
        Retrieve cached response if available.

        Args:
            cache_key: SHA-256 cache key

        Returns:
            Cached PerplexityResponse or None if not found

        """
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            cached_data = json.loads(cache_file.read_text())
            return PerplexityResponse(**cached_data)
        except (json.JSONDecodeError, ValueError) as e:
            # Invalid cache file - ignore and re-fetch
            logger.warning("Invalid cache file %s: %s", cache_file, e)
            return None

    def save_response(self, cache_key: str, response: PerplexityResponse) -> None:
        """
        Save API response to cache.

        Args:
            cache_key: SHA-256 cache key
            response: Perplexity API response to cache

        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        cache_file.write_text(response.model_dump_json(indent=2))

    def clear_cache(self) -> int:
        """
        Clear all cached responses.

        Returns:
            Number of cache files deleted

        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count
