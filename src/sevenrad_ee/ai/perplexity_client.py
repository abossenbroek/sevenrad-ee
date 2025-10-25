"""
Perplexity API client with SHA-256 caching and rate limiting.

This module provides a production-ready client for the Perplexity Sonar API with:
- Automatic SHA-256 caching to minimize API costs
- Rate limiting with configurable delays
- Structured error handling
- Rich console output for progress tracking
"""

import os
import time
from typing import Any

import requests
from dotenv import load_dotenv
from rich.console import Console

from sevenrad_ee.ai.perplexity_cache import (
    CacheManager,
    Citation,
    PerplexityAPIConfig,
    PerplexityResponse,
)

# Load environment variables from .env file
load_dotenv()

console = Console()


class PerplexityAPIError(Exception):
    """Exception raised for Perplexity API errors."""

    pass


class PerplexityClient:
    """Client for Perplexity Sonar API with caching and rate limiting."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_manager: CacheManager | None = None,
        rate_limit_delay: float = 1.0,
    ) -> None:
        """
        Initialize Perplexity API client.

        Args:
            api_key: Perplexity API key (defaults to PERPLEXITY_API_KEY env var)
            cache_manager: Cache manager instance (creates default if None)
            rate_limit_delay: Delay in seconds between API calls (default: 1.0)

        Raises:
            ValueError: If API key is not provided or found in environment

        """
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "PERPLEXITY_API_KEY not found in environment. "
                "Set it with: export PERPLEXITY_API_KEY='your-key'"
            )

        self.cache_manager = cache_manager or CacheManager()
        self.rate_limit_delay = rate_limit_delay
        self.api_url = "https://api.perplexity.ai/chat/completions"

    def query(
        self,
        query: str,
        config: PerplexityAPIConfig | None = None,
        force_refresh: bool = False,
    ) -> PerplexityResponse:
        """
        Execute Perplexity query with automatic caching.

        Args:
            query: Search query string
            config: API configuration (uses defaults if None)
            force_refresh: If True, bypass cache and fetch new response

        Returns:
            Perplexity API response with citations

        Raises:
            PerplexityAPIError: If API request fails

        """
        if config is None:
            config = PerplexityAPIConfig()

        # Generate cache key
        config_dict = config.model_dump()
        cache_key = self.cache_manager.generate_cache_key(query, config_dict)

        # Check cache unless force refresh
        if not force_refresh:
            cached = self.cache_manager.get_cached_response(cache_key)
            if cached is not None:
                console.print(
                    f"  [dim][CACHE HIT][/dim] {query[:60]}...",
                    style="cyan",
                )
                return cached

        # Execute API call
        console.print(
            f"  [dim][API CALL][/dim] {query[:60]}...",
            style="yellow",
        )

        try:
            response = self._execute_request(query, config_dict)
            self.cache_manager.save_response(cache_key, response)

            # Rate limiting courtesy delay
            time.sleep(self.rate_limit_delay)

            return response

        except requests.exceptions.RequestException as e:
            raise PerplexityAPIError(f"API request failed: {e}") from e

    def _execute_request(
        self,
        query: str,
        config_dict: dict[str, Any],
    ) -> PerplexityResponse:
        """
        Execute API request and parse response.

        Args:
            query: Search query
            config_dict: Configuration dictionary

        Returns:
            Parsed PerplexityResponse

        Raises:
            PerplexityAPIError: If request fails or response is invalid

        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Filter out None values from config
        filtered_config = {k: v for k, v in config_dict.items() if v is not None}

        payload = {
            **filtered_config,
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ],
        }

        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=30,
        )

        response.raise_for_status()
        data = response.json()

        # Parse response into Pydantic model
        try:
            return self._parse_response(data)
        except (KeyError, ValueError) as e:
            raise PerplexityAPIError(f"Failed to parse API response: {e}") from e

    def _parse_response(self, data: dict[str, Any]) -> PerplexityResponse:
        """
        Parse raw API response into structured format.

        Args:
            data: Raw API response JSON

        Returns:
            Structured PerplexityResponse

        Raises:
            KeyError: If required fields are missing

        """
        choice = data["choices"][0]
        message = choice["message"]

        # Extract citations if present
        citations: list[Citation] = []
        if "citations" in data:
            for citation_url in data["citations"]:
                # Citations are just URLs - text will be empty
                citations.append(Citation(url=citation_url, text=""))

        return PerplexityResponse(
            id=data["id"],
            model=data["model"],
            content=message["content"],
            citations=citations,
            usage=data.get("usage", {}),
        )

    def query_batch(
        self,
        queries: list[str],
        config: PerplexityAPIConfig | None = None,
    ) -> list[PerplexityResponse]:
        """
        Execute multiple queries with caching.

        Args:
            queries: List of query strings
            config: API configuration (shared across all queries)

        Returns:
            List of responses corresponding to queries

        """
        responses = []
        for query in queries:
            try:
                response = self.query(query, config)
                responses.append(response)
            except PerplexityAPIError as e:
                console.print(
                    f"  [red]✗[/red] Query failed: {e}",
                    style="bold red",
                )
                # Continue with next query even if one fails
                continue

        return responses
