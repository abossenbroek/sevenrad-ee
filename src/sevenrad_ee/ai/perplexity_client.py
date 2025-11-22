"""
Perplexity API client with SHA-256 caching and rate limiting.

This module provides a production-ready client for the Perplexity Sonar API with:
- Automatic SHA-256 caching to minimize API costs
- Rate limiting with configurable delays
- Structured error handling
- Rich console output for progress tracking
- Native structured output support via response_format parameter (Pydantic)
"""

import contextvars
import json
import logging
import os
import time
from typing import Any, TypeVar, overload

import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console

from sevenrad_ee.ai.perplexity_cache import (
    CacheManager,
    Citation,
    PerplexityAPIConfig,
    PerplexityResponse,
)

# Generic type for Pydantic models
T = TypeVar("T", bound=BaseModel)

# Load environment variables from .env file
load_dotenv()

console = Console()
logger = logging.getLogger(__name__)

# Context variables for request tracking (defined in logging_setup)
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="N/A"
)
phase_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "phase", default="setup"
)


class PerplexityAPIError(Exception):
    """Exception raised for Perplexity API errors with context."""

    def __init__(
        self,
        message: str,
        request_payload: dict[str, Any] | None = None,
        response_body: str | None = None,
        status_code: int | None = None,
    ) -> None:
        """
        Initialize API error with context.

        Args:
            message: Error message
            request_payload: Request payload sent to API
            response_body: Response body from API
            status_code: HTTP status code

        """
        super().__init__(message)
        self.request_payload = request_payload
        self.response_body = response_body
        self.status_code = status_code


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

    @overload
    def query(
        self,
        query: str,
        config: PerplexityAPIConfig | None = None,
        force_refresh: bool = False,
        response_model: None = None,
    ) -> PerplexityResponse: ...

    @overload
    def query(
        self,
        query: str,
        config: PerplexityAPIConfig | None = None,
        force_refresh: bool = False,
        *,
        response_model: type[T],
    ) -> T: ...

    def query(
        self,
        query: str,
        config: PerplexityAPIConfig | None = None,
        force_refresh: bool = False,
        response_model: type[T] | None = None,
    ) -> PerplexityResponse | T:
        """
        Execute Perplexity query with automatic caching and optional structured output.

        Args:
            query: Search query string
            config: API configuration (uses defaults if None)
            force_refresh: If True, bypass cache and fetch new response
            response_model: Optional Pydantic model class for structured output.
                          When provided, uses Perplexity's native response_format
                          parameter to enforce the schema and returns an instance
                          of the model. When None, returns PerplexityResponse.

        Returns:
            If response_model provided: Instance of the Pydantic model
            If response_model is None: PerplexityResponse with citations

        Raises:
            PerplexityAPIError: If API request fails

        Example:
            >>> from pydantic import BaseModel
            >>> class Analysis(BaseModel):
            ...     is_greenhouse: bool
            ...     confidence: float
            >>> client = PerplexityClient()
            >>> result = client.query(
            ...     "Is Marjoland a greenhouse?",
            ...     response_model=Analysis
            ... )
            >>> print(result.is_greenhouse)  # Typed access!

        """
        if config is None:
            config = PerplexityAPIConfig()

        # Generate cache key (include schema hash if using structured output)
        config_dict = config.model_dump()
        if response_model is not None:
            # Add schema to cache key to ensure different schemas get different caches
            schema = response_model.model_json_schema()
            config_dict["__schema_hash__"] = str(hash(str(schema)))

        cache_key = self.cache_manager.generate_cache_key(query, config_dict)

        # Check cache unless force refresh
        if not force_refresh:
            cached = self.cache_manager.get_cached_response(cache_key)
            if cached is not None:
                console.print(
                    f"  [dim][CACHE HIT][/dim] {query[:60]}...",
                    style="cyan",
                )
                # If using structured output, parse cached response into model
                if response_model is not None:
                    return self._parse_structured_response(
                        cached.content, response_model
                    )
                return cached

        # Execute API call
        console.print(
            f"  [dim][API CALL][/dim] {query[:60]}...",
            style="yellow",
        )

        try:
            response = self._execute_request(query, config_dict, response_model)

            # Cache the raw response (always PerplexityResponse)
            if isinstance(response, PerplexityResponse):
                self.cache_manager.save_response(cache_key, response)
            # If structured, response is already typed - cache the raw content
            # (This branch handles future enhancements)

            # Rate limiting courtesy delay
            time.sleep(self.rate_limit_delay)

            return response

        except requests.exceptions.RequestException as e:
            raise PerplexityAPIError(f"API request failed: {e}") from e

    def _execute_request(
        self,
        query: str,
        config_dict: dict[str, Any],
        response_model: type[T] | None = None,
    ) -> PerplexityResponse | T:
        """
        Execute API request and parse response.

        Args:
            query: Search query
            config_dict: Configuration dictionary
            response_model: Optional Pydantic model for structured output

        Returns:
            Parsed PerplexityResponse or typed Pydantic model instance

        Raises:
            PerplexityAPIError: If request fails or response is invalid

        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Filter out None values and internal keys from config
        filtered_config = {
            k: v
            for k, v in config_dict.items()
            if v is not None and not k.startswith("__")
        }

        payload: dict[str, Any] = {
            **filtered_config,
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ],
        }

        # Add response_format if using structured output
        if response_model is not None:
            schema = response_model.model_json_schema()
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": schema,
                },
            }

        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=60,  # Increased timeout for first schema compilation (10-30s)
        )

        response.raise_for_status()
        data = response.json()

        # Parse response based on mode
        try:
            if response_model is not None:
                # Structured output: parse into Pydantic model
                return self._parse_structured_response(
                    data["choices"][0]["message"]["content"],
                    response_model,
                )
            else:
                # Standard mode: return PerplexityResponse
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

    def _parse_structured_response(
        self,
        content: str,
        response_model: type[T],
    ) -> T:
        """
        Parse structured JSON response into Pydantic model.

        Args:
            content: JSON string from API response
            response_model: Pydantic model class to parse into

        Returns:
            Instance of response_model

        Raises:
            PerplexityAPIError: If JSON parsing or Pydantic validation fails

        """
        import json

        try:
            # Parse JSON string
            data = json.loads(content)
            # Validate and construct Pydantic model
            return response_model(**data)
        except json.JSONDecodeError as e:
            raise PerplexityAPIError(
                f"Failed to decode JSON from structured output: {e}\n"
                f"Content: {content[:200]}..."
            ) from e
        except Exception as e:
            raise PerplexityAPIError(
                f"Failed to validate structured output against {response_model.__name__}: {e}"
            ) from e

    def chat_completion(
        self,
        query: str | None = None,
        messages: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Perform chat completion with Perplexity API (DSPy-compatible).

        This is a lower-level method intended for DSPy integration. Unlike `query()`,
        it does NOT use caching and returns the raw API response dictionary.

        Args:
            query: Single query string (for backward compatibility)
            messages: OpenAI-compatible messages list (preferred for DSPy)
            model: Model name override (defaults to config default)
            response_format: Structured output schema (Perplexity native format)
            **kwargs: Additional API parameters (temperature, max_tokens, etc.)

        Returns:
            Raw API response dictionary with structure:
            {
                "id": "...",
                "model": "...",
                "choices": [{"message": {"content": "..."}}],
                "usage": {...},
                "citations": [...]
            }

        Raises:
            ValueError: If both query and messages provided, or neither
            PerplexityAPIError: If API request fails

        Example:
            >>> client = PerplexityClient()
            >>> # Using query parameter
            >>> response = client.chat_completion(query="What is Python?")
            >>> print(response["choices"][0]["message"]["content"])
            >>>
            >>> # Using messages parameter (DSPy style)
            >>> messages = [{"role": "user", "content": "What is Python?"}]
            >>> response = client.chat_completion(messages=messages)
            >>>
            >>> # With structured output
            >>> schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
            >>> response = client.chat_completion(
            ...     query="What is Python?",
            ...     response_format={"type": "json_schema", "json_schema": {"schema": schema}}
            ... )

        """
        request_id = request_id_var.get()

        logger.debug(
            f"chat_completion() called with query={query is not None}, "
            f"messages={messages is not None}, "
            f"response_format={response_format is not None}"
        )

        # Validation: ensure exactly one of query or messages is provided
        if query is not None and messages is not None:
            logger.error("Both query and messages provided")
            raise ValueError("Provide either 'query' or 'messages', not both")
        if query is None and messages is None:
            logger.error("Neither query nor messages provided")
            raise ValueError("Must provide either 'query' or 'messages'")

        # Convert query to messages format if needed
        if query is not None:
            logger.debug(
                f"Converting query to messages (query length: {len(query)} chars)"
            )
            messages = [{"role": "user", "content": query}]

        # Type narrowing: messages is guaranteed to be non-None here
        assert messages is not None, "messages should be set by now"
        logger.debug(f"Messages to API: {len(messages)} message(s)")

        # Build request payload
        assert self.api_key is not None, "API key should be validated in __init__"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": model or "llama-3.1-sonar-large-128k-online",
            "messages": messages,
            **kwargs,  # Include any additional parameters
        }

        logger.debug(f"Payload parameters: {list(payload.keys())}")

        # Add response_format if provided
        if response_format is not None:
            logger.debug("response_format provided:")
            logger.debug(f"  Type: {response_format.get('type', 'N/A')}")

            if "json_schema" in response_format:
                schema = response_format["json_schema"].get("schema", {})
                logger.debug(
                    f"  Schema name: {response_format['json_schema'].get('name', 'N/A')}"
                )
                logger.debug(
                    f"  Schema properties: {list(schema.get('properties', {}).keys())}"
                )
                logger.debug(f"  Schema required: {schema.get('required', [])}")

            payload["response_format"] = response_format

        # Log the complete final payload
        payload_json = json.dumps(payload, indent=2)
        logger.debug(f"FINAL API REQUEST PAYLOAD ({len(payload_json)} bytes):")
        logger.debug(payload_json)

        # Execute API request
        logger.info(f"Making API request to {self.api_url}")
        start_time = time.time()

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=60,
            )

            duration = time.time() - start_time

            logger.info(
                f"Response received in {duration:.2f}s (status: {response.status_code})"
            )

            response.raise_for_status()

            data: dict[str, Any] = response.json()
            logger.debug(f"Response keys: {list(data.keys())}")

            if "usage" in data:
                usage = data["usage"]
                logger.debug(
                    f"Token usage: "
                    f"prompt={usage.get('prompt_tokens', 'N/A')}, "
                    f"completion={usage.get('completion_tokens', 'N/A')}, "
                    f"total={usage.get('total_tokens', 'N/A')}"
                )

            logger.info("API request successful")
            return data

        except requests.exceptions.HTTPError as e:
            duration = time.time() - start_time

            logger.error(
                f"API REQUEST FAILED (HTTP {e.response.status_code}) after {duration:.2f}s"
            )

            error_body = e.response.text
            logger.error(f"Response body (raw): {error_body}")

            # Try to parse as JSON for readability
            try:
                error_json = json.loads(error_body)
                logger.error(
                    f"Response body (parsed): {json.dumps(error_json, indent=2)}"
                )
            except json.JSONDecodeError:
                pass

            logger.error(f"Response headers: {dict(e.response.headers)}")

            raise PerplexityAPIError(
                f"API request failed: {e}",
                request_payload=payload,
                response_body=error_body,
                status_code=e.response.status_code,
            ) from e

        except requests.exceptions.RequestException as e:
            duration = time.time() - start_time
            logger.error(f"REQUEST EXCEPTION after {duration:.2f}s: {type(e).__name__}")
            logger.exception("Network error details:")

            raise PerplexityAPIError(f"API request failed: {e}") from e

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
