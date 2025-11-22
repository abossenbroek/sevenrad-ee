"""
DSPy integration for Perplexity with native structured outputs.

This module provides a DSPy-compatible wrapper around PerplexityClient,
enabling the use of Perplexity's native structured outputs feature within
DSPy pipelines without relying on LiteLLM.

Architecture:
    User → DSPy Predictor → PerplexityLM → PerplexityClient → Perplexity API
           ^^^^^^^^^^^      ^^^^^^^^^^^^
           Formats prompts  Converts to messages

Key Design Principles:
    - PerplexityLM implements DSPy LM interface (not BaseLanguageModel)
    - DSPy predictors handle high-level prompt formatting
    - PerplexityLM converts prompts → messages for API
    - PerplexityClient handles raw API interaction
    - No LiteLLM dependency for structured outputs
"""

import contextvars
import json
import logging
from typing import Any

import dspy

from sevenrad_ee.ai.perplexity_client import PerplexityClient

logger = logging.getLogger(__name__)

# Context variables for request tracking
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="N/A"
)
phase_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "phase", default="setup"
)


class PerplexityLM(dspy.LM):  # type: ignore[misc]
    """
    DSPy language model wrapper for Perplexity API.

    Provides DSPy-compatible interface while using Perplexity's native
    structured outputs via direct API integration (bypassing LiteLLM).

    This class serves as an adapter between DSPy's prompt-based interface
    and Perplexity's message-based API, enabling seamless integration of
    Perplexity models with DSPy workflows.

    Example:
        >>> from sevenrad_ee.ai.dspy_perplexity import PerplexityLM
        >>> import dspy
        >>>
        >>> # Initialize LM
        >>> lm = PerplexityLM(model="llama-3.1-sonar-large-128k-online")
        >>> dspy.configure(lm=lm)
        >>>
        >>> # Use with DSPy predictor
        >>> class QA(dspy.Signature):
        ...     question: str = dspy.InputField()
        ...     answer: str = dspy.OutputField()
        >>>
        >>> predictor = dspy.Predict(QA)
        >>> response = predictor(question="What is Python?")
        >>> print(response.answer)
        >>>
        >>> # With structured outputs
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {"answer": {"type": "string"}},
        ...     "required": ["answer"]
        ... }
        >>> response = lm(
        ...     prompt="What is Python?",
        ...     response_format={"type": "json_schema", "json_schema": {"schema": schema}}
        ... )

    """

    def __init__(
        self,
        model: str = "llama-3.1-sonar-large-128k-online",
        api_key: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Perplexity LM for DSPy.

        Args:
            model: Perplexity model name (default: llama-3.1-sonar-large-128k-online)
            api_key: API key (reads from PERPLEXITY_API_KEY env var if not provided)
            **kwargs: Additional model parameters (temperature, max_tokens, etc.)
                     stored and passed to API calls

        Raises:
            ValueError: If API key not found in environment

        """
        super().__init__(model=model)
        self.model_name = model
        self.client = PerplexityClient(api_key=api_key)
        # Store model parameters to pass during API calls
        self.default_kwargs = kwargs

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Execute LM call with prompt or messages (DSPy interface).

        This method supports two calling patterns:
        1. With prompt (string): For backward compatibility and direct usage
        2. With messages (list): For DSPy adapters that format messages directly

        Args:
            prompt: Prompt string from DSPy predictor (optional, for backward compat)
            messages: Pre-formatted messages from DSPy adapter (optional)
            response_format: Optional structured output schema in Perplexity format:
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "SchemaName",
                        "schema": {...}  # JSON schema object
                    }
                }
            **kwargs: Additional API parameters (temperature, max_tokens, etc.)

        Returns:
            List of response dictionaries in DSPy format:
            [{"content": "response text or JSON string"}]

        Raises:
            PerplexityAPIError: If API request fails
            ValueError: If neither prompt nor messages provided

        """
        request_id = request_id_var.get()

        logger.debug("PerplexityLM.__call__() invoked")
        logger.debug(
            f"Input: prompt={'present' if prompt is not None else 'None'}, "
            f"messages={'present' if messages is not None else 'None'}"
        )

        # Handle both calling patterns
        if messages is None and prompt is None:
            logger.error("Neither prompt nor messages provided")
            raise ValueError("Must provide either prompt or messages")

        if messages is None:
            # Convert prompt to messages format (backward compat / direct usage)
            logger.debug(
                f"Converting prompt to messages (prompt length: {len(prompt) if prompt else 0} chars)"
            )
            messages = self._prompt_to_messages(prompt)  # type: ignore[arg-type]
            logger.debug(f"Converted to {len(messages)} message(s)")

        # Merge default kwargs with call-time kwargs (call-time takes precedence)
        api_kwargs = {**self.default_kwargs, **kwargs}
        logger.debug(f"API kwargs keys: {list(api_kwargs.keys())}")

        # Log response_format if provided
        if response_format is not None:
            logger.debug("response_format provided:")
            response_format_preview = json.dumps(response_format, indent=2)[:500]
            logger.debug(f"  Preview: {response_format_preview}")
            if "json_schema" in response_format:
                schema = response_format["json_schema"].get("schema", {})
                logger.debug(
                    f"  Schema properties: {list(schema.get('properties', {}).keys())}"
                )
                logger.debug(f"  Schema required fields: {schema.get('required', [])}")

        # Call Perplexity API with native structured outputs
        logger.info("Calling PerplexityClient.chat_completion()...")
        try:
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_name,
                response_format=response_format,
                **api_kwargs,
            )
            logger.info("PerplexityClient.chat_completion() successful")
            logger.debug(f"Response keys: {list(response.keys())}")
        except Exception as e:
            logger.error("PerplexityClient.chat_completion() FAILED")
            logger.error(f"Error type: {type(e).__name__}")
            logger.exception("Exception details:")
            raise

        # Convert to DSPy format
        logger.debug("Converting response to DSPy format")
        dspy_response = self._format_response(response)
        logger.debug(f"DSPy response: {len(dspy_response)} item(s)")

        return dspy_response

    def _prompt_to_messages(self, prompt: str) -> list[dict[str, str]]:
        """
        Convert DSPy prompt string to Perplexity messages format.

        DSPy predictors provide formatted prompts as strings. This method
        converts them to the OpenAI-compatible messages format expected by
        Perplexity's API.

        Args:
            prompt: Formatted prompt string from DSPy

        Returns:
            List of message dictionaries: [{"role": "user", "content": prompt}]

        """
        # Simple conversion: DSPy prompt → user message
        # More sophisticated implementations could parse multi-turn conversations
        return [{"role": "user", "content": prompt}]

    def _format_response(self, api_response: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Convert Perplexity API response to DSPy format.

        Args:
            api_response: Raw response from PerplexityClient.chat_completion()

        Returns:
            List of response dictionaries in DSPy format

        """
        # Extract content from API response
        content = api_response["choices"][0]["message"]["content"]

        # Return in DSPy expected format (DSPy expects 'text' key)
        return [{"text": content}]
