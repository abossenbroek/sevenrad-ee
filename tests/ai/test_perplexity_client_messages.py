"""
Unit tests for PerplexityClient message handling.

Tests the new chat_completion method that accepts either query or messages
parameters, with validation for mutual exclusivity and structured outputs.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from sevenrad_ee.ai.perplexity_client import PerplexityAPIError, PerplexityClient


class TestChatCompletionMessages:
    """Test chat_completion method with messages parameter."""

    @patch("sevenrad_ee.ai.perplexity_client.requests.post")
    def test_messages_parameter_success(
        self, mock_post: MagicMock
    ) -> None:
        """Test chat_completion with messages parameter works correctly."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test-id",
            "model": "llama-3.1-sonar-large-128k-online",
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"total_tokens": 10},
            "citations": [],
        }
        mock_post.return_value = mock_response

        # Create client and call with messages
        client = PerplexityClient(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        result = client.chat_completion(messages=messages)

        # Verify result
        assert result["choices"][0]["message"]["content"] == "Test response"
        assert result["model"] == "llama-3.1-sonar-large-128k-online"

        # Verify API was called with messages
        call_args = mock_post.call_args
        assert call_args is not None
        payload = call_args[1]["json"]
        assert payload["messages"] == messages

    @patch("sevenrad_ee.ai.perplexity_client.requests.post")
    def test_query_parameter_backward_compatibility(
        self, mock_post: MagicMock
    ) -> None:
        """Test query parameter still works (backward compatibility)."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test-id",
            "model": "llama-3.1-sonar-large-128k-online",
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"total_tokens": 10},
            "citations": [],
        }
        mock_post.return_value = mock_response

        # Create client and call with query
        client = PerplexityClient(api_key="test-key")
        result = client.chat_completion(query="Hello")

        # Verify result
        assert result["choices"][0]["message"]["content"] == "Test response"

        # Verify API was called with converted messages
        call_args = mock_post.call_args
        assert call_args is not None
        payload = call_args[1]["json"]
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]

    def test_mutually_exclusive_params(self) -> None:
        """Test query and messages are mutually exclusive."""
        client = PerplexityClient(api_key="test-key")

        with pytest.raises(ValueError, match="either 'query' or 'messages'"):
            client.chat_completion(
                query="Hello", messages=[{"role": "user", "content": "Hello"}]
            )

    def test_requires_one_param(self) -> None:
        """Test that at least one of query or messages is required."""
        client = PerplexityClient(api_key="test-key")

        with pytest.raises(ValueError, match="Must provide either"):
            client.chat_completion()

    @patch("sevenrad_ee.ai.perplexity_client.requests.post")
    def test_structured_output_support(
        self, mock_post: MagicMock
    ) -> None:
        """Test response_format parameter for structured outputs."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test-id",
            "model": "llama-3.1-sonar-large-128k-online",
            "choices": [{"message": {"content": '{"answer": "Test"}'}}],
            "usage": {"total_tokens": 10},
            "citations": [],
        }
        mock_post.return_value = mock_response

        # Create client and call with structured output
        client = PerplexityClient(api_key="test-key")
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "Answer", "schema": schema},
        }

        result = client.chat_completion(query="Question?", response_format=response_format)

        # Verify result
        content = result["choices"][0]["message"]["content"]
        data = json.loads(content)
        assert data["answer"] == "Test"

        # Verify API was called with response_format
        call_args = mock_post.call_args
        assert call_args is not None
        payload = call_args[1]["json"]
        assert payload["response_format"] == response_format

    @patch("sevenrad_ee.ai.perplexity_client.requests.post")
    def test_additional_kwargs_passed_through(
        self, mock_post: MagicMock
    ) -> None:
        """Test that additional kwargs are passed to API."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test-id",
            "model": "llama-3.1-sonar-large-128k-online",
            "choices": [{"message": {"content": "Test"}}],
            "usage": {"total_tokens": 10},
            "citations": [],
        }
        mock_post.return_value = mock_response

        # Create client and call with extra kwargs
        client = PerplexityClient(api_key="test-key")
        result = client.chat_completion(
            query="Test", temperature=0.5, max_tokens=100, top_p=0.9
        )

        # Verify kwargs were passed
        call_args = mock_post.call_args
        assert call_args is not None
        payload = call_args[1]["json"]
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 100
        assert payload["top_p"] == 0.9

    @patch("sevenrad_ee.ai.perplexity_client.requests.post")
    def test_api_error_handling(
        self, mock_post: MagicMock
    ) -> None:
        """Test that API errors are properly raised."""
        import requests

        # Setup mock to raise RequestException
        mock_post.side_effect = requests.exceptions.RequestException("API Error")

        # Create client and verify exception is raised
        client = PerplexityClient(api_key="test-key")
        with pytest.raises(PerplexityAPIError, match="API request failed"):
            client.chat_completion(query="Test")

    @patch("sevenrad_ee.ai.perplexity_client.requests.post")
    def test_model_override(
        self, mock_post: MagicMock
    ) -> None:
        """Test model parameter overrides default."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test-id",
            "model": "llama-3.1-sonar-small-128k-online",
            "choices": [{"message": {"content": "Test"}}],
            "usage": {"total_tokens": 10},
            "citations": [],
        }
        mock_post.return_value = mock_response

        # Create client and call with model override
        client = PerplexityClient(api_key="test-key")
        result = client.chat_completion(
            query="Test", model="llama-3.1-sonar-small-128k-online"
        )

        # Verify model was passed
        call_args = mock_post.call_args
        assert call_args is not None
        payload = call_args[1]["json"]
        assert payload["model"] == "llama-3.1-sonar-small-128k-online"
