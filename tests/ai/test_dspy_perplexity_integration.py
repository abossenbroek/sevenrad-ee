"""
Integration tests for PerplexityLM with DSPy.

Tests the full integration of PerplexityLM with DSPy predictors,
including structured outputs via native Perplexity response_format.
"""

from unittest.mock import MagicMock, patch

import dspy
import pytest

from sevenrad_ee.ai.dspy_perplexity import PerplexityLM


class TestPerplexityLMIntegration:
    """Test PerplexityLM integration with DSPy."""

    @patch("sevenrad_ee.ai.perplexity_client.requests.post")
    def test_basic_prediction(self, mock_post: MagicMock) -> None:
        """Test PerplexityLM works with basic DSPy predictor."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        # DSPy's JSONAdapter expects JSON string for structured output
        mock_response.json.return_value = {
            "id": "test-id",
            "model": "sonar-pro",
            "choices": [{"message": {"content": '{"answer": "Paris"}'}}],
            "usage": {"total_tokens": 20},
            "citations": [],
        }
        mock_post.return_value = mock_response

        # Create LM and configure DSPy
        lm = PerplexityLM(model="sonar-pro", api_key="test-key")
        dspy.configure(lm=lm)

        # Define simple signature
        class SimpleQA(dspy.Signature):  # type: ignore[misc]
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        # Create predictor and test
        predictor = dspy.Predict(SimpleQA)
        result = predictor(question="What is the capital of France?")

        # Verify result
        assert hasattr(result, "answer")
        assert len(result.answer) > 0

        # Verify API was called
        assert mock_post.called

    @patch("sevenrad_ee.ai.perplexity_client.requests.post")
    def test_chain_of_thought(self, mock_post: MagicMock) -> None:
        """Test PerplexityLM works with ChainOfThought predictor."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        # DSPy's JSONAdapter expects JSON string with all output fields
        mock_response.json.return_value = {
            "id": "test-id",
            "model": "sonar-pro",
            "choices": [
                {
                    "message": {
                        "content": '{"reasoning": "Let me think step by step. France is a country in Europe. '
                        'Paris has been its capital for centuries.", "answer": "Paris"}'
                    }
                }
            ],
            "usage": {"total_tokens": 50},
            "citations": [],
        }
        mock_post.return_value = mock_response

        # Create LM and configure DSPy
        lm = PerplexityLM(model="sonar-pro", api_key="test-key")
        dspy.configure(lm=lm)

        # Define signature
        class ReasonedQA(dspy.Signature):  # type: ignore[misc]
            question: str = dspy.InputField()
            reasoning: str = dspy.OutputField()
            answer: str = dspy.OutputField()

        # Create ChainOfThought predictor
        predictor = dspy.ChainOfThought(ReasonedQA)
        result = predictor(question="What is the capital of France?")

        # Verify result
        assert hasattr(result, "reasoning")
        assert hasattr(result, "answer")
        assert len(result.answer) > 0

    @patch("sevenrad_ee.ai.perplexity_client.requests.post")
    def test_prompt_to_messages_conversion(self, mock_post: MagicMock) -> None:
        """Test that prompts are correctly converted to messages."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test-id",
            "model": "sonar-pro",
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"total_tokens": 10},
            "citations": [],
        }
        mock_post.return_value = mock_response

        # Create LM and test direct call
        lm = PerplexityLM(model="sonar-pro", api_key="test-key")
        result = lm(prompt="What is 2+2?")

        # Verify result format
        assert isinstance(result, list)
        assert len(result) > 0
        assert "text" in result[0]

        # Verify API was called with messages
        call_args = mock_post.call_args
        assert call_args is not None
        payload = call_args[1]["json"]
        assert "messages" in payload
        assert payload["messages"][0]["role"] == "user"
        assert "2+2" in payload["messages"][0]["content"]

    @patch("sevenrad_ee.ai.perplexity_client.requests.post")
    def test_structured_output_integration(self, mock_post: MagicMock) -> None:
        """Test PerplexityLM with structured outputs via response_format."""
        # Setup mock response with structured JSON
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test-id",
            "model": "sonar-pro",
            "choices": [
                {"message": {"content": '{"answer": "Paris", "confidence": 0.95}'}}
            ],
            "usage": {"total_tokens": 15},
            "citations": [],
        }
        mock_post.return_value = mock_response

        # Create LM and test with response_format
        lm = PerplexityLM(model="sonar-pro", api_key="test-key")
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer", "confidence"],
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "Answer", "schema": schema},
        }

        result = lm(prompt="What is the capital of France?", response_format=response_format)

        # Verify result
        assert isinstance(result, list)
        assert len(result) > 0

        # Verify response_format was passed to API
        call_args = mock_post.call_args
        assert call_args is not None
        payload = call_args[1]["json"]
        assert "response_format" in payload
        assert payload["response_format"] == response_format

    @patch("sevenrad_ee.ai.perplexity_client.requests.post")
    def test_multiple_predictions(self, mock_post: MagicMock) -> None:
        """Test multiple predictions with same LM instance."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        # DSPy's JSONAdapter expects JSON string for structured output
        mock_response.json.return_value = {
            "id": "test-id",
            "model": "sonar-pro",
            "choices": [{"message": {"content": '{"answer": "Test response"}'}}],
            "usage": {"total_tokens": 10},
            "citations": [],
        }
        mock_post.return_value = mock_response

        # Create LM
        lm = PerplexityLM(model="sonar-pro", api_key="test-key")
        dspy.configure(lm=lm)

        # Define signature
        class SimpleQA(dspy.Signature):  # type: ignore[misc]
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        # Create predictor
        predictor = dspy.Predict(SimpleQA)

        # Make multiple predictions
        result1 = predictor(question="Question 1?")
        result2 = predictor(question="Question 2?")
        result3 = predictor(question="Question 3?")

        # Verify all worked
        assert hasattr(result1, "answer")
        assert hasattr(result2, "answer")
        assert hasattr(result3, "answer")

        # Verify API was called (DSPy may make multiple attempts per prediction via adapter fallback)
        assert mock_post.call_count >= 3

    @patch("sevenrad_ee.ai.perplexity_client.requests.post")
    def test_error_propagation(self, mock_post: MagicMock) -> None:
        """Test that API errors are properly propagated."""
        # Setup mock to raise exception
        mock_post.side_effect = Exception("API Error")

        # Create LM and test
        lm = PerplexityLM(model="sonar-pro", api_key="test-key")

        # Verify exception is raised
        with pytest.raises(Exception, match="API Error"):
            lm(prompt="Test")

    def test_api_key_from_environment(self) -> None:
        """Test that API key can be read from environment."""
        import os
        from unittest.mock import patch

        # Mock environment variable
        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": "env-key"}):
            lm = PerplexityLM(model="sonar-pro")
            assert lm.client.api_key == "env-key"

    def test_api_key_required(self) -> None:
        """Test that ValueError is raised if API key not found."""
        import os
        from unittest.mock import patch

        # Mock empty environment
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="PERPLEXITY_API_KEY"):
                PerplexityLM(model="sonar-pro")
