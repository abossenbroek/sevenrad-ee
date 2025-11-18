"""
Unit tests for CachedRetriever module.

Tests verify:
- Deterministic behavior (same input → same output)
- Proper data loading from JSON cache files
- Isolation from network calls
- Error handling for missing/malformed data
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from sevenrad_ee.ai.retrievers import CachedRetriever


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """
    Create temporary cache directory with sample JSON files.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Path to temporary cache directory

    """
    cache_dir = tmp_path / "research"
    cache_dir.mkdir()

    # Create sample cached research file
    sample_data = {
        "company": "Test Kwekerij BV",
        "location": "Amsterdam",
        "timestamp": "2025-11-18T10:00:00",
        "queries": [
            {
                "query": '"Test Kwekerij BV" Amsterdam AND assimilatiebelichting',
                "content": "Test company uses LED grow lights for tomato cultivation.",
                "citations": ["https://example.com/article1", "https://example.com/article2"],
            },
            {
                "query": '"Test Kwekerij BV" AND Signify',
                "content": "Partnership with Signify for LED installation confirmed.",
                "citations": ["https://example.com/article3"],
            },
        ],
    }

    cache_file = cache_dir / "Test_Kwekerij_BV_Amsterdam.json"
    cache_file.write_text(json.dumps(sample_data, indent=2))

    return cache_dir


def test_cached_retriever_loads_data(temp_cache_dir: Path) -> None:
    """
    Verify CachedRetriever loads JSON data correctly at initialization.

    Tests eager loading behavior and proper cache structure.
    """
    retriever = CachedRetriever(cache_dir=temp_cache_dir)

    # Verify cache is populated
    assert len(retriever.cache) == 1, "Should load one company from cache"

    # Verify key format
    expected_key = "Test Kwekerij BV_Amsterdam"
    assert expected_key in retriever.cache, f"Cache should contain key: {expected_key}"

    # Verify content is loaded
    cached_content = retriever.cache[expected_key]
    assert "LED grow lights" in cached_content, "Cache should contain query content"
    assert "Signify" in cached_content, "Cache should contain all queries"


def test_cached_retriever_forward_returns_correct_format(temp_cache_dir: Path) -> None:
    """
    Verify forward() returns correctly formatted evidence string.

    Tests output structure matches expected format for DSPy integration.
    """
    retriever = CachedRetriever(cache_dir=temp_cache_dir)

    # Call forward with known company
    evidence = retriever.forward(
        company_name="Test Kwekerij BV",
        location="Amsterdam",
    )

    # Verify evidence is a string
    assert isinstance(evidence, str), "Evidence should be a string"

    # Verify content structure
    assert "Query:" in evidence, "Evidence should include query labels"
    assert "Response:" in evidence, "Evidence should include response labels"
    assert "Sources:" in evidence, "Evidence should include source labels"

    # Verify actual content is present
    assert "LED grow lights" in evidence, "Evidence should contain query 1 content"
    assert "Signify" in evidence, "Evidence should contain query 2 content"

    # Verify citations are included
    assert "https://example.com/article1" in evidence, "Evidence should include citations"


def test_cached_retriever_deterministic_behavior(temp_cache_dir: Path) -> None:
    """
    Verify CachedRetriever produces deterministic output.

    Critical test: Same input must always produce identical output for
    reproducible optimization.
    """
    retriever = CachedRetriever(cache_dir=temp_cache_dir)

    # Call forward multiple times with same input
    evidence1 = retriever.forward(
        company_name="Test Kwekerij BV",
        location="Amsterdam",
    )
    evidence2 = retriever.forward(
        company_name="Test Kwekerij BV",
        location="Amsterdam",
    )
    evidence3 = retriever.forward(
        company_name="Test Kwekerij BV",
        location="Amsterdam",
    )

    # Verify all outputs are identical
    assert evidence1 == evidence2, "Repeated calls should return identical results"
    assert evidence2 == evidence3, "Determinism should hold across multiple calls"


@patch("sevenrad_ee.ai.perplexity_client.PerplexityClient")
def test_cached_retriever_no_network_calls(
    mock_perplexity_client: Mock,
    temp_cache_dir: Path,
) -> None:
    """
    Verify CachedRetriever makes NO network calls.

    Critical test: Ensures complete isolation from Perplexity API during
    cached retrieval for cost control and deterministic behavior.
    """
    retriever = CachedRetriever(cache_dir=temp_cache_dir)

    # Execute forward pass
    evidence = retriever.forward(
        company_name="Test Kwekerij BV",
        location="Amsterdam",
    )

    # Verify Perplexity client was never instantiated or called
    mock_perplexity_client.assert_not_called()

    # Verify we still got valid evidence (from cache)
    assert evidence is not None, "Should return cached evidence"
    assert len(evidence) > 0, "Cached evidence should not be empty"


def test_cached_retriever_missing_cache_dir() -> None:
    """
    Verify CachedRetriever raises FileNotFoundError for missing cache directory.

    Tests fail-fast error handling at initialization.
    """
    with pytest.raises(FileNotFoundError, match="Cache directory not found"):
        CachedRetriever(cache_dir="/nonexistent/path")


def test_cached_retriever_empty_cache_dir(tmp_path: Path) -> None:
    """
    Verify CachedRetriever raises FileNotFoundError for empty cache directory.

    Tests fail-fast error handling when no JSON files are present.
    """
    empty_dir = tmp_path / "empty_cache"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No cached research files found"):
        CachedRetriever(cache_dir=empty_dir)


def test_cached_retriever_malformed_json(tmp_path: Path) -> None:
    """
    Verify CachedRetriever raises ValueError for malformed JSON files.

    Tests error handling for corrupted cache data.
    """
    cache_dir = tmp_path / "bad_cache"
    cache_dir.mkdir()

    # Create file with invalid JSON
    bad_file = cache_dir / "bad_data.json"
    bad_file.write_text("{ this is not valid JSON }")

    with pytest.raises(ValueError, match="Failed to decode JSON"):
        CachedRetriever(cache_dir=cache_dir)


def test_cached_retriever_missing_company_data(temp_cache_dir: Path) -> None:
    """
    Verify CachedRetriever handles requests for non-existent companies gracefully.

    Tests runtime behavior when querying company not in cache.
    """
    retriever = CachedRetriever(cache_dir=temp_cache_dir)

    # Request evidence for company not in cache
    evidence = retriever.forward(
        company_name="Unknown Company",
        location="Nowhere",
    )

    # Should return helpful message, not crash
    assert "No cached data available" in evidence, "Should provide helpful message for missing data"
