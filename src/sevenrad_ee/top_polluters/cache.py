"""
Simple pickle-based caching system for API responses and EE results.

This module provides a basic file-based cache using pickle serialization.
Cache files are stored in the configured cache directory.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Any

from .config import settings


class Cache:
    """Simple file-based cache using pickle serialization."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store cache files. If None, uses settings.cache_dir.

        """
        self.cache_dir = cache_dir or settings.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """
        Generate cache file path from key.

        Args:
            key: Cache key string

        Returns:
            Path to the cache file

        """
        # Create a hash of the key for the filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{key_hash}.pkl"

    def get(self, key: str) -> Any | None:  # noqa: ANN401
        """
        Retrieve value from cache.

        Args:
            key: Cache key string

        Returns:
            Cached value if found, None otherwise

        """
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        try:
            with cache_path.open("rb") as f:
                return pickle.load(f)  # noqa: S301
        except (pickle.PickleError, EOFError, OSError):
            # If cache is corrupted, remove it and return None
            cache_path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any) -> None:  # noqa: ANN401
        """
        Store value in cache.

        Args:
            key: Cache key string
            value: Value to cache (must be picklable)

        """
        cache_path = self._get_cache_path(key)
        try:
            with cache_path.open("wb") as f:
                pickle.dump(value, f)
        except (pickle.PickleError, OSError) as e:
            # Log error but don't fail the operation
            # In production, this would use proper logging
            print(f"Warning: Failed to cache data: {e}")  # noqa: T201

    def clear(self) -> None:
        """Remove all cache files."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink(missing_ok=True)


# Global cache instance
cache = Cache()
