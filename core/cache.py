"""Cache implementation for storing and retrieving data."""
from typing import Any, Optional
import json
import time
import aiofiles
import os

class Cache:
    """Simple cache implementation with file-based persistence."""

    def __init__(self, cache_file: str = "cache.json", ttl: int = 3600) -> None:
        """Initialize cache.
        
        Args:
            cache_file: Path to cache file
            ttl: Time to live in seconds for cache entries
        """
        self.cache_file = cache_file
        self.ttl = ttl
        self.cache = {}
        self._load_cache()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["value"]
            del self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = {
            "value": value,
            "timestamp": time.time()
        }
        self._save_cache()

    def remove(self, key: str) -> None:
        """Remove a specific key from the cache.
        
        Args:
            key: Cache key to remove
        """
        if key in self.cache:
            del self.cache[key]
            self._save_cache()

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache = {}
        self._save_cache()

    async def close(self) -> None:
        """Clean up resources."""
        await self._save_cache_async()

    def _load_cache(self) -> None:
        """Load cache from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
        except Exception:
            self.cache = {}

    def _save_cache(self) -> None:
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception:
            pass  # Fail silently on cache save errors

    async def _save_cache_async(self) -> None:
        """Save cache to file asynchronously."""
        try:
            async with aiofiles.open(self.cache_file, 'w') as f:
                await f.write(json.dumps(self.cache))
        except Exception:
            pass  # Fail silently on cache save errors
