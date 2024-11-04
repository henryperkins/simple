# cache.py

import logging
from collections import OrderedDict
from typing import Dict, Any, Optional

class Cache:
    """LRU cache implementation for storing API responses."""
    
    def __init__(self, max_size_mb: int = 500):
        self._cache: OrderedDict = OrderedDict()
        self._max_size_mb = max_size_mb
        self._current_size_mb = 0
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve item from cache if it exists."""
        if key in self._cache:
            self._stats['hits'] += 1
            # Move to end to mark as recently used
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        self._stats['misses'] += 1
        return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Add item to cache with size tracking."""
        # Rough size estimation in MB
        size_mb = len(str(value)) / (1024 * 1024)
        
        # Evict items if needed
        while self._current_size_mb + size_mb > self._max_size_mb and self._cache:
            _, removed_value = self._cache.popitem(last=False)
            removed_size = len(str(removed_value)) / (1024 * 1024)
            self._current_size_mb -= removed_size
            self._stats['evictions'] += 1

        self._cache[key] = value
        self._current_size_mb += size_mb
        logging.debug(f"Added {key} to cache (size: {size_mb:.2f}MB)")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'evictions': self._stats['evictions'],
            'current_size_mb': self._current_size_mb,
            'max_size_mb': self._max_size_mb,
            'item_count': len(self._cache)
        }

# Global cache instance
prompt_cache = Cache()