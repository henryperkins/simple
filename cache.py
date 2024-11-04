# cache.py

import logging
from collections import OrderedDict
from threading import Lock
from typing import Dict, Any, Optional

class ThreadSafeCache:
    """Thread-safe LRU cache implementation for storing data."""

    def __init__(self, max_size_mb: int = 500):
        """Initialize the cache with a maximum size.

        Args:
            max_size_mb (int): Maximum size of the cache in megabytes.
        """
        self._cache: OrderedDict = OrderedDict()
        self._max_size_mb = max_size_mb
        self._current_size_mb = 0
        self._lock = Lock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve an item from the cache if it exists.

        Args:
            key (str): The key of the item to retrieve.

        Returns:
            Optional[Dict[str, Any]]: The cached item if it exists, otherwise None.
        """
        with self._lock:
            if key in self._cache:
                self._stats['hits'] += 1
                # Move to end to mark as recently used
                value = self._cache.pop(key)
                self._cache[key] = value
                return value
            self._stats['misses'] += 1
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Add an item to the cache with size tracking.

        Args:
            key (str): The key of the item to add.
            value (Dict[str, Any]): The item to add to the cache.
        """
        with self._lock:
            size_mb = self._calculate_size(value)
            self._ensure_capacity(size_mb)
            self._cache[key] = value
            self._current_size_mb += size_mb
            logging.debug(f"Added {key} to cache (size: {size_mb:.2f}MB)")

    def _calculate_size(self, value: Dict) -> float:
        """Calculate the approximate size of a cache item in MB.

        Args:
            value (Dict): The item to calculate the size of.

        Returns:
            float: The size of the item in megabytes.
        """
        import sys
        return sys.getsizeof(str(value)) / (1024 * 1024)

    def _ensure_capacity(self, needed_size: float) -> None:
        """Ensure the cache has capacity for new items.

        Args:
            needed_size (float): The size of the new item to add.

        Raises:
            RuntimeError: If the cache cannot accommodate the new item.
        """
        while (self._current_size_mb + needed_size) > self._max_size_mb and self._cache:
            _, removed_value = self._cache.popitem(last=False)
            removed_size = self._calculate_size(removed_value)
            self._current_size_mb -= removed_size
            self._stats['evictions'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict[str, Any]: A dictionary containing cache statistics such as hits, misses, evictions, current size, and item count.
        """
        return {
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'evictions': self._stats['evictions'],
            'current_size_mb': self._current_size_mb,
            'max_size_mb': self._max_size_mb,
            'item_count': len(self._cache)
        }
