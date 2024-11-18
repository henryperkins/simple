"""
Enhanced Cache Management Module

This module provides advanced caching capabilities for docstrings and API responses,
with Redis-based distributed caching and in-memory fallback.

Version: 1.3.0
Author: Development Team
"""

import json
import time
import hashlib
from typing import Optional, Dict, Any, List, Union
import redis
from dataclasses import dataclass
import asyncio
from core.logger import log_info, log_error, log_debug
from docstring_utils import DocstringValidator  # Ensure this import is added

@dataclass
class CacheStats:
    """Statistics for cache operations."""
    hits: int = 0
    misses: int = 0
    errors: int = 0
    total_requests: int = 0
    cache_size: int = 0
    avg_response_time: float = 0.0

class Cache:
    """Enhanced cache management with Redis and in-memory fallback.

    Provides distributed caching with Redis and falls back to in-memory
    caching when Redis is unavailable.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 86400,
        max_retries: int = 3,
        max_memory_items: int = 1000,
    ):
        """Initializes the cache system with proper synchronization.

        Args:
            host (str): Redis host.
            port (int): Redis port.
            db (int): Redis database number.
            password (Optional[str]): Redis password.
            default_ttl (int): Default time-to-live for cache entries in seconds.
            max_retries (int): Maximum number of Redis connection attempts.
            max_memory_items (int): Maximum number of items in memory cache.
        """
        self.default_ttl = default_ttl
        self.max_retries = max_retries
        self.stats = CacheStats()
        self._stats_lock = asyncio.Lock()  # Lock for stats
        self._redis_lock = asyncio.Lock()  # Lock for Redis operations

        # Initialize Redis connection with proper error handling
        self.redis_available = False
        self._init_redis(host, port, db, password)

        # Initialize in-memory cache with synchronization
        self.memory_cache = LRUCache(max_size=max_memory_items)

    def _init_redis(self, host, port, db, password):
        """Initialize Redis connection with retries."""
        for attempt in range(self.max_retries):
            try:
                self.redis_client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=True,
                    retry_on_timeout=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                )
                self.redis_client.ping()
                self.redis_available = True
                log_info("Redis cache initialized successfully")
                break
            except redis.RedisError as e:
                log_error(f"Redis connection error (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    log_error("Falling back to in-memory cache only")
                time.sleep(2**attempt)

    def generate_cache_key(
        self,
        model_type: str,
        function_name: str,
        params: Dict[str, Any]
    ) -> str:
        """Generate cache key including model type."""
        key_components = [
            model_type,
            function_name,
            hashlib.md5(str(params).encode()).hexdigest()
        ]
        return ":".join(key_components)

    async def _update_stats(self, hit: bool = False, error: bool = False, response_time: float = 0.0):
        """Thread-safe update of cache statistics."""
        async with self._stats_lock:
            self.stats.total_requests += 1
            if hit:
                self.stats.hits += 1
            else:
                self.stats.misses += 1
            if error:
                self.stats.errors += 1

            # Update average response time thread-safely
            self.stats.avg_response_time = (
                (self.stats.avg_response_time * (self.stats.total_requests - 1) +
                 response_time) / self.stats.total_requests
            )

    async def get_cached_docstring(
        self, key: str, return_metadata: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Thread-safe retrieval of cached docstring.

        Args:
            key (str): Cache key.
            return_metadata (bool): Whether to return metadata with the docstring.

        Returns:
            Optional[Dict[str, Any]]: Cached data if found, otherwise None.
        """
        start_time = time.time()
        try:
            # Try Redis first with proper locking
            if self.redis_available:
                async with self._redis_lock:
                    value = await self._get_from_redis(key)
                    if value:
                        await self._update_stats(hit=True, response_time=time.time() - start_time)
                        return (
                            value if return_metadata else {"docstring": value["docstring"]}
                        )

            # Try memory cache with built-in synchronization
            value = await self.memory_cache.get(key)
            if value:
                await self._update_stats(hit=True, response_time=time.time() - start_time)
                return value if return_metadata else {"docstring": value["docstring"]}

            await self._update_stats(hit=False, response_time=time.time() - start_time)
            return None

        except Exception as e:
            log_error(f"Cache retrieval error: {e}")
            await self._update_stats(error=True, response_time=time.time() - start_time)
            return None

    async def save_docstring(
        self,
        key: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Save docstring with validation.

        Args:
            key (str): Cache key.
            data (Dict[str, Any]): Data to cache.
            ttl (Optional[int]): Time-to-live in seconds.
            tags (Optional[List[str]]): List of tags for the cache entry.

        Returns:
            bool: True if save was successful, otherwise False.
        """
        ttl = ttl or self.default_ttl
        try:
            # Validate docstring before caching
            validator = DocstringValidator()
            is_valid, validation_errors = validator.validate_docstring(data)
            
            if not is_valid:
                log_error(f"Invalid docstring not cached: {validation_errors}")
                return False

            cache_entry = {
                **data,
                "cache_metadata": {
                    "timestamp": time.time(),
                    "ttl": ttl,
                    "tags": tags or [],
                    "validation_status": {
                        "is_valid": True,
                        "validated_at": time.time()
                    }
                }
            }

            # Try Redis first
            if self.redis_available:
                success = await self._set_in_redis(key, cache_entry, ttl)
                if success:
                    return True

            # Fallback to memory cache
            await self.memory_cache.set(key, cache_entry, ttl)
            return True

        except Exception as e:
            log_error(f"Cache save error: {e}")
            return False

    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidates cache entries based on tags.

        Args:
            tags (List[str]): List of tags to match.

        Returns:
            int: Number of entries invalidated.
        """
        count = 0
        try:
            if self.redis_available:
                count += await self._invalidate_redis_by_tags(tags)
            count += await self.memory_cache.invalidate_by_tags(tags)
            return count
        except Exception as e:
            log_error(f"Tag-based invalidation error: {e}")
            return count

    async def _get_from_redis(self, key: str) -> Optional[Dict[str, Any]]:
        """Gets value from Redis.

        Args:
            key (str): Cache key.

        Returns:
            Optional[Dict[str, Any]]: Cached value if found, otherwise None.
        """
        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            log_error(f"Redis get error: {e}")
            return None

    async def _set_in_redis(self, key: str, value: Dict[str, Any], ttl: int) -> bool:
        """Sets value in Redis with tags.

        Args:
            key (str): Cache key.
            value (Dict[str, Any]): Value to cache.
            ttl (int): Time-to-live in seconds.

        Returns:
            bool: True if set was successful, otherwise False.
        """
        try:
            serialized = json.dumps(value)
            self.redis_client.setex(key, ttl, serialized)

            # Store tags
            if value.get("cache_metadata", {}).get("tags"):
                for tag in value["cache_metadata"]["tags"]:
                    tag_key = f"tag:{tag}"
                    self.redis_client.sadd(tag_key, key)
                    self.redis_client.expire(tag_key, ttl)

            return True
        except Exception as e:
            log_error(f"Redis set error: {e}")
            return False

    async def _invalidate_redis_by_tags(self, tags: List[str]) -> int:
        """Invalidates Redis entries based on tags.

        Args:
            tags (List[str]): List of tags to match.

        Returns:
            int: Number of entries invalidated.
        """
        count = 0
        try:
            for tag in tags:
                tag_key = f"tag:{tag}"
                keys = self.redis_client.smembers(tag_key)
                if keys:
                    count += len(keys)
                    for key in keys:
                        await self._delete_from_redis(key)
                    self.redis_client.delete(tag_key)
            return count
        except Exception as e:
            log_error(f"Redis tag invalidation error: {e}")
            return count

    async def _delete_from_redis(self, key: str) -> bool:
        """Deletes value and associated tags from Redis.

        Args:
            key (str): Key to delete.

        Returns:
            bool: True if deletion was successful, otherwise False.
        """
        try:
            # Get tags before deletion
            value = await self._get_from_redis(key)
            if value and value.get("cache_metadata", {}).get("tags"):
                for tag in value["cache_metadata"]["tags"]:
                    self.redis_client.srem(f"tag:{tag}", key)

            self.redis_client.delete(key)
            return True
        except Exception as e:
            log_error(f"Redis delete error: {e}")
            return False

    async def clear(self) -> bool:
        """Clears all cache entries.

        Returns:
            bool: True if clear was successful, otherwise False.
        """
        try:
            if self.redis_available:
                self.redis_client.flushdb()
            await self.memory_cache.clear()
            self.stats = CacheStats()
            return True
        except Exception as e:
            log_error(f"Cache clear error: {e}")
            return False

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Gets cache statistics.

        Returns:
            Dict[str, Union[int, float]]: Dictionary of cache statistics.
        """
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "errors": self.stats.errors,
            "total_requests": self.stats.total_requests,
            "hit_ratio": self.stats.hits / max(self.stats.total_requests, 1),
            "avg_response_time": self.stats.avg_response_time,
            "redis_available": self.redis_available,
        }

class LRUCache:
    """Thread-safe LRU cache implementation for in-memory caching.

    Attributes:
        max_size (int): Maximum number of items to store.
        cache (Dict[str, Dict[str, Any]]): In-memory cache storage.
        access_order (List[str]): Order of access for cache keys.
        lock (asyncio.Lock): Lock for thread-safe operations.
    """

    def __init__(self, max_size: int = 1000):
        """Initializes LRU cache.

        Args:
            max_size (int): Maximum number of items to store.
        """
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_order: List[str] = []
        self.lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Gets value from cache and updates access order.

        Args:
            key (str): Cache key.

        Returns:
            Optional[Any]: Cached value if found and not expired, otherwise None.
        """
        async with self.lock:
            if key not in self.cache:
                return None

            entry = self.cache[key]
            if self._is_expired(entry):
                await self.delete(key)
                return None

            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)

            return entry["value"]

    async def set(self, key: str, value: Any, ttl: int):
        """Sets value in cache with TTL.

        Args:
            key (str): Cache key.
            value (Any): Value to cache.
            ttl (int): Time-to-live in seconds.
        """
        async with self.lock:
            # Evict oldest item if cache is full
            if len(self.cache) >= self.max_size:
                await self._evict_oldest()

            entry = {"value": value, "expires_at": time.time() + ttl}

            self.cache[key] = entry
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

    async def delete(self, key: str):
        """Deletes value from cache.

        Args:
            key (str): Cache key to delete.
        """
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.access_order.remove(key)

    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidates cache entries by tags.

        Args:
            tags (List[str]): List of tags to match.

        Returns:
            int: Number of entries invalidated.
        """
        count = 0
        async with self.lock:
            keys_to_delete = []
            for key, entry in self.cache.items():
                if entry["value"].get("cache_metadata", {}).get("tags"):
                    if any(
                        tag in entry["value"]["cache_metadata"]["tags"] for tag in tags
                    ):
                        keys_to_delete.append(key)

            for key in keys_to_delete:
                await self.delete(key)
                count += 1

            return count

    async def clear(self):
        """Clears all cache entries."""
        async with self.lock:
            self.cache.clear()
            self.access_order.clear()

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Checks if cache entry is expired.

        Args:
            entry (Dict[str, Any]): Cache entry to check.

        Returns:
            bool: True if entry is expired, otherwise False.
        """
        return time.time() > entry["expires_at"]

    async def _evict_oldest(self):
        """Evicts oldest cache entry."""
        if self.access_order:
            oldest_key = self.access_order[0]
            await self.delete(oldest_key)

    def get_size(self) -> int:
        """Gets current cache size.

        Returns:
            int: Number of items in cache.
        """
        return len(self.cache)
