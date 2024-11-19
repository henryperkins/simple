# cache.py
"""
Cache module for storing and retrieving AI-generated docstrings.
Provides Redis-based caching functionality with connection management.
"""

import json
from typing import Optional, Any, Dict, Union, Tuple
import aioredis
from datetime import datetime, timedelta
from core.exceptions import CacheError
from core.logger import log_info, log_error, log_debug
from core.utils import generate_hash

class Cache:
    """Redis-based caching system for AI-generated docstrings."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        enabled: bool = True,
        ttl: int = 3600,
        prefix: str = "docstring:"
    ):
        """
        Initialize the cache with Redis connection parameters.

        Args:
            host: Redis host address
            port: Redis port number
            db: Redis database number
            password: Optional Redis password
            enabled: Whether caching is enabled
            ttl: Time-to-live for cache entries in seconds
            prefix: Prefix for cache keys
        """
        self.enabled = enabled
        if not self.enabled:
            log_info("Cache disabled")
            return

        self.redis_url = f"redis://{host}:{port}/{db}"
        self.password = password
        self.ttl = ttl
        self.prefix = prefix
        self.redis: Optional[aioredis.Redis] = None
        self._stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0
        }
        log_debug(f"Cache initialized with URL: {self.redis_url}")

    async def connect(self) -> None:
        """
        Establish connection to Redis server.

        Raises:
            CacheError: If connection fails
        """
        if not self.enabled:
            return

        try:
            self.redis = await aioredis.from_url(
                self.redis_url,
                password=self.password,
                decode_responses=True
            )
            log_info("Connected to Redis cache")
        except Exception as e:
            raise CacheError(f"Failed to connect to Redis: {str(e)}")

    async def test_connection(self) -> bool:
        """
        Test Redis connection by performing a ping.

        Returns:
            bool: True if connection is successful

        Raises:
            CacheError: If connection test fails
        """
        if not self.enabled:
            return False

        try:
            if not self.redis:
                await self.connect()
            await self.redis.ping()
            return True
        except Exception as e:
            raise CacheError(f"Redis connection test failed: {str(e)}")

    async def get_cached_docstring(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached docstring by key.

        Args:
            key: Cache key for the docstring

        Returns:
            Optional[Dict[str, Any]]: Cached docstring data if found, None otherwise
        """
        if not self.enabled:
            return None

        try:
            if not self.redis:
                await self.connect()

            cache_key = f"{self.prefix}{key}"
            cached_data = await self.redis.get(cache_key)

            if cached_data:
                self._stats['hits'] += 1
                log_debug(f"Cache hit for key: {cache_key}")
                return json.loads(cached_data)

            self._stats['misses'] += 1
            log_debug(f"Cache miss for key: {cache_key}")
            return None

        except Exception as e:
            self._stats['errors'] += 1
            log_error(f"Cache get error: {str(e)}")
            return None

    async def save_docstring(
        self,
        key: str,
        data: Dict[str, Any],
        expire: Optional[int] = None
    ) -> bool:
        """
        Save docstring data to cache.

        Args:
            key: Cache key for the docstring
            data: Docstring data to cache
            expire: Optional custom expiration time in seconds

        Returns:
            bool: True if save was successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            if not self.redis:
                await self.connect()

            cache_key = f"{self.prefix}{key}"
            serialized_data = json.dumps(data)
            expiration = expire or self.ttl

            await self.redis.set(
                cache_key,
                serialized_data,
                ex=expiration
            )
            log_debug(f"Cached data for key: {cache_key}")
            return True

        except Exception as e:
            self._stats['errors'] += 1
            log_error(f"Cache save error: {str(e)}")
            return False

    async def invalidate(self, key: str) -> bool:
        """
        Invalidate a cached entry.

        Args:
            key: Cache key to invalidate

        Returns:
            bool: True if invalidation was successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            if not self.redis:
                await self.connect()

            cache_key = f"{self.prefix}{key}"
            await self.redis.delete(cache_key)
            log_debug(f"Invalidated cache key: {cache_key}")
            return True

        except Exception as e:
            self._stats['errors'] += 1
            log_error(f"Cache invalidation error: {str(e)}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict[str, Any]: Dictionary containing cache statistics
        """
        if not self.enabled:
            return {
                'enabled': False,
                'stats': None
            }

        try:
            if not self.redis:
                await self.connect()

            info = await self.redis.info()
            return {
                'enabled': True,
                'stats': {
                    'hits': self._stats['hits'],
                    'misses': self._stats['misses'],
                    'errors': self._stats['errors'],
                    'hit_rate': self._calculate_hit_rate(),
                    'memory_used': info.get('used_memory_human', 'N/A'),
                    'connected_clients': info.get('connected_clients', 0),
                    'uptime_seconds': info.get('uptime_in_seconds', 0)
                }
            }

        except Exception as e:
            log_error(f"Error getting cache stats: {str(e)}")
            return {
                'enabled': True,
                'stats': self._stats
            }

    def _calculate_hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        Returns:
            float: Cache hit rate as a percentage
        """
        total = self._stats['hits'] + self._stats['misses']
        if total == 0:
            return 0.0
        return round((self._stats['hits'] / total) * 100, 2)

    async def clear(self) -> bool:
        """
        Clear all cached entries with the configured prefix.

        Returns:
            bool: True if clear was successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            if not self.redis:
                await self.connect()

            # Get all keys with prefix
            pattern = f"{self.prefix}*"
            cursor = 0
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern)
                if keys:
                    await self.redis.delete(*keys)
                if cursor == 0:
                    break

            log_info("Cache cleared successfully")
            return True

        except Exception as e:
            log_error(f"Error clearing cache: {str(e)}")
            return False

    async def close(self) -> None:
        """Close Redis connection and perform cleanup."""
        if self.enabled and self.redis:
            try:
                await self.redis.close()
                self.redis = None
                log_info("Redis connection closed")
            except Exception as e:
                log_error(f"Error closing Redis connection: {str(e)}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()