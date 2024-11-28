"""
Cache module for storing and retrieving AI-generated docstrings.
Provides Redis-based caching functionality with connection management.
"""

import json
import atexit
from typing import Optional, Any, Dict
from redis.asyncio import Redis
from exceptions import CacheError
from core.logger import LoggerSetup

# Initialize logger
logger = LoggerSetup.get_logger(__name__)

class Cache:
    """Redis-based caching system for AI-generated docstrings."""

    _instances = set()

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        enabled: bool = True,
        ttl: int = 3600,
        prefix: str = "docstring:"
    ) -> None:
        """
        Initialize the cache with Redis connection parameters.

        Args:
            host (str): Redis server host.
            port (int): Redis server port.
            db (int): Redis database number.
            password (Optional[str]): Password for Redis server.
            enabled (bool): Enable or disable the cache.
            ttl (int): Time-to-live for cache entries in seconds.
            prefix (str): Prefix for cache keys.
        """
        self.enabled = enabled
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.ttl = ttl
        self.prefix = prefix
        self._redis: Optional[Redis] = None
        self._stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0
        }
        # Register this instance for cleanup
        Cache._instances.add(self)

    @classmethod
    async def create(cls, **kwargs) -> 'Cache':
        """
        Create and initialize a new Cache instance.

        Returns:
            Cache: An initialized Cache instance.
        """
        cache = cls(**kwargs)
        if cache.enabled:
            await cache._initialize_connection()
        return cache

    async def _initialize_connection(self) -> None:
        """Initialize Redis connection."""
        try:
            self._redis = Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            if await self._redis.ping():
                logger.info("Successfully connected to Redis")
            else:
                raise CacheError("Redis ping failed")
        except Exception as e:
            self._redis = None
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise CacheError(f"Failed to connect to Redis: {str(e)}")

    async def is_connected(self) -> bool:
        """
        Check if Redis connection is active.

        Returns:
            bool: True if connected, False otherwise.
        """
        if not self.enabled or self._redis is None:
            return False
        try:
            return await self._redis.ping()
        except Exception as e:
            logger.error(f"Error checking Redis connection: {str(e)}")
            return False

    async def get_cached_docstring(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached docstring by key.

        Args:
            key (str): The cache key to retrieve.

        Returns:
            Optional[Dict[str, Any]]: Cached data if available, otherwise None.
        """
        if not self.enabled:
            return None

        try:
            if not self._redis:
                await self._initialize_connection()

            cache_key = f"{self.prefix}{key}"
            cached_data = await self._redis.get(cache_key)

            if cached_data:
                self._stats['hits'] += 1
                logger.debug(f"Cache hit for key: {cache_key}")
                return json.loads(cached_data)

            self._stats['misses'] += 1
            logger.debug(f"Cache miss for key: {cache_key}")
            return None

        except CacheError as e:
            self._stats['errors'] += 1
            logger.error(f"Cache get error: {str(e)}")
            return None
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Unexpected error in cache get: {str(e)}")
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
            key (str): The cache key.
            data (Dict[str, Any]): The data to cache.
            expire (Optional[int]): Optional expiration time in seconds.

        Returns:
            bool: True on success, False on failure.
        """
        if not self.enabled:
            return False

        try:
            if not self._redis:
                await self._initialize_connection()

            cache_key = f"{self.prefix}{key}"
            serialized_data = json.dumps(data)
            expiration = expire or self.ttl

            await self._redis.set(
                cache_key,
                serialized_data,
                ex=expiration
            )
            logger.debug(f"Cached data for key: {cache_key}")
            return True

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Cache save error: {str(e)}")
            return False

    async def invalidate(self, key: str) -> bool:
        """
        Invalidate a cached entry.

        Args:
            key (str): The cache key to invalidate.

        Returns:
            bool: True on success, False on failure.
        """
        if not self.enabled:
            return False

        try:
            if not self._redis:
                await self._initialize_connection()

            cache_key = f"{self.prefix}{key}"
            await self._redis.delete(cache_key)
            logger.debug(f"Invalidated cache key: {cache_key}")
            return True

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Cache invalidation error: {str(e)}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics including hit rate and Redis info.

        Returns:
            Dict[str, Any]: Cache statistics and information.
        """
        if not self.enabled:
            return {'enabled': False, 'stats': None}

        try:
            if not self._redis:
                await self._initialize_connection()

            if not await self.is_connected():
                raise CacheError("Redis connection not available")

            info = await self._redis.info()
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
            self._stats['errors'] += 1
            logger.error(f"Error getting cache stats: {str(e)}")
            return {
                'enabled': True,
                'stats': {
                    'hits': self._stats['hits'],
                    'misses': self._stats['misses'],
                    'errors': self._stats['errors'],
                    'hit_rate': self._calculate_hit_rate()
                }
            }

    def _calculate_hit_rate(self) -> float:
        """
        Calculate the cache hit rate.

        Returns:
            float: The hit rate as a percentage.
        """
        total = self._stats['hits'] + self._stats['misses']
        if total == 0:
            return 0.0
        return round((self._stats['hits'] / total) * 100, 2)

    async def clear(self) -> bool:
        """
        Clear all cached entries with the configured prefix.

        Returns:
            bool: True on success, False on failure.
        """
        if not self.enabled:
            return False

        try:
            if not self._redis:
                await self._initialize_connection()

            pattern = f"{self.prefix}*"
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(cursor, match=pattern)
                if keys:
                    await self._redis.delete(*keys)
                if cursor == 0:
                    break

            logger.info("Cache cleared successfully")
            return True

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error clearing cache: {str(e)}")
            return False

    def sync_close(self) -> None:
        """
        Synchronous cleanup for shutdown.

        Properly disconnects from Redis if enabled and connected.
        """
        if self.enabled and self._redis:
            try:
                self._redis.connection_pool.disconnect()
                self._redis = None
                logger.info("Redis connection closed synchronously")
            except Exception as e:
                logger.error(f"Error in sync close: {str(e)}")

    async def close(self) -> None:
        """
        Close Redis connection and perform cleanup.

        Ensures that the Redis connection is properly closed asynchronously.
        """
        if self.enabled and self._redis:
            try:
                await self._redis.close()
                self._redis = None
                logger.info("Redis connection closed")
            except Exception as e:
                self._stats['errors'] += 1
                logger.error(f"Error closing Redis connection: {str(e)}")
            finally:
                Cache._instances.discard(self)

    async def __aenter__(self) -> "Cache":
        """Async context manager entry."""
        await self._initialize_connection()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

def cleanup_redis() -> None:
    """Synchronous cleanup of all Redis connections at exit."""
    for cache in Cache._instances:
        cache.sync_close()

# Register cleanup handler
atexit.register(cleanup_redis)