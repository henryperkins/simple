"""    
Cache module for storing and retrieving AI-generated docstrings.    
Provides Redis-based caching functionality with connection management.    
"""

import json
from typing import Optional, Any, Dict
from redis.asyncio import Redis, ConnectionError
from exceptions import CacheError
from core.logger import LoggerSetup, CorrelationLoggerAdapter

# Setup logging with correlation ID
logger = LoggerSetup.get_logger(__name__)
adapter = CorrelationLoggerAdapter(logger)

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
        prefix: str = "docstring:",
    ) -> None:
        """
        Initialize the cache with Redis connection parameters.
        """
        self.logger = LoggerSetup.get_logger(__name__)
        self.adapter = CorrelationLoggerAdapter(self.logger)
        self.correlation_id = self.adapter.correlation_id
        self.enabled = enabled
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.ttl = ttl
        self.prefix = prefix
        self._redis: Optional[Redis] = None
        self._stats = {"hits": 0, "misses": 0, "errors": 0}

    async def _initialize_connection(self) -> None:
        """Initialize Redis connection."""
        if not self.enabled or self._redis:
            return
        try:
            self._redis = Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
            )
            await self._redis.ping()
            self.adapter.info("Successfully connected to Redis", extra={'correlation_id': self.correlation_id})
        except ConnectionError as e:
            self._redis = None
            self.adapter.error(f"Failed to connect to Redis: {e}", exc_info=True, extra={'correlation_id': self.correlation_id})
            raise CacheError("Failed to connect to Redis") from e

    async def is_connected(self) -> bool:
        """
        Check if Redis connection is active.
        """
        if not self.enabled or not self._redis:
            return False
        try:
            await self._redis.ping()
            return True
        except ConnectionError:
            self.adapter.warning("Redis connection lost", extra={'correlation_id': self.correlation_id})
            self._redis = None
            return False

    async def get_cached_docstring(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached docstring by key.
        """
        if not self.enabled:
            return None

        if not self._redis:
            await self._initialize_connection()

        if not self._redis:
            self._stats["errors"] += 1
            return None

        cache_key = f"{self.prefix}{key}"
        try:
            cached_data = await self._redis.get(cache_key)

            if cached_data:
                self._stats["hits"] += 1
                self.adapter.debug(f"Cache hit for key: {cache_key}", extra={'correlation_id': self.correlation_id})
                return json.loads(cached_data)

            self._stats["misses"] += 1
            self.adapter.debug(f"Cache miss for key: {cache_key}", extra={'correlation_id': self.correlation_id})
            return None

        except Exception as e:
            self._stats["errors"] += 1
            self.adapter.error(f"Cache get error for key {cache_key}: {e}", exc_info=True, extra={'correlation_id': self.correlation_id})
            return None

    async def save_docstring(
        self, key: str, data: Dict[str, Any], expire: Optional[int] = None
    ) -> bool:
        """
        Save docstring data to cache.
        """
        if not self.enabled:
            return False

        if not self._redis:
            await self._initialize_connection()

        if not self._redis:
            self._stats["errors"] += 1
            return False

        cache_key = f"{self.prefix}{key}"
        serialized_data = json.dumps(data)
        expiration = expire or self.ttl

        try:
            await self._redis.set(cache_key, serialized_data, ex=expiration)
            self.adapter.debug(f"Cached data for key: {cache_key}", extra={'correlation_id': self.correlation_id})
            return True

        except Exception as e:
            self._stats["errors"] += 1
            self.adapter.error(f"Cache save error for key {cache_key}: {e}", exc_info=True, extra={'correlation_id': self.correlation_id})
            return False

    async def invalidate(self, key: str) -> bool:
        """
        Invalidate a cached entry.
        """
        if not self.enabled:
            return False

        if not self._redis:
            await self._initialize_connection()

        if not self._redis:
            self._stats["errors"] += 1
            return False

        cache_key = f"{self.prefix}{key}"
        try:
            await self._redis.delete(cache_key)
            self.adapter.debug(f"Invalidated cache key: {cache_key}", extra={'correlation_id': self.correlation_id})
            return True

        except Exception as e:
            self._stats["errors"] += 1
            self.adapter.error(f"Cache invalidation error for key {cache_key}: {e}", exc_info=True, extra={'correlation_id': self.correlation_id})
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        """
        if not self.enabled:
            return {"enabled": False, "stats": None}

        try:
            if not self._redis:
                await self._initialize_connection()

            try:
                info = await self._redis.info()
            except Exception as e:
                self.adapter.error(f"Failed to get Redis info: {e}", extra={'correlation_id': self.correlation_id})
                info = {}

            stats = {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "errors": self._stats["errors"],
                "hit_rate": self._calculate_hit_rate(),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
            }

            if info:
                stats.update(
                    {
                        "memory_used": info.get("used_memory_human", "N/A"),
                        "connected_clients": info.get("connected_clients", 0),
                        "total_connections_received": info.get("total_connections_received", 0),
                        "total_commands_processed": info.get("total_commands_processed", 0),
                    }
                )

            self.adapter.info(f"Cache stats: {stats}", extra={'correlation_id': self.correlation_id})
            return {"enabled": True, "stats": stats}

        except Exception as e:
            self.adapter.error(f"Error getting cache stats: {str(e)}", extra={'correlation_id': self.correlation_id})
            return {"enabled": True, "stats": self._stats}

    def _calculate_hit_rate(self) -> float:
        """
        Calculate the cache hit rate.
        """
        total = self._stats["hits"] + self._stats["misses"]
        if total == 0:
            return 0.0
        hit_rate = round((self._stats["hits"] / total) * 100, 2)
        self.adapter.debug(f"Calculated hit rate: {hit_rate}", extra={'correlation_id': self.correlation_id})
        return hit_rate

    async def clear(self) -> bool:
        """
        Clear all cached entries with the configured prefix.
        """
        if not self.enabled:
            return False

        if not self._redis:
            await self._initialize_connection()

        if not self._redis:
            self._stats["errors"] += 1
            return False

        pattern = f"{self.prefix}*"
        try:
            async for key in self._redis.scan_iter(match=pattern):
                await self._redis.delete(key)
            self.adapter.info("Cache cleared successfully", extra={'correlation_id': self.correlation_id})
            return True

        except Exception as e:
            self._stats["errors"] += 1
            self.adapter.error(f"Error clearing cache: {e}", exc_info=True, extra={'correlation_id': self.correlation_id})
            return False

    async def close(self) -> None:
        """
        Close Redis connection and perform cleanup.
        """
        if self.enabled and self._redis:
            try:
                await self._redis.close()
                self.adapter.info("Redis connection closed", extra={'correlation_id': self.correlation_id})
            except Exception as e:
                self._stats["errors"] += 1
                self.adapter.error(f"Error closing Redis connection: {e}", exc_info=True, extra={'correlation_id': self.correlation_id})
            finally:
                self._redis = None

    async def __aenter__(self) -> "Cache":
        """Async context manager entry."""
        await self._initialize_connection()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()