"""    
Cache module for storing and retrieving AI-generated docstrings.    
Provides Redis-based caching functionality with connection management.    
"""    
    
import json    
from typing import Optional, Any, Dict    
    
from redis.asyncio import Redis, ConnectionError    
from exceptions import CacheError    
from core.logger import LoggerSetup    
    
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
        self.logger = LoggerSetup.get_logger(__name__)  # Initialize logger  
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
                decode_responses=True    
            )    
            await self._redis.ping()    
            self.logger.info("Successfully connected to Redis")    
        except ConnectionError as e:    
            self._redis = None    
            self.logger.error(f"Failed to connect to Redis: {e}")    
            raise CacheError("Failed to connect to Redis") from e    
    
    async def is_connected(self) -> bool:    
        """    
        Check if Redis connection is active.    
    
        Returns:    
            bool: True if connected, False otherwise.    
        """    
        if not self.enabled or not self._redis:    
            return False    
        try:    
            await self._redis.ping()    
            return True    
        except ConnectionError:    
            self.logger.warning("Redis connection lost")    
            self._redis = None    
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
    
        if not self._redis:    
            await self._initialize_connection()    
    
        if not self._redis:    
            self._stats['errors'] += 1    
            return None    
    
        cache_key = f"{self.prefix}{key}"    
        try:    
            cached_data = await self._redis.get(cache_key)    
    
            if cached_data:    
                self._stats['hits'] += 1    
                self.logger.debug(f"Cache hit for key: {cache_key}")    
                return json.loads(cached_data)    
    
            self._stats['misses'] += 1    
            self.logger.debug(f"Cache miss for key: {cache_key}")    
            return None    
    
        except Exception as e:    
            self._stats['errors'] += 1    
            self.logger.error(f"Cache get error for key {cache_key}: {e}")    
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
    
        if not self._redis:    
            await self._initialize_connection()    
    
        if not self._redis:    
            self._stats['errors'] += 1    
            return False    
    
        cache_key = f"{self.prefix}{key}"    
        serialized_data = json.dumps(data)    
        expiration = expire or self.ttl    
    
        try:    
            await self._redis.set(    
                cache_key,    
                serialized_data,    
                ex=expiration    
            )    
            self.logger.debug(f"Cached data for key: {cache_key}")    
            return True    
    
        except Exception as e:    
            self._stats['errors'] += 1    
            self.logger.error(f"Cache save error for key {cache_key}: {e}")    
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
    
        if not self._redis:    
            await self._initialize_connection()    
    
        if not self._redis:    
            self._stats['errors'] += 1    
            return False    
    
        cache_key = f"{self.prefix}{key}"    
        try:    
            await self._redis.delete(cache_key)    
            self.logger.debug(f"Invalidated cache key: {cache_key}")    
            return True    
    
        except Exception as e:    
            self._stats['errors'] += 1    
            self.logger.error(f"Cache invalidation error for key {cache_key}: {e}")    
            return False    
    
    async def get_stats(self) -> Dict[str, Any]:    
        """    
        Get cache statistics including hit rate and Redis info.    
    
        Returns:    
            Dict[str, Any]: Cache statistics and information.    
        """    
        if not self.enabled:    
            return {'enabled': False, 'stats': None}    
    
        if not self._redis:    
            await self._initialize_connection()    
    
        if not await self.is_connected():    
            self._stats['errors'] += 1    
            self.logger.error("Redis connection not available")    
            return {'enabled': True, 'stats': self._stats}    
    
        try:    
            info = await self._redis.info()    
            stats = {    
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
            return stats    
    
        except Exception as e:    
            self._stats['errors'] += 1    
            self.logger.error(f"Error getting cache stats: {e}")    
            return {'enabled': True, 'stats': self._stats}    
    
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
    
        if not self._redis:    
            await self._initialize_connection()    
    
        if not self._redis:    
            self._stats['errors'] += 1    
            return False    
    
        pattern = f"{self.prefix}*"    
        try:    
            # Use asynchronous scan and delete    
            async for key in self._redis.scan_iter(match=pattern):    
                await self._redis.delete(key)    
            self.logger.info("Cache cleared successfully")    
            return True    
    
        except Exception as e:    
            self._stats['errors'] += 1    
            self.logger.error(f"Error clearing cache: {e}")    
            return False    
    
    async def close(self) -> None:    
        """    
        Close Redis connection and perform cleanup.    
    
        Ensures that the Redis connection is properly closed asynchronously.    
        """    
        if self.enabled and self._redis:    
            try:    
                await self._redis.close()    
                self.logger.info("Redis connection closed")    
            except Exception as e:    
                self._stats['errors'] += 1    
                self.logger.error(f"Error closing Redis connection: {e}")    
            finally:    
                self._redis = None    
    
    async def __aenter__(self) -> "Cache":    
        """Async context manager entry."""    
        await self._initialize_connection()    
        return self    
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):    
        """Async context manager exit."""    
        await self.close()    