"""
Simplified cache management with Redis and in-memory fallback.
"""

import json
import time
from typing import Optional, Dict, Any
import redis
import asyncio
from core.logger import log_info, log_error, log_debug

class Cache:
    """Simple cache implementation with Redis and memory fallback."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 86400  # 24 hours
    ):
        """Initialize cache with Redis connection."""
        self.ttl = ttl
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.memory_timestamps: Dict[str, float] = {}
        
        # Try to initialize Redis
        try:
            self.redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True
            )
            self.redis.ping()
            self.has_redis = True
            log_info("Redis cache initialized")
        except Exception as e:
            log_error(f"Redis initialization failed: {e}")
            self.has_redis = False
            log_info("Using memory-only cache")

    async def get_cached_docstring(self, key: str) -> Optional[Dict[str, Any]]:
        """Get item from cache."""
        try:
            # Try Redis first if available
            if self.has_redis:
                data = self.redis.get(key)
                if data:
                    return json.loads(data)

            # Fallback to memory cache
            if key in self.memory_cache:
                if time.time() - self.memory_timestamps[key] < self.ttl:
                    return self.memory_cache[key]
                else:
                    # Clear expired entry
                    del self.memory_cache[key]
                    del self.memory_timestamps[key]

            return None

        except Exception as e:
            log_error(f"Cache retrieval error: {e}")
            return None

    async def save_docstring(self, key: str, data: Dict[str, Any]) -> bool:
        """Save item to cache."""
        try:
            json_data = json.dumps(data)
            
            # Try Redis first if available
            if self.has_redis:
                self.redis.setex(key, self.ttl, json_data)
                
            # Always save to memory cache as backup
            self.memory_cache[key] = data
            self.memory_timestamps[key] = time.time()
            
            return True

        except Exception as e:
            log_error(f"Cache save error: {e}")
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            if self.has_redis:
                self.redis.flushdb()
            self.memory_cache.clear()
            self.memory_timestamps.clear()
            log_info("Cache cleared")
        except Exception as e:
            log_error(f"Cache clear error: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "memory_cache_size": len(self.memory_cache),
            "has_redis": self.has_redis
        }
        
        if self.has_redis:
            try:
                stats["redis_keys"] = self.redis.dbsize()
            except Exception:
                stats["redis_keys"] = 0

        return stats

    async def close(self) -> None:
        """Close cache connections."""
        try:
            if self.has_redis:
                self.redis.close()
            self.memory_cache.clear()
            self.memory_timestamps.clear()
        except Exception as e:
            log_error(f"Cache close error: {e}")