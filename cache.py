# cache.py

import os
import json
import time
import hashlib
import threading
from collections import OrderedDict
from typing import Any, Dict
from dataclasses import dataclass
from core.logger import LoggerSetup
import sentry_sdk

logger = LoggerSetup.get_logger("cache")

@dataclass
class CacheConfig:
    """Configuration for the caching system."""
    dir: str = "cache"
    max_size_mb: int = 500
    index_file: str = "index.json"

    @property
    def index_path(self) -> str:
        """Get the full path to the index file."""
        return os.path.join(self.dir, self.index_file)

class CacheManager:
    """Manages file-based caching operations."""
    
    def __init__(self, config: CacheConfig):
        """
        Initialize the cache manager.

        Args:
            config (CacheConfig): Configuration for the cache system
        """
        self.config = config
        self._lock = threading.Lock()
        self.initialize_cache()

    def initialize_cache(self) -> None:
        """Initialize cache directory and index file."""
        try:
            os.makedirs(self.config.dir, exist_ok=True)
            if not os.path.exists(self.config.index_path):
                with open(self.config.index_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f)
            logger.info("Cache initialized successfully")
        except OSError as e:
            logger.error(f"Failed to initialize cache: {e}")
            sentry_sdk.capture_exception(e)
            raise

    def get_cache_path(self, key: str) -> str:
        """
        Generate cache file path from key.

        Args:
            key (str): The cache key

        Returns:
            str: The path to the cache file
        """
        hashed_key = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join(self.config.dir, f"{hashed_key}.json")

    def load_index(self) -> OrderedDict:
        """
        Load and sort cache index.

        Returns:
            OrderedDict: The sorted cache index
        """
        with self._lock:
            try:
                with open(self.config.index_path, 'r', encoding='utf-8') as f:
                    index = json.load(f, object_pairs_hook=OrderedDict)
                return OrderedDict(sorted(index.items(), 
                                       key=lambda item: item[1]['last_access_time']))
            except Exception as e:
                logger.error(f"Failed to load cache index: {e}")
                sentry_sdk.capture_exception(e)
                return OrderedDict()

    def save_index(self, index: OrderedDict) -> None:
        """
        Save cache index to disk.

        Args:
            index (OrderedDict): The cache index to save
        """
        with self._lock:
            try:
                with open(self.config.index_path, 'w', encoding='utf-8') as f:
                    json.dump(index, f)
            except Exception as e:
                logger.error(f"Failed to save cache index: {e}")
                sentry_sdk.capture_exception(e)
                raise

    def cache_response(self, key: str, data: Dict[str, Any]) -> None:
        """
        Cache response data.

        Args:
            key (str): The cache key
            data (Dict[str, Any]): The data to cache
        """
        index = self.load_index()
        cache_path = self.get_cache_path(key)
        
        with self._lock:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
                index[key] = {
                    'cache_path': cache_path,
                    'last_access_time': time.time()
                }
                self.save_index(index)
                self.clear_old_entries(index)
            except Exception as e:
                logger.error(f"Failed to cache response: {e}")
                sentry_sdk.capture_exception(e)
                raise

    def get_cached_response(self, key: str) -> Dict[str, Any]:
        """
        Retrieve cached response.

        Args:
            key (str): The cache key

        Returns:
            Dict[str, Any]: The cached data or empty dict if not found
        """
        index = self.load_index()
        
        with self._lock:
            cache_entry = index.get(key)
            if not cache_entry:
                return {}

            cache_path = cache_entry.get('cache_path')
            if not cache_path or not os.path.exists(cache_path):
                del index[key]
                self.save_index(index)
                return {}

            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                cache_entry['last_access_time'] = time.time()
                index.move_to_end(key)
                self.save_index(index)
                return data
            except Exception as e:
                logger.error(f"Failed to retrieve cached response: {e}")
                sentry_sdk.capture_exception(e)
                return {}

    def clear_old_entries(self, index: OrderedDict) -> None:
        """
        Clear old cache entries if size limit exceeded.

        Args:
            index (OrderedDict): The current cache index
        """
        total_size = sum(
            os.path.getsize(entry['cache_path'])
            for entry in index.values()
            if os.path.exists(entry['cache_path'])
        )
        
        total_size_mb = total_size / (1024 * 1024)
        while total_size_mb > self.config.max_size_mb and index:
            key, entry = index.popitem(last=False)
            cache_path = entry['cache_path']
            if os.path.exists(cache_path):
                try:
                    file_size = os.path.getsize(cache_path)
                    os.remove(cache_path)
                    total_size_mb -= file_size / (1024 * 1024)
                except OSError as e:
                    logger.error(f"Failed to remove cache file: {e}")
                    sentry_sdk.capture_exception(e)

        self.save_index(index)

def create_cache_manager(cache_dir: str = "cache", 
                        max_cache_size_mb: int = 500) -> CacheManager:
    """
    Factory function to create a CacheManager instance.

    Args:
        cache_dir (str): Directory for cache files
        max_cache_size_mb (int): Maximum cache size in MB

    Returns:
        CacheManager: Configured cache manager instance
    """
    config = CacheConfig(dir=cache_dir, max_size_mb=max_cache_size_mb)
    return CacheManager(config)
