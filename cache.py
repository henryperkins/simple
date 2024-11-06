# cache.py

import os
import json
import hashlib
import threading
import time
from typing import Any, Dict
from collections import OrderedDict
from logging_utils import setup_logger

# Initialize logger for this module
logger = setup_logger("cache")
cache_lock = threading.Lock()

# Cache directory and configuration
CACHE_DIR = "cache"
CACHE_INDEX_FILE = os.path.join(CACHE_DIR, "index.json")
CACHE_MAX_SIZE_MB = 500

def initialize_cache():
    """Initialize the cache directory and index."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        logger.info("Created cache directory.")

    if not os.path.exists(CACHE_INDEX_FILE):
        with open(CACHE_INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        logger.info("Initialized cache index file.")

def get_cache_path(key: str) -> str:
    """Generate a cache file path based on the key."""
    hashed_key = hashlib.sha256(key.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{hashed_key}.json")
    logger.debug(f"Generated cache path for key {key}: {cache_path}")
    return cache_path

def load_cache_index() -> OrderedDict:
    """Load the cache index, sorted by last access time."""
    with cache_lock:
        try:
            if os.path.exists(CACHE_INDEX_FILE):
                with open(CACHE_INDEX_FILE, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                    # Convert index_data to OrderedDict sorted by last_access_time
                    index = OrderedDict(sorted(index_data.items(), key=lambda item: item[1]['last_access_time']))
                    logger.debug("Loaded and sorted cache index.")
                    return index
            else:
                logger.debug("Cache index file not found. Initializing empty index.")
                return OrderedDict()
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed for cache index: {e}")
            return OrderedDict()
        except OSError as e:
            logger.error(f"OS error while loading cache index: {e}")
            return OrderedDict()

def save_cache_index(index: OrderedDict) -> None:
    """Save the cache index."""
    with cache_lock:
        try:
            with open(CACHE_INDEX_FILE, 'w', encoding='utf-8') as f:
                json.dump(index, f)
            logger.debug("Saved cache index.")
        except OSError as e:
            logger.error(f"Failed to save cache index: {e}")

def cache_response(key: str, data: Dict[str, Any]) -> None:
    """Cache the response data with the given key."""
    index = load_cache_index()
    cache_path = get_cache_path(key)
    with cache_lock:
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            index[key] = {
                'cache_path': cache_path,
                'last_access_time': time.time()
            }
            save_cache_index(index)
            logger.debug(f"Cached response for key: {key}")
            clear_cache(index)
        except OSError as e:
            logger.error(f"Failed to cache response for key {key}: {e}")

def get_cached_response(key: str) -> Dict[str, Any]:
    """Retrieve cached response based on the key."""
    index = load_cache_index()
    with cache_lock:
        cache_entry = index.get(key)
        if cache_entry:
            cache_path = cache_entry.get('cache_path')
            if cache_path and os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # Update last access time
                    cache_entry['last_access_time'] = time.time()
                    index.move_to_end(key)  # Move to end to reflect recent access
                    save_cache_index(index)
                    logger.debug(f"Loaded cached response for key: {key}")
                    return data
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decoding failed for cached response {key}: {e}")
                except OSError as e:
                    logger.error(f"OS error while loading cached response for key {key}: {e}")
            else:
                logger.warning(f"Cache file does not exist for key: {key}")
                # Remove invalid cache entry
                del index[key]
                save_cache_index(index)
        else:
            logger.debug(f"No cached response found for key: {key}")
    return {}

def clear_cache(index: OrderedDict) -> None:
    """Evict least recently used cache entries if cache exceeds size limit."""
    total_size = 0
    with cache_lock:
        for key, entry in index.items():
            cache_path = entry.get('cache_path')
            if cache_path and os.path.exists(cache_path):
                try:
                    file_size = os.path.getsize(cache_path)
                    total_size += file_size
                except OSError as e:
                    logger.error(f"Error getting size for cache file {cache_path}: {e}")
                    continue
        total_size_mb = total_size / (1024 * 1024)
        if total_size_mb > CACHE_MAX_SIZE_MB:
            logger.info("Cache size exceeded limit. Starting eviction process.")
            while total_size_mb > CACHE_MAX_SIZE_MB and index:
                # Pop the least recently used item
                key, entry = index.popitem(last=False)
                cache_path = entry.get('cache_path')
                if cache_path and os.path.exists(cache_path):
                    try:
                        file_size = os.path.getsize(cache_path)
                        os.remove(cache_path)
                        total_size -= file_size
                        total_size_mb = total_size / (1024 * 1024)
                        logger.debug(f"Removed cache file {cache_path} for key {key}")
                    except OSError as e:
                        logger.error(f"Error removing cache file {cache_path}: {e}")
                else:
                    logger.debug(f"Cache file {cache_path} does not exist.")
            save_cache_index(index)
            logger.info("Cache eviction completed.")
        else:
            logger.debug(f"Cache size within limit: {total_size_mb:.2f} MB")