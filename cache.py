import os
import json
import logging
from typing import Dict, Any
from collections import OrderedDict

CACHE_FILE = 'cache.json'
MAX_CACHE_SIZE = 500 * 1024 * 1024  # 500 MB

# In-memory cache
cache = OrderedDict()

def initialize_cache():
    """Initialize the cache from the cache file."""
    global cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = OrderedDict(json.load(f))
            logging.info("Cache loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load cache: {e}")
    else:
        logging.info("No existing cache found. Starting fresh.")

def get_cached_response(function_hash: str) -> Dict[str, Any]:
    """Retrieve cached response if available."""
    if function_hash in cache:
        # Move to end to indicate recent use
        cache.move_to_end(function_hash)
        logging.info(f"Cache hit for hash: {function_hash}")
        return cache[function_hash]
    logging.info(f"Cache miss for hash: {function_hash}")
    return {}

def cache_response(function_hash: str, response: Dict[str, Any]) -> None:
    """Cache the response for future use."""
    cache[function_hash] = response
    cache.move_to_end(function_hash)
    logging.info(f"Cached response for hash: {function_hash}")
    enforce_cache_size()

def enforce_cache_size():
    """Ensure the cache does not exceed the maximum size."""
    total_size = sum(len(json.dumps(v)) for v in cache.values())
    if total_size > MAX_CACHE_SIZE:
        while total_size > MAX_CACHE_SIZE and cache:
            removed_hash, removed_value = cache.popitem(last=False)
            total_size -= len(json.dumps(removed_value))
            logging.info(f"Removed cache entry for hash: {removed_hash}")
    save_cache()

def save_cache():
    """Save the in-memory cache to the cache file."""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f)
        logging.info("Cache saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save cache: {e}")