# cache.py
import os
import json
import hashlib
from typing import Any, Dict
from logging_utils import setup_logger  # Import the logging setup from logging_utils

# Initialize logger for this module
logger = setup_logger("cache")

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
        with open(CACHE_INDEX_FILE, 'w') as f:
            json.dump({}, f)
        logger.info("Initialized cache index file.")

def get_cache_path(key: str) -> str:
    """Generate a cache file path based on the key."""
    hashed_key = hashlib.sha256(key.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{hashed_key}.json")
    logger.debug(f"Generated cache path for key {key}: {cache_path}")
    return cache_path

def load_cache_index() -> Dict[str, str]:
    """Load the cache index."""
    try:
        with open(CACHE_INDEX_FILE, 'r') as f:
            index = json.load(f)
            logger.debug("Loaded cache index.")
            return index
    except Exception as e:
        logger.error(f"Failed to load cache index: {e}")
        return {}

def save_cache_index(index: Dict[str, str]) -> None:
    """Save the cache index."""
    try:
        with open(CACHE_INDEX_FILE, 'w') as f:
            json.dump(index, f)
        logger.debug("Saved cache index.")
    except Exception as e:
        logger.error(f"Failed to save cache index: {e}")

def cache_response(key: str, data: Dict[str, Any]) -> None:
    """Cache the response data with the given key."""
    index = load_cache_index()
    cache_path = get_cache_path(key)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        index[key] = cache_path
        save_cache_index(index)
        logger.debug(f"Cached response for key: {key}")
    except Exception as e:
        logger.error(f"Failed to cache response for key {key}: {e}")

def get_cached_response(key: str) -> Dict[str, Any]:
    """Retrieve cached response based on the key."""
    index = load_cache_index()
    cache_path = index.get(key)
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Loaded cached response for key: {key}")
            return data
        except Exception as e:
            logger.error(f"Failed to load cached response for key {key}: {e}")
    logger.warning(f"No cached response found for key: {key}")
    return {}

def clear_cache():
    """Clear all cached data if the cache exceeds maximum allowed size."""
    try:
        total_size = sum(
            os.path.getsize(os.path.join(CACHE_DIR, f)) for f in os.listdir(CACHE_DIR)
            if os.path.isfile(os.path.join(CACHE_DIR, f))
        )
        total_size_mb = total_size / (1024 * 1024)
        if total_size_mb > CACHE_MAX_SIZE_MB:
            for f in os.listdir(CACHE_DIR):
                file_path = os.path.join(CACHE_DIR, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            with open(CACHE_INDEX_FILE, 'w') as f:
                json.dump({}, f)
            logger.info("Cache cleared as it exceeded maximum size.")
        else:
            logger.debug(f"Cache size within limit: {total_size_mb:.2f} MB")
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")