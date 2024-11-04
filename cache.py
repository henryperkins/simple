import os
import json
import hashlib
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

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
    return os.path.join(CACHE_DIR, f"{hashed_key}.json")

def load_cache_index() -> Dict[str, str]:
    """Load the cache index."""
    with open(CACHE_INDEX_FILE, 'r') as f:
        return json.load(f)

def save_cache_index(index: Dict[str, str]) -> None:
    """Save the cache index."""
    with open(CACHE_INDEX_FILE, 'w') as f:
        json.dump(index, f)

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
    return {}