"""
File operations and caching module for code analysis.

This module provides functionality for:
1. File system operations (reading, writing, cloning repositories)
2. Caching mechanism for API responses
3. Git repository operations
4. Python file discovery and processing
5. AST-based code analysis

The module implements a thread-safe caching system with LRU (Least Recently Used)
eviction policy and size-based cache management.
"""

import asyncio
import os
import json
import fnmatch
import time
import ast
import hashlib
import threading
import subprocess
import aiofiles
import shutil
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
from core.logger import LoggerSetup
import sentry_sdk
from extract.code import extract_classes_and_functions_from_ast
from api_interaction import analyze_function_with_openai

# Initialize logger for this module
logger = LoggerSetup.get_logger("files")

# Cache configuration
class CacheConfig:
    """Configuration constants for the caching system."""
    DIR = "cache"
    INDEX_FILE = os.path.join(DIR, "index.json")
    MAX_SIZE_MB = 500
    LOCK = threading.Lock()

def initialize_cache() -> None:
    """
    Initialize the cache directory and index file.
    
    Creates the cache directory if it doesn't exist and initializes
    an empty index file for tracking cached responses.
    
    Raises:
        OSError: If there are permission issues or filesystem errors
    """
    try:
        if not os.path.exists(CacheConfig.DIR):
            os.makedirs(CacheConfig.DIR)
            logger.info("Created cache directory.")
        if not os.path.exists(CacheConfig.INDEX_FILE):
            with open(CacheConfig.INDEX_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            logger.info("Initialized cache index file.")
    except OSError as e:
        logger.error(f"Failed to initialize cache: {e}")
        sentry_sdk.capture_exception(e)
        raise

def get_cache_path(key: str) -> str:
    """
    Generate a cache file path based on the key.
    
    Args:
        key (str): The cache key to generate a path for
        
    Returns:
        str: The full path to the cache file
    """
    hashed_key = hashlib.sha256(key.encode()).hexdigest()
    cache_path = os.path.join(CacheConfig.DIR, f"{hashed_key}.json")
    logger.debug(f"Generated cache path for key {key}: {cache_path}")
    return cache_path

def load_cache_index() -> OrderedDict:
    """
    Load and sort the cache index by last access time.
    
    Returns:
        OrderedDict: The sorted cache index
        
    Raises:
        json.JSONDecodeError: If the index file is corrupted
        OSError: If there are filesystem errors
    """
    with CacheConfig.LOCK:
        try:
            if os.path.exists(CacheConfig.INDEX_FILE):
                with open(CacheConfig.INDEX_FILE, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                    index = OrderedDict(sorted(
                        index_data.items(),
                        key=lambda item: item[1]['last_access_time']
                    ))
                    logger.debug("Loaded and sorted cache index.")
                    return index
            else:
                logger.debug("Cache index file not found. Initializing empty index.")
                return OrderedDict()
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Error loading cache index: {e}")
            sentry_sdk.capture_exception(e)
            return OrderedDict()

def save_cache_index(index: OrderedDict) -> None:
    """
    Save the cache index to disk.
    
    Args:
        index (OrderedDict): The cache index to save
        
    Raises:
        OSError: If there are filesystem errors
    """
    with CacheConfig.LOCK:
        try:
            with open(CacheConfig.INDEX_FILE, 'w', encoding='utf-8') as f:
                json.dump(index, f)
            logger.debug("Saved cache index.")
        except OSError as e:
            logger.error(f"Error saving cache index: {e}")
            sentry_sdk.capture_exception(e)
            raise

def cache_response(key: str, data: Dict[str, Any]) -> None:
    """
    Cache response data with the given key.
    
    Args:
        key (str): The key to cache the data under
        data (Dict[str, Any]): The data to cache
        
    Raises:
        OSError: If there are filesystem errors
    """
    index = load_cache_index()
    cache_path = get_cache_path(key)
    with CacheConfig.LOCK:
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
            sentry_sdk.capture_exception(e)
            raise

def get_cached_response(key: str) -> Dict[str, Any]:
    """
    Retrieve cached response based on the key.
    
    Args:
        key (str): The key to retrieve the cached response for
        
    Returns:
        Dict[str, Any]: The cached response or an empty dict if not found
        
    Raises:
        json.JSONDecodeError: If the cached data is corrupted
        OSError: If there are filesystem errors
    """
    index = load_cache_index()
    with CacheConfig.LOCK:
        cache_entry = index.get(key)
        if cache_entry:
            cache_path = cache_entry.get('cache_path')
            if cache_path and os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    cache_entry['last_access_time'] = time.time()
                    index.move_to_end(key)
                    save_cache_index(index)
                    logger.debug(f"Loaded cached response for key: {key}")
                    return data
                except (json.JSONDecodeError, OSError) as e:
                    logger.error(f"Error loading cached response for key {key}: {e}")
                    sentry_sdk.capture_exception(e)
            else:
                logger.warning(f"Cache file does not exist for key: {key}")
                del index[key]
                save_cache_index(index)
        else:
            logger.debug(f"No cached response found for key: {key}")
    return {}

def clear_cache(index: OrderedDict) -> None:
    """
    Evict least recently used cache entries if cache exceeds size limit.
    
    Args:
        index (OrderedDict): The cache index to clean up
        
    Raises:
        OSError: If there are filesystem errors during cleanup
    """
    total_size = 0
    with CacheConfig.LOCK:
        try:
            # Calculate total cache size
            for key, entry in index.items():
                cache_path = entry.get('cache_path')
                if cache_path and os.path.exists(cache_path):
                    total_size += os.path.getsize(cache_path)

            total_size_mb = total_size / (1024 * 1024)
            if total_size_mb > CacheConfig.MAX_SIZE_MB:
                logger.info("Cache size exceeded limit. Starting eviction process.")
                while total_size_mb > CacheConfig.MAX_SIZE_MB and index:
                    key, entry = index.popitem(last=False)
                    cache_path = entry.get('cache_path')
                    if cache_path and os.path.exists(cache_path):
                        file_size = os.path.getsize(cache_path)
                        os.remove(cache_path)
                        total_size -= file_size
                        total_size_mb = total_size / (1024 * 1024)
                        logger.debug(f"Removed cache file {cache_path}")
                save_cache_index(index)
                logger.info("Cache eviction completed.")
            else:
                logger.debug(f"Cache size within limit: {total_size_mb:.2f} MB")
        except OSError as e:
            logger.error(f"Error during cache cleanup: {e}")
            sentry_sdk.capture_exception(e)
            raise

async def clone_repo(repo_url: str, clone_dir: str) -> None:
    """
    Clone a GitHub repository into a specified directory.
    
    Args:
        repo_url (str): The URL of the repository to clone
        clone_dir (str): The directory to clone into
        
    Raises:
        subprocess.CalledProcessError: If git clone fails
        subprocess.TimeoutExpired: If git clone times out
        OSError: If there are filesystem errors
    """
    logger.info(f"Cloning repository {repo_url} into {clone_dir}")
    try:
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)
            logger.info(f"Removed existing directory {clone_dir}")
            
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', repo_url, clone_dir],
            capture_output=True,
            text=True,
            timeout=60,
            check=True
        )
        
        if result.stderr:
            logger.warning(f"Git clone stderr: {result.stderr}")
            
        os.chmod(clone_dir, 0o755)
        logger.info(f"Successfully cloned repository {repo_url}")
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        logger.error(f"Error cloning repository {repo_url}: {e}")
        sentry_sdk.capture_exception(e)
        raise

def load_gitignore_patterns(repo_dir: str) -> List[str]:
    """
    Load .gitignore patterns from the repository directory.
    
    Args:
        repo_dir (str): The repository directory to load patterns from
        
    Returns:
        List[str]: The list of gitignore patterns
    """
    gitignore_path = os.path.join(repo_dir, '.gitignore')
    try:
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            patterns = f.read().splitlines()
        logger.debug(f"Loaded .gitignore patterns from {gitignore_path}")
        return patterns
    except OSError as e:
        logger.error(f"Error loading .gitignore file: {e}")
        sentry_sdk.capture_exception(e)
        return []

def get_all_files(directory: str, exclude_patterns: Optional[List[str]] = None) -> List[str]:
    """
    Retrieve all Python files in the directory, excluding patterns.
    
    Args:
        directory (str): The directory to search in
        exclude_patterns (Optional[List[str]]): Patterns to exclude
        
    Returns:
        List[str]: List of Python file paths
    """
    exclude_patterns = exclude_patterns or []
    python_files: List[str] = []

    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if not any(fnmatch.fnmatch(file_path, pattern) 
                             for pattern in exclude_patterns):
                        python_files.append(file_path)
        logger.debug(f"Found {len(python_files)} Python files in {directory}")
        return python_files
    except Exception as e:
        logger.error(f"Error retrieving Python files: {e}")
        sentry_sdk.capture_exception(e)
        return []

async def process_file(filepath: str, service: str) -> Dict[str, Any]:
    """
    Read and parse a Python file.
    
    Args:
        filepath (str): Path to the Python file
        service (str): The service to use for analysis
        
    Returns:
        Dict[str, Any]: The extracted data or error information
    """
    try:
        content = await read_file_content(filepath)
        logger.debug(f"Read {len(content)} characters from {filepath}")
        tree = ast.parse(content)
        extracted_data = extract_classes_and_functions_from_ast(tree, content)
        logger.info(f"Successfully extracted data for {filepath}")
        return extracted_data
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return {"error": "File not found", "file_content": [{"content": ""}]}
    except UnicodeDecodeError:
        logger.error(f"Unicode decode error in file {filepath}")
        return {"error": "Unicode decode error", "file_content": [{"content": ""}]}
    except SyntaxError as e:
        logger.error(f"Syntax error in file {filepath}: {e}")
        return {"error": "Syntax error", "file_content": [{"content": ""}]}
    except Exception as e:
        logger.error(f"Unexpected error processing file {filepath}: {e}")
        return {"error": "Unexpected error", "file_content": [{"content": ""}]}

async def read_file_content(filepath: str) -> str:
    """
    Read the content of a file asynchronously.
    
    Args:
        filepath (str): Path to the file to read
        
    Returns:
        str: The file content
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        UnicodeDecodeError: If the file can't be decoded as UTF-8
        OSError: For other file system errors
    """
    try:
        async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
            content = await f.read()
        logger.debug(f"Read file content from {filepath}")
        return content
    except (FileNotFoundError, UnicodeDecodeError, OSError) as e:
        logger.error(f"Error reading file {filepath}: {e}")
        sentry_sdk.capture_exception(e)
        raise

async def analyze_and_update_functions(
    extracted_data: Dict[str, Any],
    tree: ast.AST,
    content: str,
    service: str
) -> str:
    """
    Analyze functions and update their docstrings.
    
    Args:
        extracted_data (Dict[str, Any]): The extracted function data
        tree (ast.AST): The AST of the file
        content (str): The original file content
        service (str): The service to use for analysis
        
    Returns:
        str: The updated file content
    """
    file_content = content
    for function in extracted_data.get('functions', []):
        try:
            analysis = await analyze_function_with_openai(function, service)
            function.update(analysis)
            file_content = update_function_docstring(file_content, tree, function, analysis)
            logger.info(f"Updated docstring for function {function['name']}")
        except Exception as e:
            logger.error(f"Error analyzing function {function.get('name', 'unknown')}: {e}")
            sentry_sdk.capture_exception(e)
    return file_content

def update_function_docstring(
    file_content: str,
    tree: ast.AST,
    function: Dict[str, Any],
    analysis: Dict[str, Any]
) -> str:
    """
    Update the docstring of a function in the file content.
    
    Args:
        file_content (str): The original file content
        tree (ast.AST): The AST of the file
        function (Dict[str, Any]): The function information
        analysis (Dict[str, Any]): The analysis results
        
    Returns:
        str: The updated file content
    """
    try:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function['name']:
                docstring = analysis.get('docstring')
                if docstring:
                    logger.debug(f"Inserting docstring for function {function['name']}")
                    return insert_docstring(file_content, node, docstring)
        logger.warning(f"Function {function['name']} not found in AST.")
        return file_content
    except Exception as e:
        logger.error(f"Error updating docstring for function {function['name']}: {e}")
        sentry_sdk.capture_exception(e)
        return file_content

def insert_docstring(
    source: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    docstring: str
) -> str:
    """
    Insert or replace a docstring in a function or class definition.
    
    Args:
        source (str): The source code
        node (Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]): The AST node
        docstring (str): The docstring to insert
        
    Returns:
        str: The updated source code
    """
    try:
        if not hasattr(node, 'body') or not node.body:
            logger.warning(f"No body found for node {getattr(node, 'name', 'unknown')}.")
            return source

        body_start = node.body[0].lineno - 1
        lines = source.splitlines()
        def_line = lines[node.lineno - 1]
        indent = " " * (len(def_line) - len(def_line.lstrip()))

        if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            docstring_start = node.body[0].lineno - 1
            docstring_end = node.body[0].end_lineno
            lines[docstring_start:docstring_end] = [
                f'{indent}"""',
                f'{indent}{docstring}',
                f'{indent}"""'
            ]
        else:
            lines.insert(body_start, f'{indent}"""')
            lines.insert(body_start + 1, f'{indent}{docstring}')
            lines.insert(body_start + 2, f'{indent}"""')

        logger.debug(f"Docstring inserted for node {getattr(node, 'name', 'unknown')}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error inserting docstring: {e}")
        sentry_sdk.capture_exception(e)
        return so