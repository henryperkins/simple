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
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from core.logger import LoggerSetup
import sentry_sdk
from extract.code import extract_classes_and_functions_from_ast
from extract.utils import add_parent_info, validate_schema
from api_interaction import analyze_function_with_openai
from metrics import CodeMetrics

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
    """Initialize the cache directory and index file."""
    try:
        if not os.path.exists(CacheConfig.DIR):
            os.makedirs(CacheConfig.DIR)
            logger.info("Created cache directory.")
        if not os.path.exists(CacheConfig.INDEX_FILE):
            with open(CacheConfig.INDEX_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            logger.info("Initialized cache index file.")
    except OSError as e:
        logger.error(f"Error initializing cache: {e}")
        raise

def get_cache_path(key: str) -> str:
    """Generate a cache file path based on the key."""
    hashed_key = hashlib.sha256(key.encode()).hexdigest()
    cache_path = os.path.join(CacheConfig.DIR, f"{hashed_key}.json")
    logger.debug(f"Generated cache path for key {key}: {cache_path}")
    return cache_path

def load_cache_index() -> OrderedDict:
    """Load and sort the cache index by last access time."""
    with CacheConfig.LOCK:
        try:
            with open(CacheConfig.INDEX_FILE, 'r', encoding='utf-8') as f:
                index = json.load(f, object_pairs_hook=OrderedDict)
            logger.debug("Loaded cache index.")
            return OrderedDict(sorted(index.items(), key=lambda item: item[1]['last_access_time']))
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Error loading cache index: {e}")
            raise

def save_cache_index(index: OrderedDict) -> None:
    """Save the cache index to disk."""
    with CacheConfig.LOCK:
        try:
            with open(CacheConfig.INDEX_FILE, 'w', encoding='utf-8') as f:
                json.dump(index, f)
            logger.debug("Saved cache index.")
        except OSError as e:
            logger.error(f"Error saving cache index: {e}")
            raise

def cache_response(key: str, data: Dict[str, Any]) -> None:
    """Cache response data with the given key."""
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
            logger.debug(f"Cached response for key {key}.")
        except OSError as e:
            logger.error(f"Error caching response for key {key}: {e}")
            raise

def get_cached_response(key: str) -> Dict[str, Any]:
    """Retrieve cached response based on the key."""
    index = load_cache_index()
    with CacheConfig.LOCK:
        cache_entry = index.get(key)
        if cache_entry:
            cache_path = cache_entry.get('cache_path')
            if cache_path and os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # Update last access time
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
    """Evict least recently used cache entries if cache exceeds size limit."""
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
    """Clone a GitHub repository into a specified directory."""
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
    """Load .gitignore patterns from the repository directory."""
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
    """Retrieve all Python files in the directory, excluding patterns."""
    exclude_patterns = exclude_patterns or []
    python_files: List[str] = []

    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if not any(fnmatch.fnmatch(file_path, pattern) for pattern in exclude_patterns):
                        python_files.append(file_path)
        logger.debug(f"Found {len(python_files)} Python files in {directory}")
        return python_files
    except Exception as e:
        logger.error(f"Error retrieving Python files: {e}")
        sentry_sdk.capture_exception(e)
        return []

async def process_file(filepath: str, service: str) -> Dict[str, Any]:
    """Read and parse a Python file."""
    if service not in ['azure', 'openai']:
        logger.error(f"Invalid service specified: {service}")
        return {
            "summary": "Error during extraction",
            "changelog": [{
                "change": f"Error: Invalid service '{service}' specified",
                "timestamp": datetime.now().isoformat()
            }],
            "classes": [],
            "functions": [],
            "file_content": [{"content": ""}]
        }

    try:
        # Read file content
        content = await read_file_content(filepath)
        logger.debug(f"Read {len(content)} characters from {filepath}")

        # Parse AST and add parent information
        try:
            tree = ast.parse(content)
            add_parent_info(tree)  # Add parent references to AST nodes
        except SyntaxError as e:
            logger.error(f"Syntax error in file {filepath}: {e}")
            return {
                "summary": "Error during extraction",
                "changelog": [{
                    "change": f"Syntax error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }],
                "classes": [],
                "functions": [],
                "file_content": [{"content": content}]
            }

        # Extract data using AST
        try:
            extracted_data = extract_classes_and_functions_from_ast(tree, content)
            
            # Validate extracted data
            try:
                validate_schema(extracted_data)
            except Exception as e:
                logger.error(f"Schema validation failed for {filepath}: {e}")
                sentry_sdk.capture_exception(e)
                return {
                    "summary": "Error during extraction",
                    "changelog": [{
                        "change": f"Schema validation failed: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }],
                    "classes": [],
                    "functions": [],
                    "file_content": [{"content": content}]
                }

            # Calculate additional metrics
            metrics = CodeMetrics()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity = metrics.calculate_complexity(node)
                    cognitive_complexity = metrics.calculate_cognitive_complexity(node)
                    halstead = metrics.calculate_halstead_metrics(node)
                    
                    # Update metrics in extracted data
                    for func in extracted_data.get("functions", []):
                        if func["name"] == node.name:
                            func["complexity_score"] = complexity
                            func["cognitive_complexity"] = cognitive_complexity
                            func["halstead_metrics"] = halstead

            logger.info(f"Successfully extracted data for {filepath}")
            return extracted_data

        except Exception as e:
            logger.error(f"Error extracting data from {filepath}: {e}")
            sentry_sdk.capture_exception(e)
            return {
                "summary": "Error during extraction",
                "changelog": [{
                    "change": f"Data extraction error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }],
                "classes": [],
                "functions": [],
                "file_content": [{"content": content}]
            }

    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return {
            "summary": "Error during extraction",
            "changelog": [{
                "change": "File not found",
                "timestamp": datetime.now().isoformat()
            }],
            "classes": [],
            "functions": [],
            "file_content": [{"content": ""}]
        }
    except UnicodeDecodeError:
        logger.error(f"Unicode decode error in file {filepath}")
        return {
            "summary": "Error during extraction",
            "changelog": [{
                "change": "Unicode decode error",
                "timestamp": datetime.now().isoformat()
            }],
            "classes": [],
            "functions": [],
            "file_content": [{"content": ""}]
        }
    except Exception as e:
        logger.error(f"Error processing file {filepath}: {e}")
        sentry_sdk.capture_exception(e)
        return {
            "summary": "Error during extraction",
            "changelog": [{
                "change": f"Unexpected error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }],
            "classes": [],
            "functions": [],
            "file_content": [{"content": ""}]
        }

async def read_file_content(filepath: str) -> str:
    """Read the content of a file asynchronously."""
    try:
        async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
            content = await f.read()
        logger.debug(f"Read content from {filepath}")
        return content
    except FileNotFoundError as e:
        logger.error(f"File not found: {filepath}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Unicode decode error in file {filepath}")
        raise
    except OSError as e:
        logger.error(f"OS error while reading file {filepath}: {e}")
        raise

async def analyze_and_update_functions(
    extracted_data: Dict[str, Any],
    tree: ast.AST,
    content: str,
    service: str
) -> str:
    """Analyze functions and update their docstrings."""
    for func in extracted_data.get("functions", []):
        analysis = await analyze_function_with_openai(func, service)
        content = update_function_docstring(content, tree, func, analysis)
    return content

def update_function_docstring(
    file_content: str,
    tree: ast.AST,
    function: Dict[str, Any],
    analysis: Dict[str, Any]
) -> str:
    """Update the docstring of a function."""
    new_docstring = analysis.get("docstring", "")
    if not new_docstring:
        return file_content

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function["name"]:
            file_content = insert_docstring(file_content, node, new_docstring)
            break
    return file_content

def insert_docstring(
    source: str,
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    docstring: str
) -> str:
    """Insert a docstring into a function or class."""
    lines = source.splitlines()
    start_line = node.body[0].lineno - 1
    indent = " " * (node.body[0].col_offset or 0)
    docstring_lines = [f'{indent}"""', f'{indent}{docstring}', f'{indent}"""']
    lines[start_line:start_line] = docstring_lines
    return "\n".join(lines)