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
from typing import Any, Dict, List
from core.logger import LoggerSetup
import sentry_sdk
from extract.code import extract_classes_and_functions_from_ast
from api_interaction import analyze_function_with_openai

# Initialize logger for this module
logger = LoggerSetup.get_logger("files")
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
                    index = OrderedDict(sorted(index_data.items(), key=lambda item: item[1]['last_access_time']))
                    logger.debug("Loaded and sorted cache index.")
                    return index
            else:
                logger.debug("Cache index file not found. Initializing empty index.")
                return OrderedDict()
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed for cache index: {e}")
            sentry_sdk.capture_exception(e)
            return OrderedDict()
        except OSError as e:
            logger.error(f"OS error while loading cache index: {e}")
            sentry_sdk.capture_exception(e)
            return OrderedDict()

def save_cache_index(index: OrderedDict) -> None:
    """Save the cache index."""
    with cache_lock:
        try:
            with open(CACHE_INDEX_FILE, 'w', encoding='utf-8') as f:
                json.dump(index, f)
            logger.debug("Saved cache index.")
        except OSError as e:
            logger.error(f"OS error while saving cache index: {e}")
            sentry_sdk.capture_exception(e)

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
            sentry_sdk.capture_exception(e)

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
                    cache_entry['last_access_time'] = time.time()
                    index.move_to_end(key)
                    save_cache_index(index)
                    logger.debug(f"Loaded cached response for key: {key}")
                    return data
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decoding failed for cached response {key}: {e}")
                    sentry_sdk.capture_exception(e)
                except OSError as e:
                    logger.error(f"OS error while loading cached response for key {key}: {e}")
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
    with cache_lock:
        for key, entry in index.items():
            cache_path = entry.get('cache_path')
            if cache_path and os.path.exists(cache_path):
                try:
                    file_size = os.path.getsize(cache_path)
                    total_size += file_size
                except OSError as e:
                    logger.error(f"Error getting size for cache file {cache_path}: {e}")
                    sentry_sdk.capture_exception(e)
                    continue
        total_size_mb = total_size / (1024 * 1024)
        if total_size_mb > CACHE_MAX_SIZE_MB:
            logger.info("Cache size exceeded limit. Starting eviction process.")
            while total_size_mb > CACHE_MAX_SIZE_MB and index:
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
                        sentry_sdk.capture_exception(e)
                else:
                    logger.debug(f"Cache file {cache_path} does not exist.")
            save_cache_index(index)
            logger.info("Cache eviction completed.")
        else:
            logger.debug(f"Cache size within limit: {total_size_mb:.2f} MB")

async def clone_repo(repo_url: str, clone_dir: str) -> None:
    """Clone a GitHub repository into a specified directory."""
    logger.info("Cloning repository %s into %s", repo_url, clone_dir)
    remove_existing_directory(clone_dir)
    try:
        execute_git_clone(repo_url, clone_dir)
        set_directory_permissions(clone_dir)
        logger.info("Successfully cloned repository %s", repo_url)
    except Exception as e:
        logger.error(f"Error cloning repository {repo_url}: {e}")
        sentry_sdk.capture_exception(e)
        raise

def remove_existing_directory(clone_dir: str) -> None:
    """Remove an existing directory if it exists."""
    if os.path.exists(clone_dir):
        try:
            shutil.rmtree(clone_dir)
            logger.info("Removed existing directory %s", clone_dir)
        except OSError as e:
            logger.error(f"Error removing directory {clone_dir}: {e}")
            sentry_sdk.capture_exception(e)
            raise

def execute_git_clone(repo_url: str, clone_dir: str) -> None:
    """Execute the git clone command."""
    try:
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', repo_url, clone_dir],
            capture_output=True,
            text=True,
            timeout=60,
            check=True
        )
        if result.stderr:
            logger.error(f"Git clone stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Git clone failed with return code {e.returncode}: {e.stderr}")
        sentry_sdk.capture_exception(e)
        raise
    except subprocess.TimeoutExpired:
        logger.error("Git clone timed out")
        sentry_sdk.capture_message("Git clone timed out")
        raise
    except OSError as e:
        logger.error(f"OS error during git clone: {e}")
        sentry_sdk.capture_exception(e)
        raise

def set_directory_permissions(clone_dir: str) -> None:
    """Set directory permissions for cloned files."""
    try:
        os.chmod(clone_dir, 0o755)
        logger.info("Set directory permissions for %s", clone_dir)
    except Exception as e:
        logger.error(f"Error setting directory permissions for %s: %s", clone_dir, e)
        sentry_sdk.capture_exception(e)
        raise

def load_gitignore_patterns(repo_dir: str) -> List[str]:
    """Load .gitignore patterns from the repository directory."""
    gitignore_path = os.path.join(repo_dir, '.gitignore')
    try:
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            patterns = f.read().splitlines()
        logger.debug("Loaded .gitignore patterns from %s", gitignore_path)
        return patterns
    except OSError as e:
        logger.error(f"Error loading .gitignore file: {e}")
        sentry_sdk.capture_exception(e)
        return []

def get_all_files(directory: str, exclude_patterns: List[str] = None) -> List[str]:
    """Retrieve all Python files in the directory, excluding patterns."""
    exclude_patterns = exclude_patterns if exclude_patterns is not None else []
    python_files: List[str] = []

    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if not any(fnmatch.fnmatch(file_path, pattern) for pattern in exclude_patterns):
                        python_files.append(file_path)
        logger.debug("Found %d Python files in %s", len(python_files), directory)
    except Exception as e:
        logger.error(f"Error retrieving Python files: {e}")
        sentry_sdk.capture_exception(e)
    return python_files

async def process_file(filepath: str, service: str) -> Dict[str, Any]:
    """Read and parse a Python file."""
    try:
        content = await read_file_content(filepath)
        logger.debug(f"Read {len(content)} characters from {filepath}")
        tree = ast.parse(content)
        extracted_data = extract_classes_and_functions_from_ast(tree, content)
        logger.info("Successfully extracted data for %s", filepath)
        return extracted_data
    except FileNotFoundError:
        logger.error("File not found: %s", filepath)
        sentry_sdk.capture_message(f"File not found: {filepath}")
        return {"error": "File not found", "file_content": [{"content": ""}]}
    except UnicodeDecodeError:
        logger.error("Unicode decode error in file %s", filepath)
        sentry_sdk.capture_message(f"Unicode decode error in file {filepath}")
        return {"error": "Unicode decode error", "file_content": [{"content": ""}]}
    except SyntaxError as e:
        logger.error("Syntax error in file %s: %s", filepath, e)
        sentry_sdk.capture_exception(e)
        return {"error": "Syntax error", "file_content": [{"content": ""}]}
    except Exception as e:
        logger.error("Unexpected error processing file %s: %s", filepath, e)
        sentry_sdk.capture_exception(e)
        return {"error": "Unexpected error", "file_content": [{"content": ""}]}

async def read_file_content(filepath: str) -> str:
    """Read the content of a file asynchronously."""
    try:
        async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
            content = await f.read()
        logger.debug(f"Read file content from {filepath}")
        return content
    except FileNotFoundError as e:
        logger.error(f"File not found: {filepath}")
        sentry_sdk.capture_exception(e)
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Unicode decode error in file {filepath}: {e}")
        sentry_sdk.capture_exception(e)
        raise
    except OSError as e:
        logger.error(f"OS error reading file {filepath}: {e}")
        sentry_sdk.capture_exception(e)
        raise

async def analyze_and_update_functions(extracted_data: Dict[str, Any], tree: ast.AST, content: str, service: str) -> str:
    """Analyze functions and update their docstrings."""
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
    logger.debug("Updated function docstrings")
    return file_content

def update_function_docstring(file_content: str, tree: ast.AST, function: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    """Update the docstring of a function in the file content."""
    try:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function['name']:
                docstring = analysis.get('docstring')
                if docstring:
                    return insert_docstring(file_content, node, docstring)
        logger.warning(f"Function {function['name']} not found in AST.")
    except Exception as e:
        logger.error(f"Error updating docstring for function {function['name']}: {e}")
        sentry_sdk.capture_exception(e)
    return file_content

def insert_docstring(
    source: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    docstring: str
) -> str:
    """Insert or replace a docstring in a function or class definition."""
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
            lines[docstring_start:docstring_end] = [f'{indent}"""', f'{indent}{docstring}', f'{indent}"""']
        else:
            lines.insert(body_start, f'{indent}"""')
            lines.insert(body_start + 1, f'{indent}{docstring}')
            lines.insert(body_start + 2, f'{indent}"""')
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error inserting docstring: {e}")
        sentry_sdk.capture_exception(e)
        return source