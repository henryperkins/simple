import os
import ast
import fnmatch
import hashlib
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from core.logger import LoggerSetup, log_debug, log_info, log_error

logger = LoggerSetup.get_logger(__name__)

def generate_hash(content: str) -> str:
    """
    Generate an MD5 hash for the given content.

    Args:
        content (str): The content to hash.

    Returns:
        str: The generated MD5 hash value.
    """
    log_debug(f"Generating hash for content of length {len(content)}.")
    hash_value = hashlib.md5(content.encode()).hexdigest()
    log_debug(f"Generated hash: {hash_value}")
    return hash_value

def get_annotation(node: ast.AST) -> str:
    """
    Get the annotation of an AST node.

    Args:
        node (ast.AST): The AST node from which to extract the annotation.

    Returns:
        str: The extracted annotation, or "Any" if no specific annotation is found.
    """
    try:
        if node is None:
            return "Any"
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return str(node.value)
        if isinstance(node, ast.Attribute):
            return f"{get_annotation(node.value)}.{node.attr}"
        if isinstance(node, ast.Subscript):
            return f"{get_annotation(node.value)}[{get_annotation(node.slice)}]"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = get_annotation(node.left)
            right = get_annotation(node.right)
            return f"Union[{left}, {right}]"
        return "Any"
    except Exception as e:
        logger.error(f"Error processing annotation: {e}")
        return "Any"

def handle_exceptions(log_func):
    """
    Decorator to handle exceptions and log errors.

    Args:
        log_func (callable): The logging function to use for error messages.

    Returns:
        function: A wrapped function that logs exceptions.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                node = kwargs.get('node', None)
                if not node and args:
                    node = next((arg for arg in args if isinstance(arg, ast.AST)), None)

                node_name = getattr(node, 'name', '<unknown>') if node else '<unknown>'
                log_func(f"Error in {func.__name__} for node {node_name}: {e}")
                return None  # Return a default value or handle as needed
        return wrapper
    return decorator

async def load_json_file(filepath: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Load and parse a JSON file with a retry mechanism.

    Args:
        filepath (str): Path to the JSON file.
        max_retries (int): Maximum number of retry attempts.

    Returns:
        Dict[str, Any]: Parsed JSON data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    log_debug(f"Loading JSON file: {filepath}")
    for attempt in range(max_retries):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                log_info(f"Successfully loaded JSON file: {filepath}")
                return data
        except FileNotFoundError:
            log_error(f"File not found: {filepath}")
            raise
        except json.JSONDecodeError as e:
            log_error(f"JSON decode error in file {filepath}: {e}")
            if attempt == max_retries - 1:
                raise
        except Exception as e:
            log_error(f"Unexpected error loading JSON file {filepath}: {e}")
            if attempt == max_retries - 1:
                raise
        await asyncio.sleep(2 ** attempt)
    
    log_error(f"Failed to load JSON file after {max_retries} attempts: {filepath}")
    return {}

def ensure_directory(directory_path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory_path (str): The path of the directory to ensure exists.
    """
    log_debug(f"Ensuring directory exists: {directory_path}")
    os.makedirs(directory_path, exist_ok=True)
    log_info(f"Directory ensured: {directory_path}")

def validate_file_path(filepath: str, extension: str = '.py') -> bool:
    """
    Validate if a file path exists and has the correct extension.

    Args:
        filepath (str): The path to the file to validate.
        extension (str): The expected file extension (default: '.py').

    Returns:
        bool: True if the file path is valid, False otherwise.
    """
    is_valid = os.path.isfile(filepath) and filepath.endswith(extension)
    log_debug(
        f"File path validation for '{filepath}' with extension '{extension}': "
        f"{is_valid}"
    )
    return is_valid

def create_error_result(error_type: str, error_message: str) -> Dict[str, str]:
    """
    Create a standardized error result dictionary.

    Args:
        error_type (str): The type of error that occurred.
        error_message (str): The detailed error message.

    Returns:
        Dict[str, str]: Dictionary containing error information.
    """
    error_result = {
        'error_type': error_type,
        'error_message': error_message,
        'timestamp': datetime.now().isoformat()
    }
    log_debug(f"Created error result: {error_result}")
    return error_result

def add_parent_info(tree: ast.AST) -> None:
    """
    Add parent node information to each node in an AST.

    Args:
        tree (ast.AST): The Abstract Syntax Tree to process.

    Returns:
        None: Modifies the tree in place.
    """
    log_debug("Adding parent information to AST nodes.")
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            setattr(child, 'parent', parent)
    log_info("Parent information added to AST nodes.")

def get_file_stats(filepath: str) -> Dict[str, Any]:
    """
    Get statistical information about a file.

    Args:
        filepath (str): Path to the file to analyze.

    Returns:
        Dict[str, Any]: Dictionary containing file statistics including size,
                        modification time, and other relevant metrics.
    """
    log_debug(f"Getting file statistics for: {filepath}")
    stats = os.stat(filepath)
    file_stats = {
        'size': stats.st_size,
        'modified': datetime.fromtimestamp(stats.st_mtime).isoformat(),
        'created': datetime.fromtimestamp(stats.st_ctime).isoformat(),
        'is_empty': stats.st_size == 0
    }
    log_info(f"File statistics for '{filepath}': {file_stats}")
    return file_stats

def filter_files(
    directory: str,
    pattern: str = '*.py',
    exclude_patterns: Optional[List[str]] = None
) -> List[str]:
    """
    Filter files in a directory based on patterns.

    Args:
        directory (str): The directory path to search in.
        pattern (str): The pattern to match files against (default: '*.py').
        exclude_patterns (Optional[List[str]]): Patterns to exclude from results.

    Returns:
        List[str]: List of file paths that match the criteria.
    """
    log_debug(
        f"Filtering files in directory '{directory}' with pattern '{pattern}'."
    )
    exclude_patterns = exclude_patterns or []
    matches = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if fnmatch.fnmatch(filename, pattern):
                filepath = os.path.join(root, filename)
                if not any(fnmatch.fnmatch(filepath, ep) for ep in exclude_patterns):
                    matches.append(filepath)
    log_info(f"Filtered files: {matches}")
    return matches

def get_all_files(directory: str, exclude_dirs: Optional[List[str]] = None) -> List[str]:
    """
    Traverse the given directory recursively and collect paths to all Python files,
    while excluding any directories specified in the `exclude_dirs` list.

    Args:
        directory (str): The root directory to search for Python files.
        exclude_dirs (Optional[List[str]]): A list of directory names to exclude from the search.

    Returns:
        List[str]: A list of file paths to Python files found in the directory, excluding specified directories.

    Raises:
        ValueError: If the provided directory does not exist or is not accessible.
    """
    if not os.path.isdir(directory):
        raise ValueError(
            f"The directory {directory} does not exist or is not accessible."
        )

    if exclude_dirs is None:
        exclude_dirs = []

    python_files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        for filename in filenames:
            if filename.endswith('.py'):
                python_files.append(os.path.join(dirpath, filename))

    return python_files
