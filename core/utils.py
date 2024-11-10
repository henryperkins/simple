# core/utils.py

import os
import ast
import json
import hashlib
from typing import Any, Dict, Optional, List
from datetime import datetime
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger("core.utils")

def generate_hash(content: str) -> str:
    """
    Generate a hash from content.

    Args:
        content (str): Content to hash

    Returns:
        str: SHA-256 hash of the content
    """
    return hashlib.sha256(content.encode()).hexdigest()

def load_json_file(filepath: str) -> Dict[str, Any]:
    """
    Load and parse a JSON file.

    Args:
        filepath (str): Path to JSON file

    Returns:
        Dict[str, Any]: Parsed JSON content

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filepath}: {e}")
        raise

def save_json_file(filepath: str, data: Dict[str, Any]) -> None:
    """
    Save data to a JSON file.

    Args:
        filepath (str): Path to save file
        data (Dict[str, Any]): Data to save

    Raises:
        OSError: If file cannot be written
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        logger.error(f"Failed to save file {filepath}: {e}")
        raise

def create_timestamp() -> str:
    """
    Create an ISO format timestamp.

    Returns:
        str: Current timestamp in ISO format
    """
    return datetime.now().isoformat()

def ensure_directory(directory: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        directory (str): Directory path to ensure exists

    Raises:
        OSError: If directory cannot be created
    """
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        raise

def validate_file_path(filepath: str, extension: Optional[str] = None) -> bool:
    """
    Validate if a file path exists and has the correct extension.

    Args:
        filepath (str): Path to validate
        extension (Optional[str]): Expected file extension (e.g., '.py')

    Returns:
        bool: True if path is valid, False otherwise
    """
    if not os.path.exists(filepath):
        return False
    if extension and not filepath.endswith(extension):
        return False
    return True

def create_error_result(error_type: str, error_message: str) -> Dict[str, Any]:
    """
    Create a standardized error result dictionary.

    Args:
        error_type (str): Type of error
        error_message (str): Error message

    Returns:
        Dict[str, Any]: Standardized error result
    """
    return {
        "summary": f"Error: {error_type}",
        "changelog": [{
            "change": f"{error_type}: {error_message}",
            "timestamp": create_timestamp()
        }],
        "classes": [],
        "functions": [],
        "file_content": [{"content": ""}]
    }

def add_parent_info(tree: ast.AST) -> None:
    """
    Add parent information to each node in an AST.

    Args:
        tree (ast.AST): The AST to process
    """
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent

def get_file_stats(filepath: str) -> Dict[str, Any]:
    """
    Get statistics about a file.

    Args:
        filepath (str): Path to the file

    Returns:
        Dict[str, Any]: File statistics including size, modification time, etc.
    """
    try:
        stats = os.stat(filepath)
        return {
            "size": stats.st_size,
            "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stats.st_atime).isoformat()
        }
    except OSError as e:
        logger.error(f"Failed to get file stats for {filepath}: {e}")
        return {}

def filter_files(directory: str, 
                pattern: str = "*.py", 
                exclude_patterns: Optional[List[str]] = None) -> List[str]:
    """
    Filter files in a directory based on pattern and exclusions.

    Args:
        directory (str): Directory to search
        pattern (str): Pattern to match (default: "*.py")
        exclude_patterns (Optional[List[str]]): Patterns to exclude

    Returns:
        List[str]: List of matching file paths
    """
    import fnmatch
    
    exclude_patterns = exclude_patterns or []
    matching_files = []

    try:
        for root, _, files in os.walk(directory):
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    filepath = os.path.join(root, filename)
                    if not any(fnmatch.fnmatch(filepath, exp) for exp in exclude_patterns):
                        matching_files.append(filepath)
        return matching_files
    except Exception as e:
        logger.error(f"Error filtering files in {directory}: {e}")
        return []

def normalize_path(path: str) -> str:
    """
    Normalize a file path for consistent handling.

    Args:
        path (str): Path to normalize

    Returns:
        str: Normalized path
    """
    return os.path.normpath(os.path.abspath(path))

def get_relative_path(path: str, base_path: str) -> str:
    """
    Get relative path from base path.

    Args:
        path (str): Path to convert
        base_path (str): Base path for relative conversion

    Returns:
        str: Relative path
    """
    return os.path.relpath(path, base_path)

def is_python_file(filepath: str) -> bool:
    """
    Check if a file is a Python file.

    Args:
        filepath (str): Path to check

    Returns:
        bool: True if file is a Python file, False otherwise
    """
    if not os.path.isfile(filepath):
        return False
    if not filepath.endswith('.py'):
        return False
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True
    except (SyntaxError, UnicodeDecodeError):
        return False
    except Exception as e:
        logger.error(f"Error checking Python file {filepath}: {e}")
        return False