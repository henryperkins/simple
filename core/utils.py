"""
Utility functions and classes for code analysis, extraction, and documentation generation.
"""

import ast
import os
import re
import sys
import json
import stat
import time
import math
import asyncio
import shutil
import fnmatch
import hashlib
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional, Union
from urllib.parse import urlparse
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)


class FileUtils:
    """Utility methods for file operations."""

    @staticmethod
    def get_file_hash(content: str) -> str:
        """Generate hash for file content."""
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def read_file_safe(file_path: Path, fallback_encoding: str = 'latin-1') -> str:
        """Safely read file content with encoding fallback."""
        try:
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            return file_path.read_text(encoding=fallback_encoding)

    @staticmethod
    def ensure_directory(path: Path) -> None:
        """Ensure directory exists."""
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def filter_files(
        directory: Path,
        pattern: str = '*.py',
        exclude_patterns: Optional[Set[str]] = None
    ) -> List[Path]:
        """Filter files based on pattern."""
        exclude_patterns = exclude_patterns or set()
        files = []
        for path in directory.rglob(pattern):
            if not any(fnmatch.fnmatch(str(path), pat) for pat in exclude_patterns):
                files.append(path)
        return files

    @staticmethod
    def generate_cache_key(data: Any) -> str:
        """
        Generate a unique cache key from input data.
        
        Args:
            data: Data to hash (will be converted to string)
            
        Returns:
            str: SHA-256 hash of the input data
        """
        if not isinstance(data, str):
            data = str(data)
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
class ValidationUtils:
    """Utility methods for validation."""

    @staticmethod
    def validate_docstring(docstring: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate docstring structure."""
        errors = []
        required_fields = {'summary', 'description', 'args', 'returns', 'raises'}

        for field in required_fields:
            if field not in docstring:
                errors.append(f"Missing required field: {field}")

        return len(errors) == 0, errors

    @staticmethod
    def validate_code(source_code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python code syntax."""
        try:
            ast.parse(source_code)
            return True, None
        except SyntaxError as e:
            return False, str(e)


class AsyncUtils:
    """Utility methods for async operations."""

    @staticmethod
    async def with_retry(
        func,
        *args,
        max_retries: int = 3,
        delay: float = 1.0,
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay * (2 ** attempt))
        raise last_error


class GitUtils:
    """Utility methods for Git operations."""

    @staticmethod
    def is_valid_git_url(url: str) -> bool:
        """Validate if a URL is a valid git repository URL."""
        try:
            result = urlparse(url)

            # Check basic URL structure
            if not all([result.scheme, result.netloc]):
                return False

            # Check for valid git URL patterns
            valid_schemes = {'http', 'https', 'git', 'ssh'}
            if result.scheme not in valid_schemes:
                return False

            # Check for common git hosting domains or .git extension
            common_domains = {
                'github.com', 'gitlab.com', 'bitbucket.org',
                'dev.azure.com'
            }

            domain = result.netloc.lower()
            if not any(domain.endswith(d) for d in common_domains):
                if not url.endswith('.git'):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating git URL: {e}")
            return False

    @staticmethod
    async def cleanup_git_directory(path: Path) -> None:
        """Safely clean up a Git repository directory."""
        try:
            # Kill any running Git processes on Windows
            if sys.platform == 'win32':
                os.system('taskkill /F /IM git.exe 2>NUL')

            await asyncio.sleep(1)

            def handle_rm_error(func, path, exc_info):
                """Handle errors during rmtree."""
                try:
                    path_obj = Path(path)
                    if path_obj.exists():
                        os.chmod(str(path_obj), stat.S_IWRITE)
                        func(str(path_obj))
                except Exception as e:
                    logger.error(f"Error removing path {path}: {e}", exc_info=True)

            # Attempt cleanup with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if path.exists():
                        shutil.rmtree(str(path), onerror=handle_rm_error)
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    logger.error(f"Error removing directory {path}: {e}", exc_info=True)
                    break
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)


class FormattingUtils:
    """Utility methods for text formatting."""

    @staticmethod
    def format_docstring(docstring_data: Dict[str, Any]) -> str:
        """Format docstring data into a string."""
        lines = []

        if docstring_data.get('summary'):
            lines.extend([docstring_data['summary'], ""])

        if docstring_data.get('description'):
            lines.extend([docstring_data['description'], ""])

        if docstring_data.get('args'):
            lines.append("Args:")
            for arg in docstring_data['args']:
                arg_desc = f"    {arg['name']} ({arg['type']}): {arg['description']}"
                if arg.get('optional', False):
                    arg_desc += " (Optional)"
                if 'default_value' in arg:
                    arg_desc += f", default: {arg['default_value']}"
                lines.append(arg_desc)
            lines.append("")

        if docstring_data.get('returns'):
            lines.append("Returns:")
            lines.append(f"    {docstring_data['returns']['type']}: "
                         f"{docstring_data['returns']['description']}")
            lines.append("")

        if docstring_data.get('raises'):
            lines.append("Raises:")
            for exc in docstring_data['raises']:
                lines.append(f"    {exc['exception']}: {exc['description']}")
            lines.append("")

        return "\n".join(lines).strip()

    @staticmethod
    def create_warning_message(complexity: int, threshold: int = 10) -> Optional[str]:
        """Create warning message for high complexity."""
        if complexity > threshold:
            return f"⚠️ High complexity warning: {complexity} exceeds threshold of {threshold}"
        return None


# Utility functions

def handle_extraction_error(
    logger_instance,
    errors_list: List[str],
    item_name: str,
    error: Exception
) -> None:
    """Handle extraction errors consistently."""
    error_msg = f"Failed to process {item_name}: {str(error)}"
    logger_instance.error(error_msg, exc_info=True)
    errors_list.append(error_msg)


def get_source_segment(source_code: str, node: ast.AST) -> str:
    """Extract the source segment for a given AST node.

    Args:
        source_code (str): The full source code from which to extract the segment.
        node (ast.AST): The AST node representing the code segment.

    Returns:
        str: The extracted source code segment.
    """
    try:
        return ast.get_source_segment(source_code, node)
    except Exception as e:
        logger.error(f"Error getting source segment for node {node}: {e}", exc_info=True)
        return ""


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system operations."""
    return "".join(c for c in filename if c.isalnum() or c in "._- ")


def generate_cache_key(content: str, prefix: str = "") -> str:
    """Generate cache key from content."""
    hash_value = hashlib.md5(content.encode()).hexdigest()
    return f"{prefix}{hash_value}"


async def load_json_file(filepath: Path) -> Dict[str, Any]:
    """Load and parse JSON file."""
    try:
        content = await FileUtils.read_file_safe(filepath)
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error loading JSON file {filepath}: {e}")
        raise


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format timestamp for logging and display."""
    dt = dt or datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def sanitize_changes(raw_changes: Any) -> List[Dict[str, str]]:
    """Validate and sanitize the raw changes data."""
    changes: List[Dict[str, str]] = []
    current_date = datetime.now().strftime('%Y-%m-%d')

    if not raw_changes:
        return changes

    if not isinstance(raw_changes, list):
        raw_changes = [raw_changes]

    for change in raw_changes:
        if isinstance(change, dict):
            date = str(change.get('date', current_date))
            description = str(change.get('description', ''))
            if description:
                changes.append({'date': date, 'description': description})
        elif isinstance(change, str):
            try:
                parsed_change = json.loads(change)
                if isinstance(parsed_change, dict):
                    date = str(parsed_change.get('date', current_date))
                    description = str(parsed_change.get('description', change))
                    if description:
                        changes.append({'date': date, 'description': description})
                else:
                    changes.append({'date': current_date, 'description': change})
            except (json.JSONDecodeError, TypeError):
                changes.append({'date': current_date, 'description': change})
        elif isinstance(change, (list, tuple)):
            try:
                date = str(change[0])
                description = str(change[1])
                changes.append({'date': date, 'description': description})
            except IndexError:
                changes.append({'date': current_date, 'description': str(change)})
        else:
            changes.append({'date': current_date, 'description': str(change)})

    return changes


def validate_file_path(filepath: str, extension: str = '.py') -> bool:
    """Validate if a file path exists and has the correct extension."""
    return os.path.isfile(filepath) and filepath.endswith(extension)


def create_error_result(error_type: str, error_message: str) -> Dict[str, str]:
    """Create a standardized error result dictionary."""
    return {
        'error_type': error_type,
        'error_message': error_message,
        'timestamp': datetime.now().isoformat()
    }