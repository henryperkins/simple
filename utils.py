"""
Core utilities module for Python code analysis and documentation generation.

This module provides comprehensive utilities for:
- AST (Abstract Syntax Tree) processing
- Repository management
- Token counting and management
- File system operations
- JSON processing
- Schema validation
- Configuration management
- String processing
- Error handling

The utilities are organized into logical groups and provide consistent
error handling and logging throughout.
"""

import ast
import re
import git
import json
import os
import shutil
import asyncio
import tiktoken
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple, Set, Type
from dataclasses import dataclass
from git.exc import GitCommandError

from core.logger import LoggerSetup
from core.types import DocstringData, TokenUsage
from exceptions import DocumentationError

# Initialize logger
logger = LoggerSetup.get_logger(__name__)

#-----------------------------------------------------------------------------
# AST Processing Utilities
#-----------------------------------------------------------------------------

class NodeNameVisitor(ast.NodeVisitor):
    """AST visitor for extracting names from nodes."""

    def __init__(self) -> None:
        self.name = ""

    def visit_Name(self, node: ast.Name) -> None:
        """Visit a Name node."""
        self.name = node.id

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visit an Attribute node."""
        self.visit(node.value)
        self.name += f".{node.attr}"

    def visit_Constant(self, node: ast.Constant) -> None:
        """Visit a Constant node."""
        self.name = repr(node.value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Visit a Subscript node."""
        try:
            value = self.visit_and_get(node.value)
            slice_val = self.visit_and_get(node.slice)
            self.name = f"{value}[{slice_val}]"
        except Exception as e:
            logger.debug(f"Error visiting Subscript node: {e}")
            self.name = "unknown_subscript"

    def visit_List(self, node: ast.List) -> None:
        """Visit a List node."""
        try:
            elements = [self.visit_and_get(elt) for elt in node.elts]
            self.name = f"[{', '.join(elements)}]"
        except Exception as e:
            logger.debug(f"Error visiting List node: {e}")
            self.name = "[]"

    def visit_Tuple(self, node: ast.Tuple) -> None:
        """Visit a Tuple node."""
        try:
            elements = [self.visit_and_get(elt) for elt in node.elts]
            self.name = f"({', '.join(elements)})"
        except Exception as e:
            logger.debug(f"Error visiting Tuple node: {e}")
            self.name = "()"

    def visit_Call(self, node: ast.Call) -> None:
        """Visit a Call node."""
        try:
            func_name = self.visit_and_get(node.func)
            args = [self.visit_and_get(arg) for arg in node.args]
            self.name = f"{func_name}({', '.join(args)})"
        except Exception as e:
            logger.debug(f"Error visiting Call node: {e}")
            self.name = "unknown_call"

    def visit_and_get(self, node: ast.AST) -> str:
        """Helper method to visit a node and return its name."""
        visitor = NodeNameVisitor()
        visitor.visit(node)
        return visitor.name or "unknown"

def get_node_name(node: Optional[ast.AST]) -> str:
    """Get the name from an AST node."""
    if node is None:
        return "Any"
    visitor = NodeNameVisitor()
    visitor.visit(node)
    return visitor.name or "unknown"

def get_source_segment(source_code: str, node: ast.AST) -> Optional[str]:
    """Extract source code segment for a given AST node."""
    try:
        if not source_code or not node:
            return None

        if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
            return None

        start_line = node.lineno - 1
        end_line = node.end_lineno

        lines = source_code.splitlines()
        if start_line >= len(lines):
            return None

        return '\n'.join(lines[start_line:end_line]).rstrip()
    except Exception as e:
        logger.error(f"Error extracting source segment: {e}")
        return None

#-----------------------------------------------------------------------------
# Repository Management Utilities
#-----------------------------------------------------------------------------

class RepositoryManager:
    """Handles git repository operations."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.repo = None

    async def clone_repository(self, repo_url: str) -> Path:
        """Clone a repository and return its path."""
        try:
            clone_dir = self.repo_path / Path(repo_url).stem
            if clone_dir.exists():
                if not self._verify_repository(clone_dir):
                    logger.warning(f"Invalid repository at {clone_dir}, re-cloning")
                    shutil.rmtree(clone_dir)
                else:
                    return clone_dir

            logger.info(f"Cloning repository from {repo_url}")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, git.Repo.clone_from, repo_url, clone_dir)

            if not self._verify_repository(clone_dir):
                raise GitCommandError("clone", "Invalid repository structure")

            return clone_dir
        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            raise

    def _verify_repository(self, path: Path) -> bool:
        """Verify repository structure."""
        return (path / ".git").exists()

    def get_python_files(self, exclude_patterns: Optional[Set[str]] = None) -> List[Path]:
        """Get all Python files in the repository."""
        python_files = []
        exclude_patterns = exclude_patterns or set()

        for file_path in self.repo_path.rglob("*.py"):
            if not any(pattern in str(file_path) for pattern in exclude_patterns):
                python_files.append(file_path)

        return python_files

#-----------------------------------------------------------------------------
# Token Management Utilities
#-----------------------------------------------------------------------------

class TokenCounter:
    """Handles token counting and usage calculation."""

    def __init__(self, model: str = "gpt-4"):
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"Model {model} not found. Using cl100k_base encoding.")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Error estimating tokens: {e}")
            return 0

    def calculate_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cost_per_1k_prompt: float = 0.03,
        cost_per_1k_completion: float = 0.06
    ) -> TokenUsage:
        """Calculate token usage and cost."""
        total_tokens = prompt_tokens + completion_tokens
        prompt_cost = (prompt_tokens / 1000) * cost_per_1k_prompt
        completion_cost = (completion_tokens / 1000) * cost_per_1k_completion

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=prompt_cost + completion_cost
        )

#-----------------------------------------------------------------------------
# JSON Processing Utilities
#-----------------------------------------------------------------------------

class CustomJSONEncoder(json.JSONEncoder):
    """Enhanced JSON encoder handling special types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (ast.AST, type)):
            return str(obj)
        if hasattr(obj, "__dict__"):
            return {
                key: value
                for key, value in obj.__dict__.items()
                if not key.startswith("_")
            }
        return super().default(obj)

def serialize_for_logging(obj: Any) -> str:
    """Safely serialize any object for logging."""
    try:
        return json.dumps(obj, cls=CustomJSONEncoder, indent=2)
    except Exception as e:
        return f"Error serializing object: {str(e)}\nObject repr: {repr(obj)}"

#-----------------------------------------------------------------------------
# Environment and Configuration Utilities
#-----------------------------------------------------------------------------

def get_env_var(
    name: str,
    default: Any = None,
    var_type: Type = str,
    required: bool = False
) -> Any:
    """
    Get environment variable with type conversion and validation.

    Args:
        name: Environment variable name
        default: Default value if not found
        var_type: Type to convert the value to
        required: Whether the variable is required

    Returns:
        The environment variable value converted to specified type

    Raises:
        ValueError: If required variable is missing or type conversion fails
    """
    value = os.getenv(name)

    if value is None:
        if required:
            raise ValueError(f"Required environment variable {name} is not set")
        return default

    try:
        if var_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        return var_type(value)
    except ValueError as e:
        raise ValueError(f"Error converting {name} to {var_type.__name__}: {str(e)}")

#-----------------------------------------------------------------------------
# File System Utilities
#-----------------------------------------------------------------------------

def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists and return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def read_file_safe(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """Safely read file content with multiple encoding attempts."""
    encodings = ['utf-8', 'latin-1', 'utf-16']

    if encoding not in encodings:
        encodings.insert(0, encoding)

    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not read file {file_path} with any supported encoding")

#-----------------------------------------------------------------------------
# String Processing Utilities
#-----------------------------------------------------------------------------

def sanitize_identifier(text: str) -> str:
    """Convert text to a valid Python identifier."""
    # Replace non-alphanumeric chars with underscores
    identifier = re.sub(r'\W+', '_', text)
    # Ensure it starts with a letter or underscore
    if identifier and identifier[0].isdigit():
        identifier = f"_{identifier}"
    return identifier

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

#-----------------------------------------------------------------------------
# Error Handling Utilities
#-----------------------------------------------------------------------------

def handle_extraction_error(
    logger: Any,
    errors: List[str],
    process_name: str,
    exception: Exception,
    **kwargs: Any
) -> None:
    """
    Handle and log extraction errors uniformly.

    Args:
        logger: Logger instance
        errors: List to store error messages
        process_name: Name of the process that failed
        exception: The exception that occurred
        **kwargs: Additional context for logging
    """
    error_message = f"{process_name}: {str(exception)}"
    errors.append(error_message)

    sanitized_info = kwargs.get('sanitized_info', {
        'error': str(exception),
        'process': process_name
    })

    logger.error(
        "%s failed: %s",
        process_name,
        exception,
        exc_info=True,
        extra={'sanitized_info': sanitized_info}
    )

#-----------------------------------------------------------------------------
# Module Inspection Utilities
#-----------------------------------------------------------------------------

def check_module_exists(module_name: str) -> bool:
    """
    Check if a Python module exists in the current environment.

    Args:
        module_name: Name of the module to check

    Returns:
        True if module exists, False otherwise
    """
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except Exception:
        return False

def get_module_path(module_name: str) -> Optional[str]:
    """
    Get the file system path for a module.

    Args:
        module_name: Name of the module

    Returns:
        Path to the module file or None if not found
    """
    try:
        spec = importlib.util.find_spec(module_name)
        if spec and spec.origin:
            return spec.origin
        return None
    except Exception:
        return None

#-----------------------------------------------------------------------------
# Time and Date Utilities
#-----------------------------------------------------------------------------

def get_timestamp(fmt: str = "%Y-%m-%d_%H-%M-%S") -> str:
    """
    Get formatted timestamp string.

    Args:
        fmt: DateTime format string

    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(fmt)

def parse_timestamp(timestamp: str, fmt: str = "%Y-%m-%d_%H-%M-%S") -> datetime:
    """
    Parse timestamp string to datetime object.

    Args:
        timestamp: Timestamp string to parse
        fmt: DateTime format string

    Returns:
        datetime object

    Raises:
        ValueError: If parsing fails
    """
    return datetime.strptime(timestamp, fmt)

#-----------------------------------------------------------------------------
# Path Manipulation Utilities
#-----------------------------------------------------------------------------

def normalize_path(path: Union[str, Path]) -> Path:
    """
    Normalize a file system path.

    Args:
        path: Path to normalize

    Returns:
        Normalized Path object
    """
    return Path(path).resolve()

def is_subpath(path: Union[str, Path], parent: Union[str, Path]) -> bool:
    """
    Check if path is a subpath of parent directory.

    Args:
        path: Path to check
        parent: Parent directory path

    Returns:
        True if path is a subpath of parent
    """
    path = normalize_path(path)
    parent = normalize_path(parent)
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False

#-----------------------------------------------------------------------------
# Type Checking Utilities
#-----------------------------------------------------------------------------

def is_optional_type(type_hint: Any) -> bool:
    """
    Check if a type hint is Optional.

    Args:
        type_hint: Type hint to check

    Returns:
        True if type is Optional
    """
    origin = getattr(type_hint, '__origin__', None)
    args = getattr(type_hint, '__args__', ())
    return origin is Union and type(None) in args

def get_optional_type(type_hint: Any) -> Optional[Type]:
    """
    Get the type parameter of an Optional type hint.

    Args:
        type_hint: Optional type hint

    Returns:
        The type parameter or None if not Optional
    """
    if is_optional_type(type_hint):
        args = [arg for arg in type_hint.__args__ if arg is not type(None)]
        return args[0] if args else None
    return None

#-----------------------------------------------------------------------------
# Main Utility Functions
#-----------------------------------------------------------------------------

def safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """
    Safely get object attribute with default value.

    Args:
        obj: Object to get attribute from
        attr: Attribute name
        default: Default value if attribute doesn't exist

    Returns:
        Attribute value or default
    """
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default

def batch_process(
    items: List[Any],
    batch_size: int,
    process_func: callable
) -> List[Any]:
    """
    Process items in batches.

    Args:
        items: Items to process
        batch_size: Size of each batch
        process_func: Function to process each batch

    Returns:
        List of processed results
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        results.extend(process_func(batch))
    return results

#-----------------------------------------------------------------------------
# Exported Utilities
#-----------------------------------------------------------------------------

# List of all utility functions and classes to be exported
__all__ = [
    # AST Processing
    'NodeNameVisitor',
    'get_node_name',
    'get_source_segment',

    # Repository Management
    'RepositoryManager',

    # Token Management
    'TokenCounter',

    # JSON Processing
    'CustomJSONEncoder',
    'serialize_for_logging',

    # Environment and Configuration
    'get_env_var',

    # File System
    'ensure_directory',
    'read_file_safe',

    # String Processing
    'sanitize_identifier',
    'truncate_text',

    # Error Handling
    'handle_extraction_error',

    # Module Inspection
    'check_module_exists',
    'get_module_path',

    # Time and Date
    'get_timestamp',
    'parse_timestamp',

    # Path Manipulation
    'normalize_path',
    'is_subpath',

    # Type Checking
    'is_optional_type',
    'get_optional_type',

    # Main Utilities
    'safe_getattr',
    'batch_process'
]