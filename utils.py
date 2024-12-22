"""
Core utilities module for Python code analysis and documentation generation.
"""

import asyncio
import ast
import importlib.util
import json
import os
import re
import shutil
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    Callable,
    Awaitable,
    TypeVar,
)

from git import Repo
from git.exc import GitCommandError
import tiktoken

from core.exceptions import LiveError, DocumentationError, WorkflowError
# Corrected import to use get_logger from logging_utils.py
from core.logging_utils import get_logger, CorrelationLoggerAdapter
from core.types.base import TokenUsage

# Initialize logger using the get_logger from logging_utils
logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# AST Node Visitor
# -----------------------------------------------------------------------------


class NodeNameVisitor(ast.NodeVisitor):
    """Visitor to extract the name from an AST node."""

    def __init__(self):
        self.name = None

    def visit_Name(self, node: ast.Name):
        self.name = node.id

    def visit_Attribute(self, node: ast.Attribute):
        self.name = node.attr


# -----------------------------------------------------------------------------
# Error Handling Utilities
# -----------------------------------------------------------------------------


def handle_extraction_error(
    logger: CorrelationLoggerAdapter,
    errors: List[str],
    context: str,
    e: Exception,
    correlation_id: Optional[str] = None,
    **kwargs,
) -> None:
    """Handle extraction process errors with enhanced logging."""
    error_message = f"Error in {context}: {str(e)}"
    errors.append(error_message)

    # Add stack trace for debugging
    logger.error(
        error_message,
        exc_info=True,
        extra={
            "correlation_id": correlation_id or get_correlation_id(),
            "error_type": type(e).__name__,
            "context": context,
            **kwargs,
        },
    )


def log_and_raise_error(
    logger: CorrelationLoggerAdapter,
    error: Exception,
    error_type: type[WorkflowError],
    message: str,
    correlation_id: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Logs an error message and raises a custom exception."""
    extra = {
        "correlation_id": correlation_id or get_correlation_id(),
        "error_type": error_type.__name__,
        "details": kwargs,
    }
    logger.error(message, exc_info=True, extra=extra)
    raise error_type(message) from error


F = TypeVar("F", bound=Callable[..., Any])


def handle_error(
    default_value: Any = None,
) -> Callable[[F], Union[F, Callable[..., Awaitable[Any]]]]:
    """Decorator for common error handling with enhanced logging and optional default return."""

    def decorator(func: F) -> Union[F, Callable[..., Awaitable[Any]]]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger = get_logger()
                logger.error(
                    f"Error in {func.__name__}: {str(e)}",
                    exc_info=True,
                    extra={
                        "correlation_id": get_correlation_id(),
                        "function_args": args,
                        "function_kwargs": kwargs,
                    },
                )
                if default_value is not None:
                    return default_value
                raise

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = get_logger()
                logger.error(
                    f"Error in {func.__name__}: {str(e)}",
                    exc_info=True,
                    extra={
                        "correlation_id": get_correlation_id(),
                        "function_args": args,
                        "function_kwargs": kwargs,
                    },
                )
                if default_value is not None:
                    return default_value
                raise

        # Check if the decorated function is a coroutine function
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# -----------------------------------------------------------------------------
# Module Existence Check Utility
# -----------------------------------------------------------------------------


def check_module_exists(module_name: str) -> bool:
    """Check if a module can be imported without actually importing it."""
    spec = importlib.util.find_spec(module_name)
    return spec is not None


# -----------------------------------------------------------------------------
# Module Path Utilities
# -----------------------------------------------------------------------------


def get_module_path(module_name: str) -> Optional[str]:
    """Get the file path of a module if it exists."""
    spec = importlib.util.find_spec(module_name)
    return spec.origin if spec else None


# -----------------------------------------------------------------------------
# AST Processing Utilities
# -----------------------------------------------------------------------------


def get_node_name(node: Optional[ast.AST]) -> str:
    """Get the name from an AST node."""
    if node is None:
        return "Any"
    visitor = NodeNameVisitor()
    visitor.visit(node)
    return visitor.name or "unknown"


def get_source_segment(source_code: str, node: ast.AST) -> Optional[str]:
    """Extract source code segment for a given AST node with proper indentation."""
    try:
        if not source_code or not node:
            return None

        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return None

        start_line = node.lineno - 1
        end_line = node.end_lineno

        lines = source_code.splitlines()
        if start_line >= len(lines):
            return None

        # Get the lines for this node
        node_lines = lines[start_line:end_line]
        if not node_lines:
            return None

        # Find the minimum indentation level (excluding empty lines)
        indentation_levels = [
            len(line) - len(line.lstrip()) for line in node_lines if line.strip()
        ]
        if not indentation_levels:
            return None
        min_indent = min(indentation_levels)

        # Remove the common indentation from all lines
        normalized_lines = []
        for line in node_lines:
            if line.strip():  # If line is not empty
                # Remove only the common indentation level
                normalized_lines.append(line[min_indent:])
            else:
                normalized_lines.append("")  # Preserve empty lines

        return "\n".join(normalized_lines).rstrip()
    except Exception as e:
        logger.error(f"Error extracting source segment: {e}", exc_info=True)
        return None


# -----------------------------------------------------------------------------
# Repository Management Utilities
# -----------------------------------------------------------------------------


class RepositoryManager:
    """Handles git repository operations."""

    def __init__(self, repo_path: Path, correlation_id: Optional[str] = None):
        self.repo_path = repo_path
        self.repo = None
        self.logger = get_logger(__name__, correlation_id)

    async def clone_repository(self, repo_url: str) -> Path:
        """Clone a repository and return its path."""
        try:
            clone_dir = self.repo_path / Path(repo_url).stem
            if clone_dir.exists():
                if not self._verify_repository(clone_dir):
                    self.logger.warning(
                        f"Invalid repository at {clone_dir}, re-cloning"
                    )
                    shutil.rmtree(clone_dir)
                else:
                    return clone_dir

            self.logger.info(f"Cloning repository from {repo_url}")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, Repo.clone_from, repo_url, clone_dir)

            if not self._verify_repository(clone_dir):
                raise GitCommandError("clone", "Invalid repository structure")

            return clone_dir
        except Exception as e:
            self.logger.error(f"Failed to clone repository: {e}", exc_info=True)
            raise

    def _verify_repository(self, path: Path) -> bool:
        """Verify repository structure."""
        return (path / ".git").exists()

    def get_python_files(
        self, exclude_patterns: Optional[Set[str]] = None
    ) -> List[Path]:
        """Get all Python files in the repository."""
        python_files = []
        exclude_patterns = exclude_patterns or set()

        for file_path in self.repo_path.rglob("*.py"):
            if not any(pattern in str(file_path) for pattern in exclude_patterns):
                python_files.append(file_path)

        return python_files


# -----------------------------------------------------------------------------
# Token Management Utilities
# -----------------------------------------------------------------------------


class TokenCounter:
    """Handles token counting and usage calculation."""

    def __init__(self, model: str = "gpt-4", correlation_id: Optional[str] = None):
        self.logger = get_logger(__name__, correlation_id)
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.logger.warning(f"Model {model} not found. Using cl100k_base encoding.")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            self.logger.error(f"Error estimating tokens: {e}", exc_info=True)
            return 0

    def calculate_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cost_per_1k_prompt: float = 0.03,
        cost_per_1k_completion: float = 0.06,
    ) -> TokenUsage:
        """Calculate token usage and cost."""

        total_tokens = prompt_tokens + completion_tokens
        prompt_cost = (prompt_tokens / 1000) * cost_per_1k_prompt
        completion_cost = (completion_tokens / 1000) * cost_per_1k_completion

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=prompt_cost + completion_cost,
        )


# -----------------------------------------------------------------------------
# JSON Processing Utilities
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Environment and Configuration Utilities
# -----------------------------------------------------------------------------


def get_env_var(
    name: str,
    default: Any = None,
    var_type: Type = str,
    required: bool = False,
    correlation_id: Optional[str] = None,
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
            return value.lower() in ("true", "1", "yes", "on")
        return var_type(value)
    except ValueError as e:
        raise ValueError(f"Error converting {name} to {var_type.__name__}: {str(e)}")


# -----------------------------------------------------------------------------
# File Reading Utility
# -----------------------------------------------------------------------------


def _read_file(file_path: Path) -> str:
    """Read a file and return its contents."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
        return ""


def read_file_safe(
    file_path: Union[str, Path], correlation_id: Optional[str] = None
) -> str:
    """Safely read a file and return its contents."""
    file_path = Path(file_path)
    content = _read_file(file_path)
    if not content:
        logger.warning(
            f"File {file_path} is empty or contains only whitespace.",
            extra={"correlation_id": correlation_id or get_correlation_id()},
        )
    return content


@handle_error(default_value="")
async def read_file_safe_async(file_path: Union[str, Path]) -> str:
    """Safely read a file asynchronously."""
    file_path = Path(file_path)
    return _read_file(file_path)


# -----------------------------------------------------------------------------
# File System Utilities
# -----------------------------------------------------------------------------


def ensure_directory(
    path: Union[str, Path], correlation_id: Optional[str] = None
) -> Path:
    """Ensure directory exists and return Path object."""
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        logger.error(
            f"Error ensuring directory {path}: {e}",
            exc_info=True,
            extra={"correlation_id": correlation_id or get_correlation_id()},
        )
        return path


# -----------------------------------------------------------------------------
# String Processing Utilities
# -----------------------------------------------------------------------------


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


# -----------------------------------------------------------------------------
# Path Manipulation Utilities
# -----------------------------------------------------------------------------


def normalize_path(
    path: Union[str, Path], correlation_id: Optional[str] = None
) -> Path:
    """
    Normalize a file system path.

    Args:
        path: Path to normalize

    Returns:
        Normalized Path object
    """
    try:
        return Path(path).resolve()
    except Exception as e:
        logger.error(
            f"Error normalizing path {path}: {e}",
            exc_info=True,
            extra={"correlation_id": correlation_id or get_correlation_id()},
        )
        return Path(path)


def fetch_dependency(name: str) -> Any:
    """Fetch a dependency using Injector.get()."""
    from core.dependency_injection import Injector  # Import here

    try:
        return Injector.get(name)
    except KeyError as e:
        logger.error(f"Error fetching dependency '{name}': {e}", exc_info=True)
        raise


# List of all utility functions and classes to be exported
__all__ = [
    # AST Processing
    "get_node_name",
    "get_source_segment",
    # Repository Management
    "RepositoryManager",
    # Token Management
    "TokenCounter",
    # JSON Processing
    "CustomJSONEncoder",
    "serialize_for_logging",
    # Environment and Configuration
    "get_env_var",
    # File System
    "ensure_directory",
    # String Processing
    "truncate_text",
    # Path Manipulation
    "normalize_path",
    "handle_extraction_error",
    "handle_error",
    "log_and_raise_error",
    "get_logger",
    "fetch_dependency",
    "read_file_safe",
    "read_file_safe_async",
]
