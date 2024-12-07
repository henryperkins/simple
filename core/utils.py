"""
Utility functions and classes for code analysis, extraction, Git repository handling, and documentation generation.
"""

import ast
import os
import sys
import asyncio
import shutil
import fnmatch
import hashlib
import stat
from pathlib import Path
from typing import Any, List, Set, Optional
from urllib.parse import urlparse
from core.logger import LoggerSetup

# Setup logging
logger = LoggerSetup.get_logger(__name__)

class FileUtils:
    """Utility methods for file operations."""

    @staticmethod
    def get_file_hash(content: str) -> str:
        """Generate hash for file content."""
        try:
            return hashlib.md5(content.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating file hash: {e}", exc_info=True)
            raise

    @staticmethod
    def read_file_safe(file_path: Path, fallback_encoding: str = "latin-1") -> str:
        """Safely read file content with encoding fallback."""
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return file_path.read_text(encoding=fallback_encoding)
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
                raise

    @staticmethod
    def ensure_directory(path: Path) -> None:
        """Ensure directory exists."""
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error ensuring directory {path}: {e}", exc_info=True)
            raise

    @staticmethod
    def filter_files(
        directory: Path,
        pattern: str = "*.py",
        exclude_patterns: Optional[Set[str]] = None,
    ) -> List[Path]:
        """Filter files based on pattern."""
        exclude_patterns = exclude_patterns or set()
        try:
            files = [
                path for path in directory.rglob(pattern)
                if not any(fnmatch.fnmatch(str(path), pat) for pat in exclude_patterns)
            ]
            return files
        except Exception as e:
            logger.error(f"Error filtering files in directory {directory}: {e}", exc_info=True)
            raise

    @staticmethod
    def generate_cache_key(data: Any) -> str:
        """Generate a unique cache key from input data."""
        try:
            if not isinstance(data, str):
                data = str(data)
            return hashlib.sha256(data.encode("utf-8")).hexdigest()
        except Exception as e:
            logger.error(f"Error generating cache key: {e}", exc_info=True)
            raise


class ValidationUtils:
    """Utility methods for validation."""

    @staticmethod
    def validate_docstring(docstring: dict[str, Any], schema: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate the docstring but allow empty lists for args and raises."""
        errors: list[str] = []
        required_fields = {"summary", "description", "args", "returns", "raises"}

        for field in required_fields:
            if field not in docstring:
                errors.append(f"Missing required field: {field}")
            elif field in ['args', 'raises']:
                # Allow empty lists for args and raises
                if not isinstance(docstring[field], list):
                    errors.append(f"Field {field} must be a list")
            elif not docstring[field] and field not in ['args', 'raises']:
                errors.append(f"Empty value for required field: {field}")

        # Only validate non-empty args and raises
        if docstring.get('args'):
            for arg in docstring['args']:
                if not all(key in arg for key in ["name", "type", "description"]):
                    missing_keys = [key for key in ["name", "type", "description"] if key not in arg]
                    errors.append(f"Incomplete argument specification: Missing {', '.join(missing_keys)}")

        if docstring.get('raises'):
            for exc in docstring['raises']:
                if not all(key in exc for key in ["exception", "description"]):
                    missing_keys = [key for key in ["exception", "description"] if key not in exc]
                    errors.append(f"Incomplete raises specification: Missing {', '.join(missing_keys)}")

        return len(errors) == 0, errors
    
    @staticmethod
    def validate_code(source_code: str) -> tuple[bool, Optional[str]]:
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
        func: Any, *args: Any, max_retries: int = 3, delay: float = 1.0, **kwargs: Any
    ) -> Any:
        """Execute function with retry logic."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay * (2 ** attempt))
        logger.error(f"All attempts failed after {max_retries} retries.")
        raise last_error


class GitUtils:
    """Utility methods for Git operations."""

    @staticmethod
    def is_valid_git_url(url: str) -> bool:
        """Validate if a URL is a valid git repository URL."""
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                return False

            valid_schemes = {"http", "https", "git", "ssh"}
            if result.scheme not in valid_schemes:
                return False

            common_domains = {
                "github.com",
                "gitlab.com",
                "bitbucket.org",
                "dev.azure.com",
            }

            domain = result.netloc.lower()
            if not any(domain.endswith(d) for d in common_domains):
                if not url.endswith(".git"):
                    return False

            return True
        except Exception as e:
            logger.error(f"Error validating git URL: {e}", exc_info=True)
            return False
        
    @staticmethod
    def get_python_files(repo_path: Path, exclude_patterns: Optional[Set[str]] = None) -> list[Path]:
        """
        Get all Python files from the repository.

        Args:
            repo_path (Path): Path to the repository
            exclude_patterns (Optional[Set[str]]): Set of patterns to exclude

        Returns:
            List[Path]: List of Python file paths
        """
        if not repo_path:
            raise ValueError("Repository path not set")

        exclude_patterns = exclude_patterns or {
            "*/venv/*",
            "*/env/*",
            "*/build/*",
            "*/dist/*",
            "*/.git/*",
            "*/__pycache__/*",
            "*/migrations/*",
        }

        try:
            # Log repository structure for debugging
            logger.debug(f"Scanning repository at: {repo_path}")
            logger.debug(f"Directory contents: {list(repo_path.iterdir())}")
            
            # Use rglob to find all Python files recursively
            python_files: list[Path] = []
            for file_path in repo_path.rglob("*.py"):
                # Convert to string for pattern matching
                file_str = str(file_path)
                # Check if file should be excluded
                if not any(fnmatch.fnmatch(file_str, pattern) for pattern in exclude_patterns):
                    python_files.append(file_path)
                    logger.debug(f"Found Python file: {file_path}")

            logger.info(f"Found {len(python_files)} Python files in {repo_path}")
            
            if not python_files:
                logger.warning(f"No Python files found in repository: {repo_path}")
            
            return python_files

        except Exception as e:
            logger.error(f"Error finding Python files in {repo_path}: {e}")
            return []
        
    @staticmethod
    async def cleanup_git_directory(path: Path) -> None:
        """Safely clean up a Git repository directory."""
        try:
            # Kill any running Git processes on Windows
            if sys.platform == "win32":
                os.system("taskkill /F /IM git.exe 2>NUL")

            await asyncio.sleep(1)

            def handle_rm_error(func: Any, path: Any, exc_info: Any) -> None:
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
                        shutil.rmtree(str(path), onexc=handle_rm_error)
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                except Exception as e:
                    logger.error(f"Error removing directory {path}: {e}", exc_info=True)
                    break
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)


class FormattingUtils:
    """Utility methods for text formatting."""

    @staticmethod
    def format_docstring(docstring_data: dict[str, Any]) -> str:
        """Format docstring data into Google style string."""
        try:
            lines: list[str] = []

            if docstring_data.get("summary"):
                lines.append(docstring_data["summary"])
                lines.append("")  # Separate summary from other parts

            if docstring_data.get("description"):
                lines.append(docstring_data["description"])
                lines.append("")

            if docstring_data.get("args"):
                lines.append("Args:")
                for arg in docstring_data["args"]:
                    type_annotation = arg.get("type", "Any")
                    description = arg.get("description", "No description provided")
                    lines.append(f"    {arg['name']} ({type_annotation}): {description}")
                lines.append("")

            if docstring_data.get("returns"):
                returns_type = docstring_data['returns'].get('type', 'Any')
                returns_desc = docstring_data['returns'].get('description', 'No description provided.')
                lines.append("Returns:")
                lines.append(f"    {returns_type}: {returns_desc}")
                lines.append("")

            if docstring_data.get("raises"):
                lines.append("Raises:")
                for exc in docstring_data["raises"]:
                    lines.append(f"    {exc['exception']}: {exc['description']}")
                lines.append("")

            return "\n".join(lines).strip()

        except Exception as e:
            logger.error(f"Error formatting docstring: {e}", exc_info=True)
            raise


# Utility functions

def handle_extraction_error(logger_instance: Any, errors_list: list[str], item_name: str, error: Exception) -> None:
    """Handle extraction errors consistently."""
    error_msg = f"Failed to process {item_name}: {str(error)}"
    logger_instance.error(error_msg, exc_info=True)
    errors_list.append(error_msg)

def get_source_segment(source_code: str, node: ast.AST) -> Optional[str]:
    """Extract the source segment for a given AST node."""
    try:
        if not source_code or not node:
            logger.debug("Source code or node is None.")
            return None

        # Only log for complex definitions that we actually want to extract
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
            # Return None silently for other node types - they should use NodeNameVisitor instead
            return None

        start_line = getattr(node, 'lineno', None)
        end_line = getattr(node, 'end_lineno', None)

        if start_line is None:
            logger.debug(f"Node {type(node).__name__} has no start line.")
            return None

        start = start_line - 1  # Convert to 0-based index

        if end_line is None:
            end = start + 1
        else:
            end = end_line

        lines = source_code.splitlines()
        if start >= len(lines):
            logger.debug(f"Start line {start} exceeds available lines in the source.")
            return None

        segment = '\n'.join(lines[start:end])
        return segment.rstrip()

    except Exception as e:
        logger.error(f"Error extracting source segment: {e}", exc_info=True)
        return None


class NodeNameVisitor(ast.NodeVisitor):
    """Extracts names and relevant information from AST nodes for docstring generation."""

    def __init__(self):
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

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Visit a BinOp node."""
        try:
            left = self.visit_and_get(node.left)
            op = type(node.op).__name__
            right = self.visit_and_get(node.right)
            self.name = f"{left} {op} {right}"
        except Exception as e:
            logger.debug(f"Error visiting BinOp node: {e}")
            self.name = "unknown_operation"

    def visit_and_get(self, node: ast.AST) -> str:
        """Helper method to visit a node and return its name."""
        visitor = NodeNameVisitor()
        visitor.visit(node)
        return visitor.name or "unknown"

    def generic_visit(self, node: ast.AST) -> None:
        """Handle any unhandled node types."""
        logger.debug(f"Unhandled node type in NodeNameVisitor: {type(node).__name__}")
        self.name = f"unknown_{type(node).__name__.lower()}"

    def _extract_type_info(self, node: ast.AST) -> str:
        """Extract accurate type information from AST node."""
        visitor = NodeNameVisitor()
        visitor.visit(node)
        return visitor.name or "Any"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system operations."""
    try:
        return "".join(c for c in filename if c.isalnum() or c in "._- ")
    except Exception as e:
        logger.error(f"Error sanitizing filename '{filename}': {e}", exc_info=True)
        raise

def validate_file_path(filepath: str, extension: str = ".py") -> bool:
    """Validate if a file path exists and has the correct extension."""
    try:
        return os.path.isfile(filepath) and filepath.endswith(extension)
    except Exception as e:
        logger.error(f"Error validating file path '{filepath}': {e}", exc_info=True)
        raise
