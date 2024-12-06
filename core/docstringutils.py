"""Utilities for handling Python docstrings and AST operations."""

import ast
from typing import Dict, Any, List, Optional, Union

from core.logger import LoggerSetup
from core.utils import get_node_name  # Ensure this import is present

logger = LoggerSetup.get_logger(__name__)

try:
    import astor
except ImportError as e:
    raise ImportError(
        "The 'astor' library is required for Python versions < 3.9 to unparse AST nodes. "
        "Please install it using 'pip install astor'"
    ) from e

class DocstringUtils:
    """Utility methods for extracting docstring and metadata from AST nodes."""

    @staticmethod
    def extract_metadata(node: ast.AST) -> Dict[str, Any]:
        """Extract common metadata and docstring info from an AST node."""
        metadata = {
            "name": getattr(node, "name", None),
            "lineno": getattr(node, "lineno", None),
            "type": "async function" if isinstance(node, ast.AsyncFunctionDef) else "function" if isinstance(node, ast.FunctionDef) else "class",
            "docstring_info": DocstringUtils.extract_docstring_info(node),
            "decorators": [get_node_name(d) for d in getattr(node, "decorator_list", [])],
            "bases": [get_node_name(base) for base in getattr(node, "bases", [])] if isinstance(node, ast.ClassDef) else [],
            "raises": DocstringUtils.extract_raises(node)
        }
        return metadata

    @staticmethod
    def extract_docstring_info(node: ast.AST) -> Dict[str, Any]:
        """Extract detailed docstring information."""
        try:
            docstring = ast.get_docstring(node) or ""
            args = DocstringUtils.extract_function_args(node) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else []
            returns = DocstringUtils.extract_return_info(node) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else {}
            return {
                "docstring": docstring,
                "args": args,
                "returns": returns,
            }
        except Exception as e:
            logger.error("Error extracting docstring info: %s", e)
            return {"error": f"Extraction failed due to: {e}"}

    @staticmethod
    def extract_function_args(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict[str, Any]]:
        """Extract function argument information."""
        args = []
        try:
            for arg in node.args.args:
                args.append({
                    "name": arg.arg,
                    "type": get_node_name(arg.annotation) if arg.annotation else "Any",
                    "description": "",  # Placeholder
                })
        except Exception as e:
            logger.error("Error extracting function arguments: %s", e)
        return args

    @staticmethod
    def extract_return_info(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        """Extract return type information."""
        if node.returns:
            return {
                "type": get_node_name(node.returns),
                "description": "",  # Placeholder
            }
        return {"type": "Any", "description": ""}

    @staticmethod
    def extract_raises(node: ast.AST) -> List[Dict[str, Any]]:
        """Extract exception information from the AST."""
        raises = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Raise) and child.exc:
                exc_name = get_node_name(child.exc)
                if exc_name:
                    raises.add(exc_name)
        return [{"exception": exc, "description": ""} for exc in raises]

    @staticmethod
    def get_exception_name(node: ast.AST) -> Optional[str]:
        """Get the name of an exception node."""
        try:
            if isinstance(node, ast.Call):
                return get_node_name(node.func)
            elif isinstance(node, (ast.Name, ast.Attribute)):
                return get_node_name(node)
            return "Exception"
        except Exception as e:
            logger.error(f"Error getting exception name: {e}", exc_info=True)
            return None
