"""
This module provides utility functions and classes for extracting metadata and docstring
information from Python Abstract Syntax Trees (ASTs). It is useful for analyzing Python code,
retrieving function arguments, return types, raised exceptions, and other annotations directly
from the AST nodes. It handles both synchronous and asynchronous functions, classes, and methods,
and supports detailed introspection of function signatures and docstrings.

Features:
- Extract metadata such as name, line number, type (function, class, async function), decorators,
  bases (for classes)
- Retrieve and parse docstring information for AST nodes
- Extract function arguments along with their annotations
- Determine return type annotations and exception types that functions may raise

Dependencies:
- The 'astor' library is required for unparse operations on Python versions less than 3.9.

Note:
- This module requires a `NodeNameVisitor` class from `core.utils`.
"""

import ast
from typing import Dict, Any, List, Optional, Union

from core.logger import LoggerSetup
from core.utils import NodeNameVisitor

logger = LoggerSetup.get_logger(__name__)

class DocstringUtils:
    """Utility class providing methods for extracting docstring and metadata from AST nodes."""

    @staticmethod
    def extract_metadata(node: ast.AST) -> Dict[str, Any]:
        """Extract common metadata and docstring information from an AST node."""
        try:
            metadata = {
                "name": getattr(node, "name", None),
                "lineno": getattr(node, "lineno", None),
                "type": DocstringUtils._get_node_type(node),
                "docstring_info": DocstringUtils.extract_docstring_info(node),
                "decorators": [],
                "bases": [],
                "raises": []
            }

            # Handle decorators
            if hasattr(node, "decorator_list"):
                metadata["decorators"] = [
                    DocstringUtils._get_decorator_name(d) for d in node.decorator_list
                ]

            # Handle bases for classes
            if isinstance(node, ast.ClassDef):
                metadata["bases"] = [
                    DocstringUtils._get_node_name(base) for base in node.bases
                ]

            # Extract raises information
            metadata["raises"] = DocstringUtils.extract_raises(node)

            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {
                "name": getattr(node, "name", "unknown"),
                "lineno": getattr(node, "lineno", 0),
                "type": "unknown",
                "docstring_info": {"docstring": "", "args": [], "returns": {}, "raises": []},
                "decorators": [],
                "bases": [],
                "raises": []
            }

    @staticmethod
    def extract_docstring_info(node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> Dict[str, Any]:
        """Extract detailed docstring information from an AST node."""
        try:
            docstring = ast.get_docstring(node) or ""
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = DocstringUtils.extract_function_args(node)
                returns = DocstringUtils.extract_return_info(node)
            else:
                args = []
                returns = {}

            return {
                "docstring": docstring,
                "args": args,
                "returns": returns,
                "raises": DocstringUtils.extract_raises(node)
            }
        except Exception as e:
            logger.error("Error extracting docstring info: %s", e)
            return {"error": f"Extraction failed due to: {e}"}

    @staticmethod
    def extract_function_args(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict[str, Any]]:
        """Extract argument information from a function AST node."""
        args = []
        try:
            for arg in node.args.args:
                arg_type = DocstringUtils._get_node_name(arg.annotation) if arg.annotation else "Any"
                args.append({
                    "name": arg.arg,
                    "type": arg_type,
                    "description": "",  # Placeholder
                })
        except Exception as e:
            logger.error("Error extracting function arguments: %s", e)
        return args

    @staticmethod
    def extract_return_info(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        """Extract return type information from a function AST node."""
        if node.returns:
            return {
                "type": DocstringUtils._get_node_name(node.returns),
                "description": "",  # Placeholder
            }
        return {"type": "Any", "description": ""}

    @staticmethod
    def extract_raises(node: ast.AST) -> List[Dict[str, Any]]:
        """Extract exception information from an AST node."""
        raises = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Raise) and child.exc:
                exc_name = DocstringUtils._get_node_name(child.exc)
                if exc_name:
                    raises.add(exc_name)
        return [{"exception": exc, "description": ""} for exc in raises]

    @staticmethod
    def _get_decorator_name(node: ast.AST) -> str:
        """Extract decorator name using NodeNameVisitor."""
        visitor = NodeNameVisitor()
        try:
            visitor.visit(node)
            return visitor.name
        except Exception as e:
            logger.error(f"Error getting decorator name: {e}")
            return "unknown_decorator"

    @staticmethod
    def _get_node_type(node: ast.AST) -> str:
        """Determine the type of the node."""
        if isinstance(node, ast.AsyncFunctionDef):
            return "async function"
        elif isinstance(node, ast.FunctionDef):
            return "function"
        elif isinstance(node, ast.ClassDef):
            return "class"
        return "unknown"

    @staticmethod
    def _get_node_name(node: Optional[ast.AST]) -> str:
        """Use NodeNameVisitor to get node name."""
        if node is None:
            return "Any"
        visitor = NodeNameVisitor()
        try:
            visitor.visit(node)
            return visitor.name
        except Exception as e:
            logger.error(f"Error getting node name: {e}")
            return "unknown"

    @staticmethod
    def get_exception_name(node: ast.AST) -> Optional[str]:
        """Get the name of an exception from an exception AST node."""
        try:
            if isinstance(node, ast.Call):
                return DocstringUtils._get_node_name(node.func)
            elif isinstance(node, (ast.Name, ast.Attribute)):
                return DocstringUtils._get_node_name(node)
            return "Exception"
        except Exception as e:
            logger.error(f"Error getting exception name: {e}", exc_info=True)
            return None