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
        if isinstance(node, ast.AsyncFunctionDef):
            node_type = "async function"
        elif isinstance(node, ast.FunctionDef):
            node_type = "function"
        elif isinstance(node, ast.ClassDef):
            node_type = "class"
        else:
            node_type = "unknown"

        if isinstance(node, ast.ClassDef):
            bases = [DocstringUtils._get_node_name(base) for base in getattr(node, "bases", [])]
        else:
            bases = []

        metadata = {
            "name": getattr(node, "name", None),
            "lineno": getattr(node, "lineno", None),
            "type": node_type,
            "docstring_info": DocstringUtils.extract_docstring_info(node),
            "decorators": [DocstringUtils._get_node_name(d) for d in getattr(node, "decorator_list", [])],
            "bases": bases,
            "raises": DocstringUtils.extract_raises(node)
        }
        return metadata

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
    def _get_node_name(node: Optional[ast.AST]) -> str:
        """Helper method to get the name of an AST node using NodeNameVisitor."""
        if node is None:
            return "Any"
        visitor = NodeNameVisitor()
        visitor.visit(node)
        return visitor.name

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