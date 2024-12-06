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
- This module requires a `get_node_name` function from `core.utils`.
"""

import ast
from typing import Dict, Any, List, Optional, Union

from core.logger import LoggerSetup
from core.utils import get_node_name  # Ensure this import is present

logger = LoggerSetup.get_logger(__name__)

class DocstringUtils:
    """Utility class providing methods for extracting docstring and metadata from AST nodes.

    This class contains static methods that can be used to parse and extract information
    from Python AST nodes, such as docstrings, function arguments, return types, and exceptions raised.
    """

    @staticmethod
    def extract_metadata(node: ast.AST) -> Dict[str, Any]:
        """Extract common metadata and docstring information from an AST node.

        This method extracts various metadata from a Python AST node, such as the node's name,
        line number, type (function, async function, or class), docstring information, decorators,
        base classes, and exceptions raised.

        Args:
            node (ast.AST): The AST node from which to extract metadata. This should be an instance
                of ast.FunctionDef, ast.AsyncFunctionDef, or ast.ClassDef.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted metadata with the following keys:
                - 'name' (str): The name of the node (e.g., function or class name).
                - 'lineno' (int): The starting line number of the node in the source code.
                - 'type' (str): The type of the node ('function', 'async function', or 'class').
                - 'docstring_info' (Dict): Detailed docstring information extracted by `extract_docstring_info`.
                - 'decorators' (List[str]): A list of decorator names applied to the node.
                - 'bases' (List[str]): A list of base class names if the node is a class.
                - 'raises' (List[Dict[str, Any]]): A list of exceptions that the node may raise, with each exception
                    represented as a dictionary with keys 'exception' and 'description'.

        Raises:
            None
        """
        if isinstance(node, ast.AsyncFunctionDef):
            node_type = "async function"
        elif isinstance(node, ast.FunctionDef):
            node_type = "function"
        elif isinstance(node, ast.ClassDef):
            node_type = "class"
        else:
            node_type = "unknown"

        if isinstance(node, ast.ClassDef):
            bases = [get_node_name(base) for base in getattr(node, "bases", [])]
        else:
            bases = []

        metadata = {
            "name": getattr(node, "name", None),
            "lineno": getattr(node, "lineno", None),
            "type": node_type,
            "docstring_info": DocstringUtils.extract_docstring_info(node),
            "decorators": [get_node_name(d) for d in getattr(node, "decorator_list", [])],
            "bases": bases,
            "raises": DocstringUtils.extract_raises(node)
        }
        return metadata

    @staticmethod
    def extract_docstring_info(node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> Dict[str, Any]:
        """Extract detailed docstring information from an AST node.

        This method retrieves the docstring of a given AST node and extracts information such as
        function arguments and return type if the node represents a function or an async function.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]): The AST node from which to extract docstring information.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted docstring information with the following keys:
                - 'docstring' (str): The docstring of the node.
                - 'args' (List[Dict[str, Any]]): A list of argument information if the node is a function,
                    with each argument represented as a dictionary containing:
                    - 'name' (str): The argument name.
                    - 'type' (str): The type annotation of the argument, or 'Any' if not specified.
                    - 'description' (str): A placeholder for argument description (empty string by default).
                - 'returns' (Dict[str, Any]): Return type information if the node is a function, containing:
                    - 'type' (str): The return type annotation, or 'Any' if not specified.
                    - 'description' (str): A placeholder for return description (empty string by default).

        Raises:
            None
        """
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
                "raises": DocstringUtils.extract_raises(node)  # Ensure raises are included
            }
        except Exception as e:
            logger.error("Error extracting docstring info: %s", e)
            return {"error": f"Extraction failed due to: {e}"}

    @staticmethod
    def extract_function_args(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict[str, Any]]:
        """Extract argument information from a function AST node.

        This method retrieves the list of arguments from a function or async function AST node,
        along with their type annotations and placeholders for descriptions.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function AST node from which to extract arguments.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing each argument, with the following keys:
                - 'name' (str): The name of the argument.
                - 'type' (str): The type annotation of the argument, or 'Any' if not specified.
                - 'description' (str): A placeholder for the argument description (empty string by default).

        Raises:
            None
        """
        args = []
        try:
            for arg in node.args.args:
                arg_type = get_node_name(arg.annotation) if arg.annotation else "Any"
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
        """Extract return type information from a function AST node.

        This method retrieves the return type annotation from a function or async function AST node.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function AST node from which to extract return type information.

        Returns:
            Dict[str, Any]: A dictionary containing the return type information with the following keys:
                - 'type' (str): The return type annotation, or 'Any' if not specified.
                - 'description' (str): A placeholder for the return description (empty string by default).

        Raises:
            None
        """
        if node.returns:
            return {
                "type": get_node_name(node.returns),
                "description": "",  # Placeholder
            }
        return {"type": "Any", "description": ""}

    @staticmethod
    def extract_raises(node: ast.AST) -> List[Dict[str, Any]]:
        """Extract exception information from an AST node.

        This method walks through the AST of the given node and collects the names of exceptions
        that are raised within the node. It returns a list of exceptions with placeholders for descriptions.

        Args:
            node (ast.AST): The AST node from which to extract exception information.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing each exception that may be raised, with the following keys:
                - 'exception' (str): The name of the exception.
                - 'description' (str): A placeholder for the exception description (empty string by default).

        Raises:
            None
        """
        raises = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Raise) and child.exc:
                exc_name = get_node_name(child.exc)
                if exc_name:
                    raises.add(exc_name)
        return [{"exception": exc, "description": ""} for exc in raises]

    @staticmethod
    def get_exception_name(node: ast.AST) -> Optional[str]:
        """Get the name of an exception from an exception AST node.

        This method attempts to retrieve the name of an exception from an AST node representing a raised exception.

        Args:
            node (ast.AST): The AST node representing the raised exception (e.g., an ast.Call, ast.Name, or ast.Attribute node).

        Returns:
            Optional[str]: The name of the exception if it can be determined, otherwise None.
        """
        try:
            if isinstance(node, ast.Call):
                return get_node_name(node.func)
            elif isinstance(node, (ast.Name, ast.Attribute)):
                return get_node_name(node)
            return "Exception"
        except Exception as e:
            logger.error(f"Error getting exception name: {e}", exc_info=True)
            return None
