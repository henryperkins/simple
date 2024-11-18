"""
Base Extraction Module

This module provides a base class for extracting information from AST nodes.
It defines common functionality and utility methods that can be used by subclasses
to extract specific details from Python source code.

Version: 1.0.0
Author: Development Team
"""

import ast
from abc import ABC, abstractmethod
from typing import Generator, Optional, Dict, Any, List
from core.logger import log_info, log_error, log_debug

class BaseExtractor(ABC):
    """Base class for extracting information from AST nodes.

    Provides common functionality and utility methods for subclasses to extract
    specific details from Python source code.
    """

    def __init__(self, source_code: str):
        """Initializes the BaseExtractor with the source code and parses it into an AST.

        Args:
            source_code (str): The source code to parse.

        Raises:
            SyntaxError: If the source code cannot be parsed into an AST.
        """
        try:
            log_debug("Attempting to parse source code into AST.")
            self.tree = ast.parse(source_code)
            log_info("AST successfully parsed for extraction.")
        except SyntaxError as e:
            log_error(f"Failed to parse source code: {e}")
            raise e

    def walk_tree(self) -> Generator[ast.AST, None, None]:
        """Walks the AST and yields all nodes.

        Yields:
            ast.AST: The next node in the AST.
        """
        log_debug("Walking through the AST nodes.")
        for node in ast.walk(self.tree):
            log_debug(f"Yielding AST node: {type(node).__name__}")
            yield node

    def extract_docstring(self, node: ast.AST) -> Optional[str]:
        """Extracts the docstring from an AST node.

        Args:
            node (ast.AST): The node to extract the docstring from.

        Returns:
            Optional[str]: The extracted docstring, or None if not present.
        """
        try:
            log_debug(f"Extracting docstring from node: {type(node).__name__}")
            docstring = ast.get_docstring(node)
            if docstring:
                log_info(f"Docstring extracted from node: {type(node).__name__}")
            else:
                log_info(f"No docstring found in node: {type(node).__name__}")
            return docstring
        except Exception as e:
            log_error(f"Failed to extract docstring: {e}")
            return None

    def _get_type_annotation(self, arg: ast.arg) -> str:
        """Gets type annotation as a string.

        Args:
            arg (ast.arg): The argument node to extract type annotation from.

        Returns:
            str: The type annotation, or "Any" if not specified.
        """
        annotation = "Any"
        if arg.annotation:
            try:
                annotation = ast.unparse(arg.annotation)
            except Exception as e:
                log_error(f"Error unparsing annotation for argument '{arg.arg}': {e}")
        log_debug(f"Type annotation for '{arg.arg}': {annotation}")
        return annotation

    def _has_default(self, arg: ast.arg, node: ast.FunctionDef) -> bool:
        """Checks if an argument has a default value.

        Args:
            arg (ast.arg): The argument node to check.
            node (ast.FunctionDef): The function node containing the argument.

        Returns:
            bool: True if the argument has a default value, False otherwise.
        """
        return arg in node.args.defaults

    def _get_default_value(self, arg: ast.arg, node: ast.FunctionDef) -> Optional[str]:
        """Gets default value as a string if it exists.

        Args:
            arg (ast.arg): The argument node to extract default value from.
            node (ast.FunctionDef): The function node containing the argument.

        Returns:
            Optional[str]: The default value or None if not present.
        """
        try:
            index = node.args.args.index(arg) - (len(node.args.args) - len(node.args.defaults))
            if index >= 0:
                return ast.unparse(node.args.defaults[index])
        except Exception as e:
            log_error(f"Error extracting default value for argument '{arg.arg}': {e}")
        return None

    def _extract_common_details(self, node: ast.AST) -> Dict[str, Any]:
        """Extracts common details from an AST node.

        Args:
            node (ast.AST): The AST node to extract details from.

        Returns:
            Dict[str, Any]: A dictionary containing common details such as name, docstring, and line number.
        """
        return {
            'name': getattr(node, 'name', '<unknown>'),
            'docstring': self.extract_docstring(node),
            'lineno': getattr(node, 'lineno', 0)
        }

    def _extract_decorators(self, node: ast.FunctionDef) -> List[str]:
        """Extracts decorators from a function node.

        Args:
            node (ast.FunctionDef): The function node to extract decorators from.

        Returns:
            List[str]: A list of decorator names.
        """
        return [ast.unparse(decorator) for decorator in node.decorator_list]

    def _detect_exceptions(self, node: ast.FunctionDef) -> List[str]:
        """Detects exceptions that could be raised by the function.

        Args:
            node (ast.FunctionDef): The function node to analyze.

        Returns:
            List[str]: A list of exception names that could be raised.
        """
        exceptions = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                if isinstance(child.exc, ast.Name):
                    exceptions.add(child.exc.id)
                elif isinstance(child.exc, ast.Call) and isinstance(child.exc.func, ast.Name):
                    exceptions.add(child.exc.func.id)
        return list(exceptions)

    @abstractmethod
    def extract_details(self, node: ast.AST) -> dict:
        """Abstract method to extract details from a given AST node.

        Must be implemented by subclasses.

        Args:
            node (ast.AST): The node to extract details from.

        Returns:
            dict: A dictionary containing extracted details.
        """
        pass
