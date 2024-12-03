"""Utility functions for code extraction.

This module provides utility functions and classes for working with the Abstract Syntax Tree (AST)
in Python source code. It includes functions for adding parent references, extracting names and
source segments, and identifying variables and constants.
"""

import ast
from typing import Optional, Dict, Any, List, Union
import importlib.util
import sys
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class ASTUtils:
    """Utility class for AST operations.

    Provides methods for manipulating and extracting information from AST nodes.
    """

    def __init__(self):
        """Initialize AST utilities."""
        self.logger = logger
        self.logger.debug("Initialized ASTUtils")

    def add_parents(self, node: ast.AST) -> None:
        """Add parent references to AST nodes.

        This method traverses the AST and sets a 'parent' attribute on each node,
        pointing to its parent node.

        Args:
            node (ast.AST): The root AST node.
        """
        self.logger.debug("Adding parent references to AST nodes")
        for child in ast.iter_child_nodes(node):
            setattr(child, 'parent', node)
            self.add_parents(child)

    def get_name(self, node: Optional[ast.AST]) -> str:
        """Get string representation of a node.

        Converts an AST node into a string representation, handling different types
        of nodes such as names, attributes, subscripts, calls, tuples, and lists.

        Args:
            node (Optional[ast.AST]): The AST node to analyze.

        Returns:
            str: The string representation of the node.
        """
        self.logger.debug(f"Getting name for node: {ast.dump(node) if node else 'None'}")
        if node is None:
            return "Any"

        try:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return f"{self.get_name(node.value)}.{node.attr}"
            elif isinstance(node, ast.Subscript):
                value = self.get_name(node.value)
                slice_val = self.get_name(node.slice)
                return f"{value}[{slice_val}]"
            elif isinstance(node, ast.Call):
                return f"{self.get_name(node.func)}()"
            elif isinstance(node, (ast.Tuple, ast.List)):
                elements = ', '.join(self.get_name(e) for e in node.elts)
                return f"({elements})" if isinstance(node, ast.Tuple) else f"[{elements}]"
            elif hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                import astor
                return astor.to_source(node).strip()
        except Exception as e:
            self.logger.error(f"Error getting name from node {type(node).__name__}: {e}", exc_info=True)
            return f'Unknown<{type(node).__name__}>'

    def get_source_segment(self, node: ast.AST, include_source: bool = True) -> Optional[str]:
        """Get source code segment for a node.

        Retrieves the source code segment corresponding to an AST node.

        Args:
            node (ast.AST): The AST node to analyze.
            include_source (bool): Whether to include the source code segment.

        Returns:
            Optional[str]: The source code segment as a string, or None if not included.
        """
        self.logger.debug(f"Getting source segment for node: {ast.dump(node)}")
        if not include_source:
            return None
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                import astor
                return astor.to_source(node).strip()
        except Exception as e:
            self.logger.error(f"Error getting source segment: {e}", exc_info=True)
            return f"<unparseable: {type(node).__name__}>"

    def extract_variables(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract variables from the AST.

        Identifies variable assignments in the AST and extracts relevant information.

        Args:
            tree (ast.AST): The AST tree to analyze.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing variable information.
        """
        self.logger.info("Extracting variables from AST")
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        var_info = self._create_variable_info(target, node)
                        if var_info:
                            variables.append(var_info)
                            self.logger.debug(f"Extracted variable: {var_info['name']}")
        return variables

    def extract_constants(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract module-level constants.

        Identifies constant assignments in the AST and extracts relevant information.

        Args:
            tree (ast.AST): The AST tree to analyze.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing constant information.
        """
        self.logger.info("Extracting constants from AST")
        constants = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        try:
                            constant_info = self._create_constant_info(target, node)
                            if constant_info:
                                constants.append(constant_info)
                                self.logger.debug(f"Extracted constant: {constant_info['name']}")
                        except Exception as e:
                            self.logger.error(f"Error extracting constant {target.id}: {e}", exc_info=True)
        return constants

    def _create_variable_info(self, target: ast.Name, node: Union[ast.Assign, ast.AnnAssign]) -> Optional[Dict[str, Any]]:
        """Create variable information dictionary.

        Constructs a dictionary containing information about a variable assignment.

        Args:
            target (ast.Name): The target variable node.
            node (Union[ast.Assign, ast.AnnAssign]): The assignment node.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing variable information, or None if an error occurs.
        """
        self.logger.debug(f"Creating variable info for target: {target.id}")
        try:
            var_name = target.id
            annotation = None
            value = None

            if isinstance(node, ast.AnnAssign) and node.annotation:
                annotation = self.get_name(node.annotation)
            if hasattr(node, 'value') and node.value:
                try:
                    value = self.get_name(node.value)
                except Exception as e:
                    self.logger.error(f"Failed to get value for {var_name}: {e}", exc_info=True)
                    value = "UnknownValue"

            return {
                'name': var_name,
                'type': annotation or "UnknownType",
                'value': value
            }
        except Exception as e:
            self.logger.error(f"Error creating variable info: {e}", exc_info=True)
            return None

    def _create_constant_info(self, target: ast.Name, node: ast.Assign) -> Optional[Dict[str, Any]]:
        """Create constant information dictionary.

        Constructs a dictionary containing information about a constant assignment.

        Args:
            target (ast.Name): The target constant node.
            node (ast.Assign): The assignment node.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing constant information, or None if an error occurs.
        """
        self.logger.debug(f"Creating constant info for target: {target.id}")
        try:
            value = self.get_name(node.value)
            try:
                value_type = type(ast.literal_eval(node.value)).__name__
            except Exception:
                value_type = "UnknownType"
            return {
                'name': target.id,
                'value': value,
                'type': value_type
            }
        except Exception as e:
            self.logger.error(f"Error creating constant info: {e}", exc_info=True)
            return None