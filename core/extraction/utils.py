"""Utility functions for code extraction."""

import ast
from typing import Optional, Dict, Any, List, Set, Union
import importlib.util
import sys
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class ASTUtils:
    """Utility class for AST operations."""

    def __init__(self):
        """Initialize AST utilities."""
        self.logger = logger
        self.logger.debug("Initialized ASTUtils")

    def add_parents(self, node: ast.AST) -> None:
        """
        Add parent references to AST nodes.

        Args:
            node (ast.AST): The root AST node.
        """
        self.logger.debug("Adding parent references to AST nodes")
        for child in ast.iter_child_nodes(node):
            setattr(child, 'parent', node)
            self.add_parents(child)

    def get_name(self, node: Optional[ast.AST]) -> str:
        """
        Get string representation of a node.

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
            elif hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                import astor
                return astor.to_source(node).strip()
        except Exception as e:
            self.logger.error(f"Error getting name from node {type(node).__name__}: {e}", exc_info=True)
            return 'Unknown'

    def get_source_segment(self, node: ast.AST, include_source: bool = True) -> Optional[str]:
        """
        Get source code segment for a node.

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
            return None

    def extract_variables(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Extract variables from the AST.

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
        """
        Extract module-level constants.

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
        """
        Create variable information dictionary.

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
                    value = "Unknown"

            return {
                'name': var_name,
                'type': annotation,
                'value': value
            }
        except Exception as e:
            self.logger.error(f"Error creating variable info: {e}", exc_info=True)
            return None

    def _create_constant_info(self, target: ast.Name, node: ast.Assign) -> Optional[Dict[str, Any]]:
        """
        Create constant information dictionary.

        Args:
            target (ast.Name): The target constant node.
            node (ast.Assign): The assignment node.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing constant information, or None if an error occurs.
        """
        self.logger.debug(f"Creating constant info for target: {target.id}")
        try:
            value = self.get_name(node.value)
            return {
                'name': target.id,
                'value': value,
                'type': type(ast.literal_eval(node.value)).__name__ if isinstance(node.value, ast.Constant) else None
            }
        except Exception as e:
            self.logger.error(f"Error creating constant info: {e}", exc_info=True)
            return None