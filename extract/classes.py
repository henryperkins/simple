import ast
from typing import Any, Dict, List
from .functions import FunctionExtractor
from .utils import get_annotation
from logging_utils import setup_logger

# Initialize a logger for this module
logger = setup_logger("extract.classes")

class ClassExtractor:
    """
    Handles extraction of class details from AST nodes.

    Args:
        node (ast.ClassDef): The class definition node.
        content (str): The source code content.
    """

    def __init__(self, node: ast.ClassDef, content: str):
        self.node = node
        self.content = content
        logger.debug(f"Initialized ClassExtractor for class: {node.name}")

    def extract_details(self) -> Dict[str, Any]:
        """
        Extract comprehensive information from a class node.

        Returns:
            Dict[str, Any]: The extracted class details.
        """
        try:
            details = {
                "name": self.node.name,
                "docstring": ast.get_docstring(self.node) or "",
                "methods": self.extract_methods(),
                "attributes": self.extract_attributes(),
                "base_classes": self.extract_base_classes(),
            }
            logger.debug(f"Extracted details for class {self.node.name}: {details}")
            return details
        except Exception as e:
            logger.error(f"Error extracting details for class {self.node.name}: {e}")
            return {}

    def extract_methods(self) -> List[Dict[str, Any]]:
        """
        Extract methods from the class.

        Returns:
            List[Dict[str, Any]]: A list of method details.
        """
        methods = []
        try:
            for item in self.node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_info = FunctionExtractor(item, self.content).extract_details()
                    methods.append(method_info)
                    logger.debug(f"Extracted method: {method_info['name']}")
            return methods
        except Exception as e:
            logger.error(f"Error extracting methods in class {self.node.name}: {e}")
            return methods

    def extract_attributes(self) -> List[Dict[str, str]]:
        """
        Extract class attributes.

        Returns:
            List[Dict[str, str]]: A list of attribute details.
        """
        attributes = []
        try:
            for item in self.node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    attr_name = item.target.id
                    attr_type = get_annotation(item.annotation)
                    attributes.append({
                        "name": attr_name,
                        "type": attr_type
                    })
                    logger.debug(f"Extracted attribute: {attr_name} with type {attr_type}")
            return attributes
        except Exception as e:
            logger.error(f"Error extracting attributes in class {self.node.name}: {e}")
            return attributes

    def extract_base_classes(self) -> List[str]:
        """
        Extract base classes.

        Returns:
            List[str]: A list of base class names.
        """
        try:
            base_classes = [get_annotation(base) for base in self.node.bases]
            logger.debug(f"Extracted base classes for {self.node.name}: {base_classes}")
            return base_classes
        except Exception as e:
            logger.error(f"Error extracting base classes for {self.node.name}: {e}")
            return []