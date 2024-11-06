import ast
from typing import Any, Dict, List
from .functions import FunctionExtractor
from .utils import get_annotation
from logging_utils import setup_logger

# Initialize a logger for this module
logger = setup_logger("classes")

class ClassExtractor:
    """Handles extraction of class details from AST nodes."""

    def __init__(self, node: ast.ClassDef, content: str):
        self.node = node
        self.content = content
        logger.debug(f"Initialized ClassExtractor for class: {node.name}")

    def extract_details(self) -> Dict[str, Any]:
        """Extract comprehensive information from a class node."""
        details = {
            "name": self.node.name,
            "docstring": ast.get_docstring(self.node) or "",
            "methods": self.extract_methods(),
            "attributes": self.extract_attributes(),
            "base_classes": self.extract_base_classes(),
        }
        logger.debug(f"Extracted details for class {self.node.name}: {details}")
        return details

    def extract_methods(self) -> List[Dict[str, Any]]:
        """Extract methods from the class."""
        methods = []
        for item in self.node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = FunctionExtractor(item, self.content).extract_details()
                methods.append(method_info)
                logger.debug(f"Extracted method: {method_info['name']}")
        return methods

    def extract_attributes(self) -> List[Dict[str, str]]:
        """Extract class attributes."""
        attributes = []
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

    def extract_base_classes(self) -> List[str]:
        """Extract base classes."""
        base_classes = [base.id for base in self.node.bases if isinstance(base, ast.Name)]
        logger.debug(f"Extracted base classes for {self.node.name}: {base_classes}")
        return base_classes
