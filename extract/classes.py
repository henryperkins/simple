import ast
from typing import Any, Dict, List
from .base import BaseExtractor
from .functions import FunctionExtractor
from core.logging.setup import LoggerSetup

# Initialize a logger specifically for this module
logger = LoggerSetup.get_logger("extract.classes")

class ClassExtractor(BaseExtractor):
    """
    Extractor for class details from AST nodes.
    """

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
                "instance_variables": self.extract_instance_variables(),
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
                    method_extractor = FunctionExtractor(item, self.content)
                    method_info = method_extractor.extract_details()
                    methods.append(method_info)
                    logger.debug(f"Extracted method: {method_info['name']}")
            return methods
        except Exception as e:
            logger.error(f"Error extracting methods in class {self.node.name}: {e}")
            return methods

    def extract_attributes(self) -> List[Dict[str, str]]:
        """
        Extract class-level attributes (annotated assignments).

        Returns:
            List[Dict[str, str]]: A list of attribute details.
        """
        attributes = []
        try:
            for item in self.node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    attr_name = item.target.id
                    attr_type = self.get_annotation(item.annotation)
                    attributes.append({
                        "name": attr_name,
                        "type": attr_type
                    })
                    logger.debug(f"Extracted attribute: {attr_name} with type {attr_type}")
            return attributes
        except Exception as e:
            logger.error(f"Error extracting attributes in class {self.node.name}: {e}")
            return attributes

    def extract_instance_variables(self) -> List[Dict[str, Any]]:
        """
        Extract instance variables initialized within methods.

        Returns:
            List[Dict[str, Any]]: A list of instance variable details.
        """
        instance_vars = []
        try:
            for item in self.node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                                    var_name = target.attr
                                    var_line = stmt.lineno
                                    instance_vars.append({
                                        "name": var_name,
                                        "line_number": var_line
                                    })
                                    logger.debug(f"Extracted instance variable: {var_name} at line {var_line}")
            return instance_vars
        except Exception as e:
            logger.error(f"Error extracting instance variables in class {self.node.name}: {e}")
            return instance_vars

    def extract_base_classes(self) -> List[str]:
        """
        Extract base classes of the class.

        Returns:
            List[str]: A list of base class names.
        """
        base_classes = []
        try:
            for base in self.node.bases:
                base_name = self.get_annotation(base)
                base_classes.append(base_name)
                logger.debug(f"Extracted base class: {base_name}")
            return base_classes
        except Exception as e:
            logger.error(f"Error extracting base classes for class {self.node.name}: {e}")
            return base_classes