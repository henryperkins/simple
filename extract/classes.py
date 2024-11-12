# extract/classes.py

from typing import Dict, Any, List
import ast
from core.logger import LoggerSetup
from extract.base import BaseExtractor
from extract.functions import FunctionExtractor
from extract.utils import get_annotation

logger = LoggerSetup.get_logger("extract.classes")

class ClassExtractor(BaseExtractor):
    """Extractor for class definitions in AST."""

    def extract_details(self) -> Dict[str, Any]:
        """Extract details of the class."""
        details = self._get_empty_details()
        try:
            details.update({
                "name": self.node.name,
                "docstring": self.get_docstring(),
                "methods": self.extract_methods(),
                "attributes": self.extract_attributes(),
                "instance_variables": self.extract_instance_variables(),
                "base_classes": self.extract_base_classes(),
                "summary": self._generate_summary(),
                "changelog": []  # Initialize changelog
            })
        except Exception as e:
            logger.error(f"Error extracting class details: {e}")
        return details

    def extract_methods(self) -> List[Dict[str, Any]]:
        """Extract methods from the class."""
        methods = []
        try:
            for node in self.node.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    extractor = FunctionExtractor(node, self.content)
                    method_details = extractor.extract_details()
                    methods.append(method_details)
        except Exception as e:
            logger.error(f"Error extracting methods: {e}")
        return methods

    def extract_attributes(self) -> List[Dict[str, Any]]:
        """Extract attributes from the class."""
        attributes = []
        try:
            for node in self.node.body:
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    attributes.append({
                        "name": node.target.id,
                        "type": get_annotation(node.annotation),
                        "line_number": node.lineno
                    })
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            attributes.append({
                                "name": target.id,
                                "type": "Any",
                                "line_number": node.lineno
                            })
        except Exception as e:
            logger.error(f"Error extracting attributes: {e}")
        return attributes

    def extract_instance_variables(self) -> List[Dict[str, Any]]:
        """Extract instance variables from the class."""
        instance_vars = []
        try:
            for node in self.node.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
                    for sub_node in ast.walk(node):
                        if isinstance(sub_node, ast.Attribute) and isinstance(sub_node.value, ast.Name):
                            if sub_node.value.id == "self":
                                instance_vars.append({
                                    "name": sub_node.attr,
                                    "line_number": sub_node.lineno
                                })
        except Exception as e:
            logger.error(f"Error extracting instance variables: {e}")
        return instance_vars

    def extract_base_classes(self) -> List[str]:
        """Extract base classes of the class."""
        base_classes = []
        try:
            for base in self.node.bases:
                if isinstance(base, ast.Name):
                    base_classes.append(base.id)
                elif isinstance(base, ast.Attribute):
                    parts = []
                    node = base
                    while isinstance(node, ast.Attribute):
                        parts.append(node.attr)
                        node = node.value
                    if isinstance(node, ast.Name):
                        parts.append(node.id)
                        base_classes.append(".".join(reversed(parts)))
        except Exception as e:
            logger.error(f"Error extracting base classes: {e}")
        return base_classes

    def _generate_summary(self) -> str:
        """Generate a summary of the class."""
        parts = []
        try:
            if self.node.bases:
                base_classes = self.extract_base_classes()
                parts.append(f"Inherits from: {', '.join(base_classes)}")
            
            method_count = len(self.extract_methods())
            attr_count = len(self.extract_attributes())
            instance_var_count = len(self.extract_instance_variables())
            
            parts.append(f"Methods: {method_count}")
            parts.append(f"Attributes: {attr_count}")
            parts.append(f"Instance Variables: {instance_var_count}")
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
        
        return " | ".join(parts)
