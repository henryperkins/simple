from typing import Any, Dict, List, Optional
from core.logger import LoggerSetup
from extract.base import BaseExtractor
from extract.functions import FunctionExtractor
from extract.utils import get_annotation

# Initialize logger for this module
logger = LoggerSetup.get_logger("extract.classes")

class ClassExtractor(BaseExtractor):
    def extract_details(self) -> Dict[str, Any]:
        try:
            details = {
                "name": self.node.name,
                "docstring": self.get_docstring(),
                "methods": self.extract_methods(),
                "attributes": self.extract_attributes(),
                "instance_variables": self.extract_instance_variables(),
                "base_classes": self.extract_base_classes(),
                "summary": self._generate_summary(),
                "changelog": []
            }
            return details
        except Exception as e:
            logger.error(f"Error extracting class details: {e}")
            return self._get_empty_details()

    def extract_methods(self) -> List[Dict[str, Any]]:
        methods = []
        try:
            for node in self.node.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_extractor = FunctionExtractor(node, self.content)
                    methods.append(method_extractor.extract_details())
        except Exception as e:
            logger.error(f"Error extracting methods: {e}")
        return methods

    def extract_attributes(self) -> List[Dict[str, Any]]:
        attributes = []
        try:
            for node in self.node.body:
                if isinstance(node, ast.AnnAssign):
                    attributes.append({
                        "name": node.target.id,
                        "type": self.get_annotation(node.annotation)
                    })
        except Exception as e:
            logger.error(f"Error extracting attributes: {e}")
        return attributes

    def extract_instance_variables(self) -> List[Dict[str, Any]]:
        instance_vars = []
        try:
            for node in self.node.body:
                if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                    for stmt in node.body:
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                                    instance_vars.append({
                                        "name": target.attr,
                                        "line_number": target.lineno
                                    })
        except Exception as e:
            logger.error(f"Error extracting instance variables: {e}")
        return instance_vars

    def extract_base_classes(self) -> List[str]:
        base_classes = []
        try:
            for base in self.node.bases:
                if isinstance(base, ast.Name):
                    base_classes.append(base.id)
        except Exception as e:
            logger.error(f"Error extracting base classes: {e}")
        return base_classes

    def _get_empty_details(self) -> Dict[str, Any]:
        return {
            "name": "",
            "docstring": "",
            "methods": [],
            "attributes": [],
            "instance_variables": [],
            "base_classes": [],
            "summary": "",
            "changelog": []
        }

    def _generate_summary(self) -> str:
        parts = []
        if self.node.bases:
            parts.append(f"Inherits from {', '.join(base.id for base in self.node.bases if isinstance(base, ast.Name))}")
        return " | ".join(parts)