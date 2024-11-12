import ast
from typing import Dict, Any, List
from core.logger import LoggerSetup
from extract.base import BaseExtractor

logger = LoggerSetup.get_logger("classes")

class ClassExtractor(BaseExtractor):
    """Extractor for class definitions in AST."""

    def extract_details(self) -> Dict[str, Any]:
        """Extract details of the class."""
        details = self._get_empty_details()
        try:
            if isinstance(self.node, ast.ClassDef):
                base_classes = []
                for base in self.node.bases:
                    if isinstance(base, ast.Name):
                        base_classes.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_classes.append(self._get_full_qualified_name(base))
                
                details.update({
                    "name": getattr(self.node, 'name', 'unknown'),
                    "docstring": self.get_docstring(),
                    "base_classes": base_classes,
                    "methods": self.extract_methods(),
                    "attributes": self.extract_attributes(),
                    "instance_variables": self.extract_instance_variables(),
                    "summary": self._generate_summary(),
                    "changelog": []
                })
        except Exception as e:
            logger.error(f"Error extracting class details: {e}")
        return details

    def _get_full_qualified_name(self, node: ast.Attribute) -> str:
        """Helper function to get the full qualified name of an attribute."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return '.'.join(reversed(parts))


    def extract_methods(self) -> List[Dict[str, Any]]:
        """Extract methods of the class."""
        methods = []
        try:
            if isinstance(self.node, ast.ClassDef):
                for node in self.node.body:
                    if isinstance(node, ast.FunctionDef):
                        method_info = {
                            "name": node.name,
                            "docstring": ast.get_docstring(node) or "",
                            "params": [{"name": arg.arg, "type": "Any"} for arg in node.args.args],
                            "returns": {"type": "None", "description": ""},
                            "line_number": node.lineno,
                            "end_line_number": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                            "code": self.get_source_segment(node),
                            "is_async": isinstance(node, ast.AsyncFunctionDef),
                            "is_generator": any(isinstance(n, ast.Yield) for n in ast.walk(node)),
                            "is_recursive": any(n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == node.name),
                            "summary": "",
                            "changelog": []
                        }
                        methods.append(method_info)
        except Exception as e:
            logger.error(f"Error extracting methods: {e}")
        return methods

    def extract_attributes(self) -> List[Dict[str, Any]]:
        """Extract attributes of the class."""
        attributes = []
        try:
            if isinstance(self.node, ast.ClassDef):
                for node in self.node.body:
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                attributes.append({
                                    "name": target.id,
                                    "type": "Any",
                                    "line_number": target.lineno
                                })
        except Exception as e:
            logger.error(f"Error extracting attributes: {e}")
        return attributes

    def extract_instance_variables(self) -> List[Dict[str, Any]]:
        """Extract instance variables of the class."""
        instance_vars = []
        try:
            if isinstance(self.node, ast.ClassDef):
                for node in self.node.body:
                    if isinstance(node, ast.FunctionDef):
                        for subnode in ast.walk(node):
                            if isinstance(subnode, ast.Assign):
                                for target in subnode.targets:
                                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                                        instance_vars.append({
                                            "name": target.attr,
                                            "line_number": target.lineno
                                        })
        except Exception as e:
            logger.error(f"Error extracting instance variables: {e}")
        return instance_vars

    def _generate_summary(self) -> str:
        """Generate a summary of the class."""
        parts = []
        try:
            if isinstance(self.node, ast.ClassDef):
                if self.node.bases:
                    parts.append(f"Base classes: {', '.join(base.id for base in self.node.bases if isinstance(base, ast.Name))}")
                if len(self.extract_methods()) > 10:
                    parts.append("⚠️ High number of methods")
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            parts.append("Error generating complete summary")
        
        return " | ".join(parts)

def extract_classes_from_ast(tree: ast.AST, content: str) -> List[Dict[str, Any]]:
    """
    Extract class definitions from the AST of a Python file.

    Args:
        tree (ast.AST): The abstract syntax tree of the Python file.
        content (str): The content of the Python file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing class information.
    """
    logger.debug("Starting extraction of classes from AST")
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            logger.debug(f"Found class: {node.name}")
            class_info = {
                "name": node.name,
                "base_classes": [base.id for base in node.bases if isinstance(base, ast.Name)],
                "methods": [],
                "attributes": [],
                "instance_variables": [],
                "summary": "",
                "changelog": []
            }
            classes.append(class_info)

    logger.debug("Class extraction complete")
    return classes
