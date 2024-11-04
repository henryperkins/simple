import ast
import logging
from typing import Any, Dict, List, Tuple

class CodeExtractor:
    """A class to extract classes and functions from Python source code using AST."""

    def extract_classes_and_functions_from_ast(
        self, tree: ast.AST, content: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract class and function details from an AST."""
        classes: List[Dict[str, Any]] = []
        functions: List[Dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(self.extract_class_details(node, content))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not self._is_method(node):
                functions.append(self.extract_function_details(node, content))
        
        return {"classes": classes, "functions": functions}

    def extract_class_details(self, node: ast.ClassDef, content: str) -> Dict[str, Any]:
        """Extract details from a class node."""
        try:
            class_name = node.name
            class_docstring = ast.get_docstring(node) or ""
            class_code = ast.get_source_segment(content, node)
            methods: List[Dict[str, Any]] = []
            nested_classes: List[Dict[str, Any]] = []
            for body_item in node.body:
                if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(self.extract_function_details(body_item, content))
                elif isinstance(body_item, ast.ClassDef):
                    nested_classes.append(self.extract_class_details(body_item, content))
            return {
                "name": class_name,
                "docstring": class_docstring,
                "code": class_code,
                "methods": methods,
                "nested_classes": nested_classes
            }
        except Exception as e:
            logging.warning(f"Error extracting class details: {e}")
            return {"name": "Unknown", "docstring": "", "code": "", "methods": [], "nested_classes": []}

    def extract_function_details(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Extract details from a function node."""
        try:
            function_name = node.name
            function_docstring = ast.get_docstring(node) or ""
            function_code = ast.get_source_segment(content, node)
            params = self._get_parameters(node)
            return_type = self._get_return_type(node)
            complexity_score = self.calculate_complexity(node)
            return {
                "name": function_name,
                "docstring": function_docstring,
                "code": function_code,
                "params": params,
                "return_type": return_type,
                "complexity_score": complexity_score
            }
        except Exception as e:
            logging.warning(f"Error extracting function details: {e}")
            return {"name": "Unknown", "docstring": "", "code": "", "params": [], "return_type": "Unknown", "complexity_score": None}

    def _get_parameters(self, node: ast.FunctionDef) -> List[Tuple[str, str]]:
        """Retrieve function parameters and their types."""
        params = []
        for arg in node.args.args:
            param_name = arg.arg
            param_type = self._get_annotation(arg.annotation)
            params.append((param_name, param_type))
        return params

    def _get_return_type(self, node: ast.FunctionDef) -> str:
        """Retrieve function return type."""
        return self._get_annotation(node.returns)

    def _get_annotation(self, annotation) -> str:
        """Convert AST annotation to string."""
        if annotation is None:
            return "None"
        elif isinstance(annotation, ast.Str):
            return annotation.s
        elif isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Subscript):
            value = self._get_annotation(annotation.value)
            slice_ = self._get_annotation(annotation.slice)
            return f"{value}[{slice_}]"
        elif isinstance(annotation, ast.Attribute):
            return f"{self._get_annotation(annotation.value)}.{annotation.attr}"
        else:
            return "Unknown"

    def _is_method(self, node: ast.FunctionDef) -> bool:
        """Determine if a function node is a method of a class."""
        return isinstance(getattr(node, 'parent', None), ast.ClassDef)

    def calculate_complexity(self, node: ast.AST) -> int:
        """Calculate the cyclomatic complexity score for a function or method."""
        complexity = 1  # Start with one for the function entry point
        try:
            for subnode in ast.walk(node):
                if isinstance(
                    subnode,
                    (
                        ast.If,
                        ast.For,
                        ast.While,
                        ast.Try,
                        ast.With,
                        ast.ListComp,
                        ast.DictComp,
                        ast.SetComp,
                        ast.GeneratorExp,
                    ),
                ):
                    complexity += 1
                elif isinstance(subnode, ast.BoolOp):
                    # Each boolean operation (and/or) adds branches
                    complexity += len(subnode.values) - 1
        except Exception as e:
            logging.warning(f"Error calculating complexity for function {getattr(node, 'name', 'unknown')}: {e}")
        return complexity

def add_parent_info(node: ast.AST, parent: ast.AST = None) -> None:
    """Add parent links to AST nodes."""
    for child in ast.iter_child_nodes(node):
        child.parent = node
        add_parent_info(child, node)