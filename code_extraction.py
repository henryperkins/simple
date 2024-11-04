# code_extraction.py

import ast
import logging
from typing import Dict, List, Any

class CodeExtractor:
    """A class to extract classes and functions from Python source code using AST."""

    def extract_classes_and_functions_from_ast(self, tree: ast.AST, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract class and function details from an Abstract Syntax Tree (AST).

        Args:
            tree (ast.AST): The AST of the Python file.
            content (str): The source code content of the file.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary containing lists of extracted classes
            and functions, each represented as a dictionary with relevant details.
        """
        classes = []
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(self.extract_class_details(node, content))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not self._is_method(node):
                functions.append(self.extract_function_details(node, content))

        return {"classes": classes, "functions": functions}

    def extract_class_details(self, node: ast.ClassDef, content: str) -> Dict[str, Any]:
        """Extract details from a class node in the AST.

        Args:
            node (ast.ClassDef): The class node in the AST.
            content (str): The source code content.

        Returns:
            Dict[str, Any]: A dictionary containing details about the class, including its
            name, docstring, source code, methods, nested classes, and decorators.
        """
        class_name = node.name
        class_docstring = ast.get_docstring(node) or ""
        class_code = ast.get_source_segment(content, node)

        methods = []
        nested_classes = []
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
            "nested_classes": nested_classes,
            "decorators": [self.extract_decorator_details(d) for d in node.decorator_list],
            "node": node,
        }

    def extract_function_details(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Extract details from a function or method node in the AST.

        Args:
            node (ast.FunctionDef or ast.AsyncFunctionDef): The function or method node in the AST.
            content (str): The source code content.

        Returns:
            Dict[str, Any]: A dictionary containing details about the function or method,
            including its name, parameters, return type, docstring, source code, complexity score,
            and decorators.
        """
        func_name = node.name
        params = [
            (
                arg.arg,
                ast.unparse(arg.annotation) if hasattr(arg, "annotation") and arg.annotation else "Unknown",
            )
            for arg in node.args.args
        ]
        return_type = ast.unparse(node.returns) if node.returns else "Unknown"
        docstring = ast.get_docstring(node) or ""
        function_code = ast.get_source_segment(content, node)
        complexity_score = self.calculate_complexity(node)

        return {
            "name": func_name,
            "params": params,
            "return_type": return_type,
            "docstring": docstring,
            "code": function_code,
            "complexity_score": complexity_score,
            "decorators": [self.extract_decorator_details(d) for d in node.decorator_list],
            "node": node,
        }

    def extract_decorator_details(self, node: ast.expr) -> str:
        """Extract details from a decorator node in the AST.

        Args:
            node (ast.expr): The decorator node in the AST.

        Returns:
            str: The decorator as a string representation.
        """
        return ast.unparse(node)

    def calculate_complexity(self, node: ast.AST) -> int:
        """Calculate the cyclomatic complexity score for a function or method.

        Cyclomatic complexity is a software metric used to measure the complexity of a program.
        It directly measures the number of linearly independent paths through a program's source code.

        Args:
            node (ast.AST): The AST node representing a function or method.

        Returns:
            int: The cyclomatic complexity score, which starts at 1 and increases with each
            decision point (e.g., if, for, while, try, etc.) in the function or method.
        """
        complexity = 1  # Start with one for the function entry point
        for subnode in ast.walk(node):
            if isinstance(subnode, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                complexity += 1
            elif isinstance(subnode, ast.BoolOp):
                # Each boolean operation (and/or) adds a branch
                complexity += len(subnode.values) - 1
        return complexity

    def _is_method(self, node: ast.FunctionDef) -> bool:
        """Determine if a function node is a method of a class.

        Args:
            node (ast.FunctionDef): The function node to check.

        Returns:
            bool: True if the function is a method, False otherwise.
        """
        return isinstance(getattr(node, 'parent', None), ast.ClassDef)

# Example usage
if __name__ == "__main__":
    source_code = """
class OuterClass:
    \"\"\"This is an outer class.\"\"\"
    
    class InnerClass:
        \"\"\"This is an inner class.\"\"\"
        
        def inner_method(self):
            pass
    
    def outer_method(self):
        pass

def standalone_function():
    pass
"""

    extractor = CodeExtractor()
    tree = ast.parse(source_code)
    result = extractor.extract_classes_and_functions_from_ast(tree, source_code)
    print(result)
