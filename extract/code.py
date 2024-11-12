import ast
from typing import Dict, Any, List
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger("code")

def extract_classes_and_functions_from_ast(tree: ast.AST, content: str) -> Dict[str, Any]:
    """
    Extract classes and functions from the AST of a Python file.

    Args:
        tree (ast.AST): The abstract syntax tree of the Python file.
        content (str): The content of the Python file.

    Returns:
        Dict[str, Any]: A dictionary containing extracted classes and functions.
    """
    logger.debug("Starting extraction of classes and functions from AST")
    extracted_data = {
        "classes": [],
        "functions": []
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            logger.debug(f"Found class: {node.name}")
            base_classes = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_classes.append(base.id)
                elif isinstance(base, ast.Attribute):
                    base_classes.append(get_full_qualified_name(base))
            
            class_info = {
                "name": node.name,
                "base_classes": base_classes,
                "methods": [],
                "attributes": [],
                "instance_variables": [],
                "summary": "",
                "changelog": []
            }
            extracted_data["classes"].append(class_info)

        elif isinstance(node, ast.FunctionDef):
            logger.debug(f"Found function: {node.name}")
            function_info = {
                "name": node.name,
                "params": [{"name": arg.arg, "type": "Any"} for arg in node.args.args],
                "returns": {"type": "None", "description": ""},
                "complexity_score": 0,
                "line_number": node.lineno,
                "end_line_number": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                "code": ast.get_source_segment(content, node),
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "is_generator": any(isinstance(n, ast.Yield) for n in ast.walk(node)),
                "is_recursive": any(n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == node.name),
                "summary": "",
                "changelog": []
            }
            extracted_data["functions"].append(function_info)

    logger.debug("Extraction complete")
    return extracted_data

def get_full_qualified_name(node: ast.Attribute) -> str:
    """Helper function to get the full qualified name of an attribute."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return '.'.join(reversed(parts))
