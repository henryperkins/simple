import ast
from typing import Dict, Any, List
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger("classes")

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
