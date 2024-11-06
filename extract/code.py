import ast
from typing import Any, Dict
from .functions import FunctionExtractor
from .classes import ClassExtractor
from .utils import add_parent_info
from logging_utils import setup_logger

# Initialize a logger for this module
logger = setup_logger("extract.code")

def extract_classes_and_functions_from_ast(tree: ast.AST, content: str) -> Dict[str, Any]:
    """
    Extract detailed class and function information from an AST.

    Args:
        tree (ast.AST): The abstract syntax tree of the code.
        content (str): The source code content.

    Returns:
        Dict[str, Any]: A dictionary containing extracted classes and functions.
    """
    logger.debug("Starting extraction of classes and functions from AST.")
    classes = []
    functions = []

    try:
        # Add parent information to AST nodes for better structure
        add_parent_info(tree)

        # Traverse AST nodes to extract class and function information
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_extractor = ClassExtractor(node, content)
                class_info = class_extractor.extract_details()
                if class_info:
                    classes.append(class_info)
                    logger.debug(f"Extracted class: {class_info['name']}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only extract if the function is not a method
                if not is_method(node):
                    function_extractor = FunctionExtractor(node, content)
                    func_info = function_extractor.extract_details()
                    if func_info:
                        functions.append(func_info)
                        logger.debug(f"Extracted function: {func_info['name']}")
        logger.debug("Completed extraction from AST.")
    except Exception as e:
        logger.error(f"Error during extraction from AST: {e}")
        return {}

    return {
        "classes": classes,
        "functions": functions,
        "file_content": [{"content": content}]
    }

def is_method(node: ast.FunctionDef) -> bool:
    """
    Determine if a function node is a method within a class.

    Args:
        node (ast.FunctionDef): The function definition node.

    Returns:
        bool: True if the function is a method, False otherwise.
    """
    return isinstance(node.parent, ast.ClassDef)