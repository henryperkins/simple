import ast
from typing import Any, Dict
from .functions import FunctionExtractor
from .classes import ClassExtractor
from .utils import add_parent_info
from logging_utils import setup_logger  # Import the setup_logger utility

# Initialize a logger for this module
logger = setup_logger("code")

def extract_classes_and_functions_from_ast(tree: ast.AST, content: str) -> Dict[str, Any]:
    """Extract detailed class and function information from an AST."""
    logger.debug("Starting extraction of classes and functions from AST.")
    classes = []
    functions = []

    # Add parent information to AST nodes for better structure
    add_parent_info(tree)

    # Traverse AST nodes to extract class and function information
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_extractor = ClassExtractor(node, content)
            class_info = class_extractor.extract_details()
            logger.debug(f"Extracted class: {class_info['name']}")
            classes.append(class_info)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Only extract if the function is not a method
            if not FunctionExtractor.is_method(node):
                function_extractor = FunctionExtractor(node, content)
                func_info = function_extractor.extract_details()
                logger.debug(f"Extracted function: {func_info['name']}")
                functions.append(func_info)

    logger.debug("Completed extraction from AST.")
    return {
        "classes": classes,
        "functions": functions,
        "file_content": [{"content": content}]
    }