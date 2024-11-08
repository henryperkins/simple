# extract/code.py
import ast
from typing import Dict, Any, List
from core.logger import LoggerSetup
from extract.base import BaseExtractor
from extract.utils import validate_schema
from extract.classes import ClassExtractor
from extract.functions import FunctionExtractor

logger = LoggerSetup.get_logger("extract.code")

def extract_classes_and_functions_from_ast(tree: ast.AST, content: str) -> Dict[str, Any]:
    """Extract all classes and functions from an AST."""
    try:
        result = {
            "summary": "",
            "changelog": [],
            "classes": [],
            "functions": [],
            "file_content": [{"content": content}]
        }

        # Extract classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                extractor = ClassExtractor(node, content)
                class_info = extractor.extract_details()
                result["classes"].append(class_info)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not isinstance(node.parent, ast.ClassDef):  # Only top-level functions
                    extractor = FunctionExtractor(node, content)
                    func_info = extractor.extract_details()
                    result["functions"].append(func_info)

        # Generate summary
        result["summary"] = f"Found {len(result['classes'])} classes and {len(result['functions'])} functions"
        
        # Validate against schema
        validate_schema(result)
        return result

    except Exception as e:
        logger.error(f"Error extracting classes and functions: {e}")
        return {
            "summary": "Error during extraction",
            "changelog": [],
            "classes": [],
            "functions": [],
            "file_content": [{"content": content}]
        }