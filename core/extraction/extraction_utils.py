"""
Utility functions for code extraction.
"""

import ast
from typing import List, Dict, Any, Optional
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from utils import handle_extraction_error, get_node_name

logger = LoggerSetup.get_logger(__name__)


def extract_decorators(node: ast.AST) -> List[str]:
    """Extract decorator names from a node (class or function).

    Args:
        node (ast.AST): The AST node to extract decorators from.

    Returns:
        List[str]: A list of decorator names.
    """
    decorators = []
    for decorator in getattr(node, "decorator_list", []):
        if isinstance(decorator, ast.Name):
            decorators.append(decorator.id)
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                decorators.append(decorator.func.id)
            elif isinstance(decorator.func, ast.Attribute):
                if hasattr(decorator.func.value, "id"):
                    decorators.append(
                        f"{decorator.func.value.id}.{decorator.func.attr}"
                    )
                else:
                    decorators.append(
                        decorator.func.attr
                    )  # Fallback if value.id doesn't exist
        elif isinstance(decorator, ast.Attribute):
            if hasattr(decorator.value, "id"):
                decorators.append(f"{decorator.value.id}.{decorator.attr}")
            else:
                decorators.append(decorator.attr)  # Fallback if value.id doesn't exist
    return decorators


def extract_attributes(node: ast.ClassDef, source_code: str, instance: bool = False) -> List[Dict[str, Any]]:
    """Extract class-level or instance-level attributes.

    Args:
        node (ast.ClassDef): The class node to extract attributes from.
        source_code (str): The source code of the module.
        instance (bool): Whether to extract instance attributes (default: False).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing attribute information.
    """
    attributes = []
    for child in ast.walk(node) if instance else node.body:
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            attributes.append(
                {
                    "name": child.target.id,
                    "type": get_node_name(child.annotation),
                    "value": ast.unparse(child.value) if child.value else None,
                    "lineno": child.lineno,
                }
            )
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    attributes.append(
                        {
                            "name": target.id,
                            "type": "Any",  # Infer type if possible in the future
                            "value": ast.unparse(child.value),
                            "lineno": child.lineno,
                        }
                    )
                elif (
                    instance
                    and isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    attributes.append(
                        {
                            "name": target.attr,
                            "type": "Any",
                            "value": ast.unparse(child.value),
                            "lineno": child.lineno,
                        }
                    )
    return attributes


def extract_bases(node: ast.ClassDef) -> List[str]:
    """Extract base class names.

    Args:
        node (ast.ClassDef): The class node to extract base classes from.

    Returns:
        List[str]: A list of base class names.
    """
    bases = []
    for base in node.bases:
        bases.append(get_node_name(base))
    return bases
