# extract/utils.py
import ast
from typing import Optional
import jsonschema
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger("extract.utils")
def add_parent_info(tree: ast.AST) -> None:
    """
    Add parent information to each node in the AST.
    
    This function traverses the AST and adds a 'parent' attribute to each node,
    which is needed for correctly identifying top-level functions vs methods.
    
    Args:
        tree (ast.AST): The AST to process
    """
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent
            
def get_annotation(node: Optional[ast.AST]) -> str:
    """Convert AST annotation to string representation."""
    try:
        if node is None:
            return "Any"
            
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Attribute):
            parts = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        elif isinstance(node, ast.Subscript):
            return f"{get_annotation(node.value)}[{get_annotation(node.slice)}]"
        else:
            return "Any"
    except Exception as e:
        logger.error(f"Error processing type annotation: {e}")
        return "Any"

def validate_schema(data: dict) -> None:
    """Validate extracted data against schema."""
    try:
        with open('function_schema.json') as schema_file:
            schema = json.load(schema_file)
        jsonschema.validate(instance=data, schema=schema)
        logger.debug("Schema validation successful")
    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        raise