import ast
import json
import os
from typing import Optional, Dict, Any
import jsonschema
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger("extract.utils")

# Cache for the schema to avoid repeated file reads
_schema_cache: Dict[str, Any] = {}

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
    """
    Convert AST annotation to string representation.
    
    Args:
        node (Optional[ast.AST]): The AST node containing type annotation
        
    Returns:
        str: String representation of the type annotation
    """
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
        elif isinstance(node, ast.BinOp):
            # Handle Union types written with | operator (Python 3.10+)
            if isinstance(node.op, ast.BitOr):
                left = get_annotation(node.left)
                right = get_annotation(node.right)
                return f"Union[{left}, {right}]"
        else:
            return "Any"
    except Exception as e:
        logger.error(f"Error processing type annotation: {e}")
        return "Any"

def _load_schema() -> Dict[str, Any]:
    """
    Load the JSON schema from file with caching.
    
    Returns:
        Dict[str, Any]: The loaded schema
        
    Raises:
        FileNotFoundError: If schema file is not found
        json.JSONDecodeError: If schema file is invalid JSON
    """
    if 'schema' not in _schema_cache:
        schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'function_schema.json')
        try:
            with open(schema_path, 'r', encoding='utf-8') as schema_file:
                _schema_cache['schema'] = json.load(schema_file)
                logger.debug("Loaded schema from file")
        except FileNotFoundError:
            logger.error(f"Schema file not found at {schema_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in schema file: {e}")
            raise
    return _schema_cache['schema']

def validate_schema(data: Dict[str, Any]) -> None:
    """
    Validate extracted data against schema.
    
    Args:
        data (Dict[str, Any]): The data to validate
        
    Raises:
        jsonschema.ValidationError: If validation fails
        jsonschema.SchemaError: If schema is invalid
        FileNotFoundError: If schema file is not found
        json.JSONDecodeError: If schema file is invalid JSON
    """
    try:
        schema = _load_schema()
        jsonschema.validate(instance=data, schema=schema)
        logger.debug("Schema validation successful")
    except jsonschema.ValidationError as e:
        logger.error(f"Schema validation failed: {e.message}")
        logger.error(f"Failed at path: {' -> '.join(str(p) for p in e.path)}")
        logger.error(f"Instance: {e.instance}")
        raise
    except jsonschema.SchemaError as e:
        logger.error(f"Invalid schema: {e.message}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during schema validation: {e}")
        raise

def format_validation_error(error: jsonschema.ValidationError) -> str:
    """
    Format a validation error into a human-readable message.
    
    Args:
        error (jsonschema.ValidationError): The validation error
        
    Returns:
        str: Formatted error message
    """
    path = ' -> '.join(str(p) for p in error.path) if error.path else 'root'
    return (
        f"Validation error at {path}:\n"
        f"Message: {error.message}\n"
        f"Failed value: {error.instance}\n"
        f"Schema path: {' -> '.join(str(p) for p in error.schema_path)}"
    )