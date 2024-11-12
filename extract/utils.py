# extract/utils.py

import ast
import json
import os
from typing import Optional, Dict, Any, Union, List
import jsonschema
from datetime import datetime
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger("extract.utils")

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

# extract/utils.py

def convert_changelog(changelog: Union[List, str, None]) -> str:
    """Convert changelog to string format."""
    if changelog is None:
        return "No changes recorded"
        
    if isinstance(changelog, str):
        return changelog if changelog.strip() else "No changes recorded"
        
    if isinstance(changelog, list):
        if not changelog:
            return "No changes recorded"
            
        entries = []
        for entry in changelog:
            if isinstance(entry, dict):
                timestamp = entry.get("timestamp", datetime.now().isoformat())
                change = entry.get("change", "No description")
                entries.append(f"[{timestamp}] {change}")
            else:
                entries.append(str(entry))
        return " | ".join(entries)
        
    return "No changes recorded"

def format_function_response(function_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format function data ensuring proper changelog format."""
    result = function_data.copy()
    
    # Ensure changelog is string
    result["changelog"] = convert_changelog(result.get("changelog"))
    
    # Add other required fields with defaults
    result.setdefault("summary", "No summary available")
    result.setdefault("docstring", "")
    result.setdefault("params", [])
    result.setdefault("returns", {"type": "None", "description": ""})
    
    return result

def validate_function_data(data: Dict[str, Any]) -> None:
    """Validate function data before processing."""
    try:
        # Convert changelog before validation
        if "changelog" in data:
            data["changelog"] = convert_changelog(data["changelog"])
            
        schema = _load_schema()
        jsonschema.validate(instance=data, schema=schema)
        
    except jsonschema.ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise
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
            if isinstance(node.op, ast.BitOr):
                left = get_annotation(node.left)
                right = get_annotation(node.right)
                return f"Union[{left}, {right}]"
        else:
            return "Any"
    except Exception as e:
        logger.error(f"Error processing type annotation: {e}")
        return "Any"

def format_response(sections: Dict[str, Any]) -> Dict[str, Any]:
    """Format parsed sections into a standardized response with validation."""
    
    # Initialize with required fields
    result = {
        "summary": sections.get("summary", "No summary available"),
        "docstring": sections.get("docstring", "No documentation available"),
        "params": sections.get("params", []),
        "returns": sections.get("returns", {
            "type": "None",
            "description": ""
        }),
        "examples": sections.get("examples", []),
        "changelog": convert_changelog(sections.get("changelog", []))
    }
    
    return result

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
