import ast
from typing import Optional
from logging_utils import setup_logger

# Initialize a logger for this module
logger = setup_logger("utils")

def add_parent_info(node: Optional[ast.AST], parent: Optional[ast.AST] = None) -> None:
    """Add parent links to AST nodes."""
    if node is None:
        logger.warning("No node provided to add_parent_info.")
        return
    for child in ast.iter_child_nodes(node):
        setattr(child, 'parent', node)
        logger.debug(f"Set parent of node {getattr(child, 'name', 'unknown')} to {getattr(node, 'name', 'unknown')}")
        add_parent_info(child, node)
    logger.debug(f"Added parent info to node: {getattr(node, 'name', 'unknown')}")

def get_annotation(annotation: Optional[ast.AST]) -> str:
    """Convert AST annotation to string."""
    if annotation is None:
        logger.debug("No annotation found; returning 'None'")
        return "None"
    elif isinstance(annotation, ast.Str):
        logger.debug(f"Annotation is a string: {annotation.s}")
        return annotation.s
    elif isinstance(annotation, ast.Name):
        logger.debug(f"Annotation is a name: {annotation.id}")
        return annotation.id
    elif isinstance(annotation, ast.Subscript):
        value = get_annotation(annotation.value)
        slice_ = get_annotation(annotation.slice)
        logger.debug(f"Annotation is a subscript: {value}[{slice_}]")
        return f"{value}[{slice_}]"
    elif isinstance(annotation, ast.Attribute):
        value = get_annotation(annotation.value)
        logger.debug(f"Annotation is an attribute: {value}.{annotation.attr}")
        return f"{value}.{annotation.attr}"
    else:
        logger.warning("Unknown annotation type encountered.")
        return "Unknown"