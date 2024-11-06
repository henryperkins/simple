import ast
from typing import Optional, Any
from core.logging.setup import LoggerSetup

# Initialize a logger for this module
logger = LoggerSetup.get_logger("extract.utils")

def add_parent_info(node: Optional[ast.AST], parent: Optional[ast.AST] = None) -> None:
    """
    Add parent links to AST nodes recursively.

    This function traverses the AST nodes and sets a 'parent' attribute on each node,
    which refers to its immediate parent node. This is useful for backtracking and
    context-aware analyses when processing the AST.

    Args:
        node (Optional[ast.AST]): The current AST node to process.
        parent (Optional[ast.AST]): The parent of the current node.
    """
    if node is None:
        logger.warning("No node provided to add_parent_info.")
        return

    try:
        for child in ast.iter_child_nodes(node):
            setattr(child, 'parent', node)
            child_name = getattr(child, 'name', getattr(child, 'id', 'unknown'))
            parent_name = getattr(node, 'name', getattr(node, 'id', 'unknown'))
            logger.debug(f"Set parent of node '{child_name}' to '{parent_name}'")
            add_parent_info(child, node)
        node_name = getattr(node, 'name', getattr(node, 'id', 'unknown'))
        logger.debug(f"Added parent info to node: '{node_name}'")
    except Exception as e:
        node_repr = getattr(node, 'name', str(node))
        logger.error(f"Error adding parent info to node '{node_repr}': {e}")

def get_annotation(annotation: Optional[ast.AST]) -> str:
    """
    Convert an AST annotation to a string representation.

    This function handles various types of annotations in AST nodes and converts them
    into their string representations, which can include complex types like subscripts
    and attributes.

    Args:
        annotation (Optional[ast.AST]): The annotation node to process.

    Returns:
        str: The string representation of the annotation.

    Raises:
        ValueError: If an unknown annotation type is encountered.
    """
    if annotation is None:
        logger.debug("No annotation found; returning 'None'")
        return "None"

    try:
        if isinstance(annotation, ast.Str):
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
        elif isinstance(annotation, ast.Tuple):
            elements = [get_annotation(elt) for elt in annotation.elts]
            logger.debug(f"Annotation is a tuple: ({', '.join(elements)})")
            return f"({', '.join(elements)})"
        elif isinstance(annotation, ast.Constant):
            logger.debug(f"Annotation is a constant: {annotation.value}")
            return str(annotation.value)
        else:
            logger.warning(f"Unknown annotation type encountered: {type(annotation)}")
            return "Unknown"
    except Exception as e:
        logger.error(f"Error processing annotation: {e}")
        return "Unknown"