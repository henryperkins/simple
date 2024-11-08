import ast
from typing import Optional
from core.logger import LoggerSetup

# Initialize a logger specifically for this module
logger = LoggerSetup.get_logger("extract.utils")

def add_parent_info(tree: ast.AST) -> None:
    """
    Add parent information to AST nodes.

    Args:
        tree (ast.AST): The abstract syntax tree to process.
    """
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    logger.debug("Added parent information to AST nodes.")

def get_annotation(annotation: Optional[ast.AST]) -> str:
    """
    Get the string representation of an annotation.

    Args:
        annotation (Optional[ast.AST]): The annotation node.

    Returns:
        str: The string representation of the annotation.
    """
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
            logger.warning(f"Unhandled annotation type: {type(annotation)} with value: {ast.dump(annotation)}")
            return "Unknown"
    except Exception as e:
        logger.error(f"Error processing annotation: {e}")
        return "Unknown"
