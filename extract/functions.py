import ast
from typing import Any, Dict, List
from .utils import get_annotation
from logging_utils import setup_logger

# Initialize a logger for this module
logger = setup_logger("extract.functions")

class FunctionExtractor:
    """
    Handles extraction of function details from AST nodes.

    Args:
        node (ast.FunctionDef): The function definition node.
        content (str): The source code content.
    """

    def __init__(self, node: ast.FunctionDef, content: str):
        self.node = node
        self.content = content
        logger.debug(f"Initialized FunctionExtractor for function: {node.name}")

    def extract_details(self) -> Dict[str, Any]:
        """
        Extract comprehensive information from a function node.

        Returns:
            Dict[str, Any]: The extracted function details.
        """
        try:
            details = {
                "name": self.node.name,
                "docstring": ast.get_docstring(self.node) or "",
                "params": self.extract_parameters(),
                "complexity_score": self.calculate_complexity(),
                "line_number": self.node.lineno,
                "end_line_number": getattr(self.node, 'end_lineno', self.node.lineno),
                "code": self.extract_function_code(),
                "is_async": isinstance(self.node, ast.AsyncFunctionDef),
                "is_generator": self.is_generator(),
                "is_recursive": self.is_recursive(),
            }
            logger.debug(f"Extracted details for function {self.node.name}: {details}")
            return details
        except Exception as e:
            logger.error(f"Error extracting details for function {self.node.name}: {e}")
            return {}

    def extract_parameters(self) -> List[Dict[str, Any]]:
        """
        Extract parameters from the function.

        Returns:
            List[Dict[str, Any]]: A list of parameter details.
        """
        params = []
        try:
            for arg in self.node.args.args:
                param_info = {
                    "name": arg.arg,
                    "type": get_annotation(arg.annotation),
                    "has_type_hint": arg.annotation is not None,
                }
                params.append(param_info)
                logger.debug(f"Extracted parameter: {param_info}")
            return params
        except Exception as e:
            logger.error(f"Error extracting parameters for function {self.node.name}: {e}")
            return params

    def calculate_complexity(self) -> int:
        """
        Calculate the function's complexity.

        Returns:
            int: The complexity score.
        """
        complexity = 1  # Start with one for the function entry point
        try:
            for subnode in ast.walk(self.node):
                if isinstance(subnode, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp)):
                    complexity += 1
            logger.debug(f"Calculated complexity for function {self.node.name}: {complexity}")
            return complexity
        except Exception as e:
            logger.error(f"Error calculating complexity for function {self.node.name}: {e}")
            return complexity

    def extract_function_code(self) -> str:
        """
        Extract the function's source code.

        Returns:
            str: The source code of the function.
        """
        try:
            if not hasattr(self.node, 'lineno') or not hasattr(self.node, 'end_lineno'):
                return ""
            
            lines = self.content.splitlines()
            start_line = self.node.lineno - 1
            end_line = getattr(self.node, 'end_lineno', start_line + 1)
            
            code = "\n".join(lines[start_line:end_line])
            logger.debug(f"Extracted code for function {self.node.name}")
            return code
        except Exception as e:
            logger.error(f"Error extracting code for function {self.node.name}: {e}")
            return ""

    def is_generator(self) -> bool:
        """
        Check if the function is a generator.

        Returns:
            bool: True if the function is a generator, False otherwise.
        """
        try:
            for node in ast.walk(self.node):
                if isinstance(node, (ast.Yield, ast.YieldFrom)):
                    logger.debug(f"Function {self.node.name} is a generator")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking if function {self.node.name} is a generator: {e}")
            return False

    def is_recursive(self) -> bool:
        """
        Check if the function calls itself.

        Returns:
            bool: True if the function is recursive, False otherwise.
        """
        try:
            for node in ast.walk(self.node):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == self.node.name:
                        logger.debug(f"Function {self.node.name} is recursive")
                        return True
            return False
        except Exception as e:
            logger.error(f"Error checking if function {self.node.name} is recursive: {e}")
            return False