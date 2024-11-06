import ast
from typing import Any, Dict, List
from .utils import get_annotation
from logging_utils import setup_logger

# Initialize a logger for this module
logger = setup_logger("functions")

class FunctionExtractor:
    """Handles extraction of function details from AST nodes."""
    def __init__(self, node: ast.FunctionDef, content: str):
        self.node = node
        self.content = content
        logger.debug(f"Initialized FunctionExtractor for function: {node.name}")

    def extract_details(self) -> Dict[str, Any]:
        """Extract comprehensive information from a function node."""
        details = {
            "name": self.node.name,
            "docstring": ast.get_docstring(self.node) or "",
            "params": self.extract_parameters(),
            "complexity_score": self.calculate_complexity(),
            "line_number": self.node.lineno,
            "end_line_number": self.node.end_lineno or self.node.lineno,
            "code": self.extract_function_code(),
            "is_async": isinstance(self.node, ast.AsyncFunctionDef),
            "is_generator": self.is_generator(),
            "is_recursive": self.is_recursive(),
        }
        logger.debug(f"Extracted details for function {self.node.name}: {details}")
        return details

    def extract_parameters(self) -> List[Dict[str, str]]:
        """Extract parameters from the function."""
        params = []
        for arg in self.node.args.args:
            param_info = {
                "name": arg.arg,
                "type": get_annotation(arg.annotation),
                "has_type_hint": arg.annotation is not None,
            }
            params.append(param_info)
            logger.debug(f"Extracted parameter: {param_info}")
        return params

    def calculate_complexity(self) -> int:
        """Calculate the function's complexity."""
        complexity = 1  # Start with one for the function entry point
        for subnode in ast.walk(self.node):
            if isinstance(subnode, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                complexity += 1
        logger.debug(f"Calculated complexity for function {self.node.name}: {complexity}")
        return complexity

    def extract_function_code(self) -> str:
        """Extract the function's source code."""
        if not hasattr(self.node, 'lineno') or not hasattr(self.node, 'end_lineno'):
            return ""
        
        lines = self.content.splitlines()
        start_line = self.node.lineno - 1
        end_line = self.node.end_lineno or start_line + 1
        
        code = "\n".join(lines[start_line:end_line])
        logger.debug(f"Extracted code for function {self.node.name}: {code}")
        return code

    def is_generator(self) -> bool:
        """Check if the function is a generator."""
        for node in ast.walk(self.node):
            if isinstance(node, ast.Yield) or isinstance(node, ast.YieldFrom):
                logger.debug(f"Function {self.node.name} is a generator")
                return True
        return False

    def is_recursive(self) -> bool:
        """Check if the function calls itself."""
        for node in ast.walk(self.node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == self.node.name:
                    logger.debug(f"Function {self.node.name} is recursive")
                    return True
        return False