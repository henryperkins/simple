# functions.py
from typing import Any, Dict, List, Optional
from core.logger import LoggerSetup
from extract.base import BaseExtractor
from extract.utils import get_annotation

# Initialize logger for this module
logger = LoggerSetup.get_logger("extract.functions")

class FunctionExtractor(BaseExtractor):
    def extract_details(self) -> Dict[str, Any]:
        try:
            details = {
                "name": self.node.name,
                "docstring": self.get_docstring(),
                "params": self.extract_parameters(),
                "returns": self._extract_return_annotation(),
                "complexity_score": self.calculate_complexity(),
                "line_number": self.node.lineno,
                "end_line_number": self.node.end_lineno,
                "code": self.get_source_segment(self.node),
                "is_async": self.is_async(),
                "is_generator": self.is_generator(),
                "is_recursive": self.is_recursive(),
                "summary": self._generate_summary(),
                "changelog": []
            }
            return details
        except Exception as e:
            logger.error(f"Error extracting function details: {e}")
            return self._get_empty_details()

    def extract_parameters(self) -> List[Dict[str, Any]]:
        params = []
        try:
            for param in self.node.args.args:
                param_info = {
                    "name": param.arg,
                    "type": self.get_annotation(param.annotation),
                    "has_type_hint": param.annotation is not None
                }
                params.append(param_info)
        except Exception as e:
            logger.error(f"Error extracting parameters: {e}")
        return params

    def calculate_complexity(self) -> int:
        complexity = 1  # Base score
        try:
            for node in ast.walk(self.node):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                  ast.ExceptHandler, ast.With, ast.AsyncWith)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
        except Exception as e:
            logger.error(f"Error calculating complexity: {e}")
        return complexity

    def _generate_summary(self) -> str:
        parts = []
        try:
            if self.node.returns:
                parts.append(f"Returns: {self.get_annotation(self.node.returns)}")
            
            if self.is_generator():
                parts.append("Generator function")
            
            if self.is_async():
                parts.append("Async function")
            
            if self.is_recursive():
                parts.append("Recursive function")
            
            complexity = self.calculate_complexity()
            parts.append(f"Complexity: {complexity}")
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
        
        return " | ".join(parts)