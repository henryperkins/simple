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
                    "default": self._get_default_value(param)
                }
                params.append(param_info)
        except Exception as e:
            logger.error(f"Error extracting parameters: {e}")
        return params

    def calculate_complexity(self) -> int:
        complexity = 1  # Base score
        # Add logic to calculate complexity
        return complexity

    def is_async(self) -> bool:
        return isinstance(self.node, ast.AsyncFunctionDef)

    def is_generator(self) -> bool:
        return any(isinstance(stmt, ast.Yield) for stmt in ast.walk(self.node))

    def is_recursive(self) -> bool:
        return any(isinstance(stmt, ast.Call) and stmt.func.id == self.node.name for stmt in ast.walk(self.node))

    def _get_empty_details(self) -> Dict[str, Any]:
        return {
            "name": "",
            "docstring": "",
            "params": [],
            "returns": "",
            "complexity_score": 0,
            "line_number": 0,
            "end_line_number": 0,
            "code": "",
            "is_async": False,
            "is_generator": False,
            "is_recursive": False,
            "summary": "",
            "changelog": []
        }

    def _extract_return_annotation(self) -> Dict[str, Any]:
        return {
            "type": self.get_annotation(self.node.returns),
            "has_type_hint": self.node.returns is not None
        }

    def _generate_summary(self) -> str:
        parts = []
        if self.node.returns:
            parts.append(f"Returns {get_annotation(self.node.returns)}")
        if self.is_generator():
            parts.append("Generator")
        if self.is_async():
            parts.append("Async")
        if self.is_recursive():
            parts.append("Recursive")
        complexity = self.calculate_complexity()
        parts.append(f"Complexity: {complexity}")
        return " | ".join(parts)