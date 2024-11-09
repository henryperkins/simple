from typing import Dict, Any, List
import ast
from core.logger import LoggerSetup
from extract.base import BaseExtractor
from extract.utils import get_annotation
from metrics import CodeMetrics

logger = LoggerSetup.get_logger("extract.functions")

class FunctionExtractor(BaseExtractor):
    def __init__(self, node: ast.AST, content: str):
        super().__init__(node, content)
        self.metrics = CodeMetrics()

    def extract_details(self) -> Dict[str, Any]:
        try:
            # Calculate all metrics first
            complexity_score = self.calculate_complexity()
            cognitive_score = self.calculate_cognitive_complexity()
            halstead_metrics = self.calculate_halstead_metrics()

            details = {
                "name": self.node.name,
                "docstring": self.get_docstring(),
                "params": self.extract_parameters(),
                "returns": self._extract_return_annotation(),
                "complexity_score": complexity_score,
                "cognitive_complexity": cognitive_score,
                "halstead_metrics": halstead_metrics,
                "line_number": self.node.lineno,
                "end_line_number": self.node.end_lineno,
                "code": self.get_source_segment(self.node),
                "is_async": self.is_async(),
                "is_generator": self.is_generator(),
                "is_recursive": self.is_recursive(),
                "summary": self._generate_summary(complexity_score, cognitive_score, halstead_metrics),
                "changelog": ""  # Initialize as an empty string
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
                    "type": get_annotation(param.annotation),
                    "has_type_hint": param.annotation is not None
                }
                params.append(param_info)
        except Exception as e:
            logger.error(f"Error extracting parameters: {e}")
        return params

    def calculate_complexity(self) -> int:
        """Calculate cyclomatic complexity."""
        return self.metrics.calculate_complexity(self.node)

    def calculate_cognitive_complexity(self) -> int:
        """Calculate cognitive complexity."""
        return self.metrics.calculate_cognitive_complexity(self.node)

    def calculate_halstead_metrics(self) -> Dict[str, float]:
        """Calculate Halstead metrics."""
        return self.metrics.calculate_halstead_metrics(self.node)

    def _extract_return_annotation(self) -> Dict[str, Any]:
        """Extract return type annotation."""
        try:
            return {
                "type": get_annotation(self.node.returns),
                "has_type_hint": self.node.returns is not None
            }
        except Exception as e:
            logger.error(f"Error extracting return annotation: {e}")
            return {"type": "Any", "has_type_hint": False}

    def is_async(self) -> bool:
        """Check if the function is async."""
        return isinstance(self.node, ast.AsyncFunctionDef)

    def is_generator(self) -> bool:
        """Check if the function is a generator."""
        try:
            for node in ast.walk(self.node):
                if isinstance(node, ast.Yield) or isinstance(node, ast.YieldFrom):
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking generator status: {e}")
            return False

    def is_recursive(self) -> bool:
        """Check if the function is recursive."""
        try:
            function_name = self.node.name
            for node in ast.walk(self.node):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == function_name:
                        return True
            return False
        except Exception as e:
            logger.error(f"Error checking recursive status: {e}")
            return False

    def _generate_summary(self, complexity: int, cognitive: int, halstead: Dict[str, float]) -> str:
        """Generate a comprehensive summary of the function."""
        parts = []
        try:
            # Basic function characteristics
            if self.node.returns:
                parts.append(f"Returns: {get_annotation(self.node.returns)}")
            
            if self.is_generator():
                parts.append("Generator function")
            
            if self.is_async():
                parts.append("Async function")
            
            if self.is_recursive():
                parts.append("Recursive function")
            
            # Complexity metrics
            parts.append(f"Cyclomatic Complexity: {complexity}")
            parts.append(f"Cognitive Complexity: {cognitive}")
            
            # Halstead metrics summary
            if halstead.get("program_volume", 0) > 0:
                parts.append(f"Volume: {halstead['program_volume']:.2f}")
            if halstead.get("difficulty", 0) > 0:
                parts.append(f"Difficulty: {halstead['difficulty']:.2f}")
            
            # Quality assessment
            if complexity > 10:
                parts.append("⚠️ High cyclomatic complexity")
            if cognitive > 15:
                parts.append("⚠️ High cognitive complexity")
            if halstead.get("difficulty", 0) > 20:
                parts.append("⚠️ High difficulty score")

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            parts.append("Error generating complete summary")
        
        return " | ".join(parts)

    def _get_empty_details(self) -> Dict[str, Any]:
        """Return empty details structure matching schema."""
        return {
            "name": "",
            "docstring": "",
            "params": [],
            "returns": {"type": "Any", "has_type_hint": False},
            "complexity_score": 0,
            "cognitive_complexity": 0,
            "halstead_metrics": {
                "program_length": 0,
                "vocabulary_size": 0,
                "program_volume": 0,
                "difficulty": 0,
                "effort": 0
            },
            "line_number": 0,
            "end_line_number": 0,
            "code": "",
            "is_async": False,
            "is_generator": False,
            "is_recursive": False,
            "summary": "",
            "changelog": ""  # Initialize as an empty string
        }
