import ast
import base64
import io
import math
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from core.console import (
    create_progress,
    display_metrics,
    print_debug,
    print_error,
    print_info,
    print_warning,
)
from core.metrics_collector import MetricsCollector

if TYPE_CHECKING:
    from core.metrics_collector import MetricsCollector

# Try to import matplotlib, but provide fallback if not available
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class Metrics:
    """Calculates various code complexity metrics for Python code."""

    def __init__(
        self,
        metrics_collector: Optional["MetricsCollector"] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Initialize the Metrics class."""
        self.module_name: Optional[str] = None
        self.logger = self._get_logger()
        self.error_counts: Dict[str, int] = {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.metrics_collector = metrics_collector or MetricsCollector(
            correlation_id=self.correlation_id
        )

        # Ensure metrics calculator is registered with Injector
        self._register_with_injector()

    def _get_logger(self) -> "CorrelationLoggerAdapter":
        """Get a logger instance."""
        from core.logger import CorrelationLoggerAdapter, LoggerSetup

        return CorrelationLoggerAdapter(LoggerSetup.get_logger(__name__))

    def _register_with_injector(self) -> None:
        """Registers the metrics calculator with the Injector."""
        from core.dependency_injection import Injector

        if "metrics_calculator" not in Injector._dependencies:
            Injector.register("metrics_calculator", self)

    def calculate_metrics(self, code: str, module_name: Optional[str] = None) -> Any:
        """Calculate all metrics for the given code.

        Args:
            code: The source code to analyze
            module_name: Optional name of the module being analyzed

        Returns:
            MetricData containing all calculated metrics
        """
        try:
            from core.types import MetricData

            self.module_name = module_name
            # Parse code once and reuse the AST
            tree = ast.parse(code)

            # Calculate base metrics first
            lines_of_code = len(code.splitlines())
            cyclomatic = self._calculate_cyclomatic_complexity(tree)
            cognitive = self._calculate_cognitive_complexity(tree)

            # Calculate Halstead metrics without recursion
            halstead = self._calculate_halstead_metrics(code)

            # Calculate maintainability using pre-calculated values
            maintainability = self._calculate_maintainability_direct(
                lines_of_code, cyclomatic, halstead.get("volume", 0)
            )

            metrics = MetricData()
            metrics.cyclomatic_complexity = cyclomatic
            metrics.cognitive_complexity = cognitive
            metrics.maintainability_index = maintainability
            metrics.halstead_metrics = halstead
            metrics.lines_of_code = lines_of_code

            # Count total functions and classes
            total_functions = sum(
                1
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            total_classes = sum(
                1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            )

            metrics.total_functions = total_functions
            metrics.total_classes = total_classes

            # Note: scanned_functions and scanned_classes will be set by the extractors
            # Default to 0 here as they'll be updated during extraction
            metrics.scanned_functions = 0
            metrics.scanned_classes = 0

            if MATPLOTLIB_AVAILABLE:
                metrics.complexity_graph = self._generate_complexity_graph()
            else:
                metrics.complexity_graph = None

            # Log metrics collection
            self.metrics_collector.collect_metrics(module_name or "unknown", metrics)

            return metrics

        except Exception as e:
            self.logger.error(
                f"Error calculating metrics: {e} with correlation ID: {self.correlation_id}",
                exc_info=True,
            )
            # Return default metrics on error
            from core.types import MetricData

            return MetricData()

    def calculate_maintainability_index(self, code: str) -> float:
        """Calculate maintainability index for the given code.

        Args:
            code: The source code to analyze

        Returns:
            float: The maintainability index score (0-100)
        """
        try:
            tree = ast.parse(code)
            lines_of_code = len(code.splitlines())
            cyclomatic = self._calculate_cyclomatic_complexity(tree)
            halstead_metrics = self._calculate_halstead_metrics(code)
            maintainability = self._calculate_maintainability_direct(
                lines_of_code,
                cyclomatic,
                halstead_metrics.get("volume", 0),
            )
            return maintainability
        except Exception as e:
            self.logger.error(
                f"Error calculating maintainability index: {e} with correlation ID: {self.correlation_id}",
                exc_info=True,
            )
            return 0.0

    def calculate_metrics_for_class(self, class_data: Any) -> Any:
        """Calculate metrics for a class.

        Args:
            class_data: The class data to analyze

        Returns:
            MetricData containing the calculated metrics
        """
        try:
            from core.types import MetricData

            source_code = class_data.source
            if not source_code:
                return MetricData()

            metrics = self.calculate_metrics(source_code)
            # Mark this as a successfully scanned class
            metrics.scanned_classes = 1
            metrics.total_classes = 1
            return metrics

        except Exception as e:
            self.logger.error(
                f"Error calculating class metrics: {e} with correlation ID: {self.correlation_id}",
                exc_info=True,
            )
            from core.types import MetricData

            return MetricData()

    def calculate_metrics_for_function(self, function_data: Any) -> Any:
        """Calculate metrics for a function.

        Args:
            function_data: The function data to analyze

        Returns:
            MetricData containing the calculated metrics
        """
        try:
            from core.types import MetricData

            source_code = function_data.source
            if not source_code:
                return MetricData()

            metrics = self.calculate_metrics(source_code)
            # Mark this as a successfully scanned function
            metrics.scanned_functions = 1
            metrics.total_functions = 1
            return metrics

        except Exception as e:
            self.logger.error(
                f"Error calculating function metrics: {e} with correlation ID: {self.correlation_id}",
                exc_info=True,
            )
            from core.types import MetricData

            return MetricData()

    def _calculate_cyclomatic_complexity(self, tree: Union[ast.AST, ast.Module]) -> int:
        """Calculate cyclomatic complexity."""
        try:
            complexity = 1  # Base complexity

            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (
                        ast.If,
                        ast.While,
                        ast.For,
                        ast.Assert,
                        ast.Try,
                        ast.ExceptHandler,
                    ),
                ):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1

            return complexity
        except Exception as e:
            self.logger.error(
                f"Error calculating cyclomatic complexity: {e} with correlation ID: {self.correlation_id}",
                exc_info=True,
            )
            return 1

    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity."""
        try:
            complexity = 0
            nesting_level = 0

            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For)):
                    complexity += 1 + nesting_level
                    nesting_level += 1
                elif isinstance(node, ast.Try):
                    complexity += nesting_level

            return complexity
        except Exception as e:
            self.logger.error(
                f"Error calculating cognitive complexity: {e} with correlation ID: {self.correlation_id}",
                exc_info=True,
            )
            return 0

    def _calculate_maintainability_direct(
        self, loc: int, cyclomatic: int, volume: float
    ) -> float:
        """Calculate maintainability index using pre-calculated metrics."""
        try:
            # Ensure non-zero values
            loc = max(1, loc)
            volume = max(1, volume)
            cyclomatic = max(1, cyclomatic)

            # Use log1p to handle small values safely
            mi = (
                171
                - 5.2 * math.log1p(volume)
                - 0.23 * cyclomatic
                - 16.2 * math.log1p(loc)
            )
            return max(0.0, min(100.0, mi))

        except Exception as e:
            self.logger.error(
                f"Error calculating maintainability index: {e} with correlation ID: {self.correlation_id}",
                exc_info=True,
            )
            return 50.0  # Return a neutral value on error

    def _calculate_halstead_metrics(self, code: str) -> Dict[str, float]:
        """Calculate Halstead metrics."""
        try:
            operators = set()
            operands = set()

            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.operator):
                    operators.add(node.__class__.__name__)
                elif isinstance(node, ast.Name):
                    operands.add(node.id)

            n1 = max(1, len(operators))  # Ensure non-zero values
            n2 = max(1, len(operands))
            N1 = max(
                1,
                sum(1 for node in ast.walk(tree) if isinstance(node, ast.operator)),
            )
            N2 = max(
                1,
                sum(1 for node in ast.walk(tree) if isinstance(node, ast.Name)),
            )

            # Use log1p for safe logarithm calculation
            volume = (N1 + N2) * math.log1p(n1 + n2)
            difficulty = (n1 / 2) * (N2 / n2)
            effort = difficulty * volume

            return {
                "volume": max(0.0, volume),
                "difficulty": max(0.0, difficulty),
                "effort": max(0.0, effort),
                "time": max(0.0, effort / 18),
                "bugs": max(0.0, volume / 3000),
            }

        except Exception as e:
            self.logger.error(
                f"Error calculating Halstead metrics: {e} with correlation ID: {self.correlation_id}",
                exc_info=True,
            )
            return {
                "volume": 0.0,
                "difficulty": 0.0,
                "effort": 0.0,
                "time": 0.0,
                "bugs": 0.0,
            }

    def _calculate_halstead_volume(self, code: str) -> float:
        """Calculate Halstead volume metric."""
        try:
            metrics = self._calculate_halstead_metrics(code)
            return max(0.0, metrics["volume"])
        except Exception as e:
            self.logger.error(
                f"Error calculating Halstead volume: {e} with correlation ID: {self.correlation_id}",
                exc_info=True,
            )
            return 0.0

    def _generate_complexity_graph(self) -> Optional[str]:
        """Generate a base64 encoded PNG of the complexity metrics graph."""
        # Skip graph generation by default to avoid threading issues
        return None

        # Note: The graph generation code is disabled to prevent Tcl threading errors.
        # To re-enable, implement with proper thread safety measures and
        # use the Agg backend explicitly if needed.
