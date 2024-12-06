"""
Metrics module for calculating code complexity and performance metrics.

Provides comprehensive code analysis metrics including cyclomatic complexity,
cognitive complexity, and Halstead metrics.
"""

import ast
import math
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.logger import LoggerSetup, log_error, log_debug, log_info

logger = LoggerSetup.get_logger(__name__)


class MetricsError(Exception):
    """Base exception for metrics calculation errors."""


class Metrics:
    """Calculates various code complexity metrics for Python code."""

    MAINTAINABILITY_THRESHOLDS: Dict[str, int] = {
        "good": 80,
        "moderate": 60,
        "poor": 40,
    }

    def __init__(self) -> None:
        """Initialize the Metrics class."""
        self.module_name: Optional[str] = None
        self.logger = LoggerSetup.get_logger(__name__)
        self.error_counts: Dict[str, int] = {}

    def calculate_function_metrics(self, node: ast.AST) -> Dict[str, Any]:
        """
        Calculate metrics for a function.

        Args:
            node: The AST node representing a function

        Returns:
            Dict containing function metrics
        """
        logger.debug(
            "Calculating metrics for function: %s",
            getattr(node, 'name', 'unknown')
        )
        complexity = self.calculate_cyclomatic_complexity(node)
        metrics = {
            "cyclomatic_complexity": complexity,
            "cognitive_complexity": self.calculate_cognitive_complexity(node),
            "halstead_metrics": self.calculate_halstead_metrics(node),
            "maintainability_index": self.calculate_maintainability_index(node),
        }
        # Add complexity warning if necessary
        if complexity > 10:
            metrics["complexity_warning"] = "⚠️ High complexity"
        return metrics

    def calculate_class_metrics(self, node: ast.AST) -> Dict[str, Any]:
        """
        Calculate metrics for a class.

        Args:
            node: The AST node representing a class

        Returns:
            Dict containing class metrics
        """
        logger.debug(
            "Calculating metrics for class: %s",
            getattr(node, 'name', 'unknown')
        )
        complexity = self.calculate_cyclomatic_complexity(node)
        metrics = {
            "method_count": self.count_methods(node),
            "cyclomatic_complexity": complexity,
            "cognitive_complexity": self.calculate_cognitive_complexity(node),
            "halstead_metrics": self.calculate_halstead_metrics(node),
            "maintainability_index": self.calculate_maintainability_index(node),
        }
        # Add complexity warning if necessary
        if complexity > 10:
            metrics["complexity_warning"] = "⚠️ High complexity"
        return metrics

    def calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """
        Calculate cyclomatic complexity for a function or class.

        Args:
            node: The AST node to analyze

        Returns:
            The cyclomatic complexity score
        """
        complexity = 1
        for child in ast.walk(node):
            if isinstance(
                child,
                (
                    ast.If,
                    ast.For,
                    ast.While,
                    ast.AsyncFor,
                    ast.Try,
                    ast.ExceptHandler,
                    ast.With,
                    ast.AsyncWith,
                    ast.BoolOp,
                ),
            ):
                complexity += 1
        logger.debug("Cyclomatic complexity: %d", complexity)
        return complexity

    def calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """
        Calculate cognitive complexity for a function or class.

        Args:
            node: The AST node to analyze

        Returns:
            The cognitive complexity score
        """
        complexity = 0

        def _increment_complexity(node: ast.AST, nesting: int) -> None:
            nonlocal complexity
            if isinstance(
                node, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler, ast.With)
            ):
                complexity += 1
                nesting += 1
            elif isinstance(
                node,
                (
                    ast.BoolOp,
                    ast.Break,
                    ast.Continue,
                    ast.Raise,
                    ast.Return,
                    ast.Yield,
                    ast.YieldFrom,
                ),
            ):
                complexity += nesting + 1

            for child in ast.iter_child_nodes(node):
                _increment_complexity(child, nesting)

        _increment_complexity(node, 0)
        logger.debug("Cognitive complexity: %d", complexity)
        return complexity

    def calculate_halstead_metrics(self, node: ast.AST) -> Dict[str, float]:
        """
        Calculate Halstead metrics for the given AST node.

        Args:
            node: The AST node to analyze

        Returns:
            Dict containing Halstead metrics
        """
        log_debug("Calculating Halstead metrics.")
        operators = set()
        operands = set()
        operator_count = 0
        operand_count = 0

        operator_nodes = (
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
            ast.Pow, ast.MatMult, ast.LShift, ast.RShift, ast.BitOr,
            ast.BitXor, ast.BitAnd, ast.And, ast.Or, ast.Not, ast.Eq,
            ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
            ast.In, ast.NotIn, ast.UAdd, ast.USub, ast.Invert, ast.Assign,
            ast.AugAssign, ast.AnnAssign, ast.Call, ast.Attribute,
            ast.Subscript, ast.Index, ast.Slice
        )

        operand_nodes = (
            ast.Constant, ast.Name, ast.List, ast.Tuple, ast.Set, ast.Dict,
            ast.JoinedStr, ast.FormattedValue, ast.Bytes, ast.NameConstant,
            ast.Num, ast.Str
        )

        for n in ast.walk(node):
            if isinstance(n, operator_nodes):
                operator_name = type(n).__name__
                operators.add(operator_name)
                operator_count += 1

                if isinstance(n, ast.AugAssign):
                    op_type = type(n.op).__name__
                    operators.add(op_type)
                    operator_count += 1

            elif isinstance(n, operand_nodes):
                operand_name = self._get_operand_name(n)
                operands.add(operand_name)
                operand_count += 1

        try:
            n1 = len(operators)
            n2 = len(operands)
            program_length = operator_count + operand_count
            vocabulary_size = n1 + n2

            program_volume = (program_length * math.log2(vocabulary_size)
                            if vocabulary_size > 0 else 0)
            
            difficulty = ((n1 * operand_count) / (2 * n2)) if n2 > 0 else 0
            effort = difficulty * program_volume

            metrics = {
                "program_length": program_length,
                "vocabulary_size": vocabulary_size,
                "program_volume": program_volume,
                "difficulty": difficulty,
                "effort": effort,
                "time_to_program": effort / 18,
                "bugs_delivered": program_volume / 3000,
            }

            log_info(
                "Calculated Halstead metrics - Length=%d, Vocabulary=%d, Volume=%f",
                program_length, vocabulary_size, program_volume
            )
            return metrics

        except Exception as e:
            log_error("Error calculating Halstead metrics: %s", e)
            return {
                "program_length": 0,
                "vocabulary_size": 0,
                "program_volume": 0,
                "difficulty": 0,
                "effort": 0,
                "time_to_program": 0,
                "bugs_delivered": 0,
            }

    def calculate_maintainability_index(self, node: ast.AST) -> float:
        """
        Calculate maintainability index.

        Args:
            node: The AST node to analyze

        Returns:
            The maintainability index score
        """
        halstead = self.calculate_halstead_metrics(node)
        complexity = self.calculate_cyclomatic_complexity(node)
        sloc = self._count_source_lines(node)
        volume = halstead["program_volume"]

        if volume == 0 or sloc == 0:
            return 100.0

        mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(sloc)
        mi = max(0, mi)
        mi = min(100, mi * 100 / 171)

        logger.debug("Maintainability index: %f", mi)
        return round(mi, 2)

    def count_methods(self, node: ast.ClassDef) -> int:
        """
        Count the number of methods in a class.

        Args:
            node: The AST node representing a class

        Returns:
            Number of methods in the class
        """
        return len(
            [
                n
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
        )

    def _count_source_lines(self, node: ast.AST) -> int:
        """
        Count source lines of code (excluding comments and blank lines).

        Args:
            node: The AST node to analyze

        Returns:
            Number of source lines
        """
        if hasattr(ast, "unparse"):
            source = ast.unparse(node)
        else:
            source = self._get_source_code(node)
            
        lines = [line.strip() for line in source.splitlines()]
        sloc = len([line for line in lines if line and not line.startswith("#")])
        logger.debug("Source lines of code: %d", sloc)
        return sloc

    def _get_source_code(self, node: ast.AST) -> str:
        """
        Extract source code for Python < 3.9.

        Args:
            node: The AST node to analyze

        Returns:
            The source code as a string
        """
        return ast.dump(node)

    def _get_operand_name(self, node: ast.AST) -> str:
        """
        Get the name of an operand node.

        Args:
            node: The AST node to analyze

        Returns:
            The operand name as a string
        """
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return str(node.value)
        return ""


class MetricsCollector:
    """Collects and manages operation metrics."""

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self.metrics_store: List[Dict[str, Any]] = []

    async def track_operation(
        self,
        operation_type: str,
        success: bool,
        duration: float,
        usage: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Track detailed operation metrics."""
        metric = {
            "operation_type": operation_type,
            "success": success,
            "duration": duration,
            "usage": usage or {},
            "error": error,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        self.metrics_store.append(metric)

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics data."""
        return {"operations": self.metrics_store.copy()}

    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.metrics_store.clear()

    async def close(self) -> None:
        """Clean up and close the metrics collector."""
        try:
            self.clear_metrics()
            logger.info("MetricsCollector closed successfully")
        except Exception as e:
            logger.error("Error closing MetricsCollector: %s", e)
            raise