import ast
import math
from collections import defaultdict
from typing import Dict, Set, List, Union, Optional, Any
from datetime import datetime
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class MetricsError(Exception):
    """Base exception for metrics calculation errors."""
    pass

class Metrics:
    """
    A class to calculate various code complexity metrics for Python code.
    """

    MAINTAINABILITY_THRESHOLDS: Dict[str, int] = {
        'good': 80,
        'moderate': 60,
        'poor': 40
    }

    def __init__(self) -> None:
        """Initializes the Metrics class."""
        self.module_name: Union[str, None] = None
        self.logger = LoggerSetup.get_logger(__name__)
        self.error_counts: Dict[str, int] = {}
        
    def calculate_function_metrics(self, node: ast.AST) -> Dict[str, Any]:
        """
        Calculates metrics for a function.

        Args:
            node (ast.AST): The AST node representing a function.

        Returns:
            Dict[str, Any]: A dictionary containing function metrics.
        """
        logger.debug(f"Calculating metrics for function: {getattr(node, 'name', 'unknown')}")
        return {
            'cyclomatic_complexity': self.calculate_cyclomatic_complexity(node),
            'cognitive_complexity': self.calculate_cognitive_complexity(node),
            'halstead_metrics': self.calculate_halstead_metrics(node),
            'maintainability_index': self.calculate_maintainability_index(node),
        }

    def calculate_class_metrics(self, node: ast.AST) -> Dict[str, Any]:
        """
        Calculates metrics for a class.

        Args:
            node (ast.AST): The AST node representing a class.

        Returns:
            Dict[str, Any]: A dictionary containing class metrics.
        """
        logger.debug(f"Calculating metrics for class: {getattr(node, 'name', 'unknown')}")
        return {
            'method_count': self.count_methods(node),
            'cyclomatic_complexity': self.calculate_cyclomatic_complexity(node),
            'cognitive_complexity': self.calculate_cognitive_complexity(node),
            'halstead_metrics': self.calculate_halstead_metrics(node),
            'maintainability_index': self.calculate_maintainability_index(node),
        }

    def calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """
        Calculates cyclomatic complexity for a function or class.

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            int: The cyclomatic complexity as an integer.
        """
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.AsyncFor,
                                  ast.Try, ast.ExceptHandler, ast.With,
                                  ast.AsyncWith, ast.BoolOp)):
                complexity += 1
        logger.debug(f"Cyclomatic complexity: {complexity}")
        return complexity

    def calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """
        Calculates cognitive complexity for a function or class.

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            int: The cognitive complexity as an integer.
        """
        complexity = 0

        def _increment_complexity(node: ast.AST, nesting: int) -> None:
            nonlocal complexity
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try,
                                  ast.ExceptHandler, ast.With)):
                complexity += 1
                nesting += 1
            elif isinstance(node, (ast.BoolOp, ast.Break, ast.Continue,
                                   ast.Raise, ast.Return, ast.Yield,
                                   ast.YieldFrom)):
                complexity += nesting + 1

            for child in ast.iter_child_nodes(node):
                _increment_complexity(child, nesting)

        _increment_complexity(node, 0)
        logger.debug(f"Cognitive complexity: {complexity}")
        return complexity

    def calculate_halstead_metrics(self, node: ast.AST) -> Dict[str, float]:
        """
        Calculates Halstead metrics.

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            Dict[str, float]: A dictionary containing various Halstead metrics.
        """
        operators: Set[str] = set()
        operands: Set[str] = set()
        operator_counts: Dict[str, int] = {}
        operand_counts: Dict[str, int] = {}

        for n in ast.walk(node):
            if isinstance(n, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
                              ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
                              ast.FloorDiv, ast.And, ast.Or, ast.Not, ast.Invert,
                              ast.UAdd, ast.USub, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
                              ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
                              ast.Call, ast.Attribute, ast.Subscript, ast.Assign,
                              ast.AugAssign, ast.AnnAssign, ast.Yield, ast.YieldFrom)):
                operators.add(type(n).__name__)
                operator_counts[type(n).__name__] = operator_counts.get(type(n).__name__, 0) + 1
            elif isinstance(n, (ast.Constant, ast.Name, ast.List, ast.Tuple, ast.Set, ast.Dict)):
                operand_name = self._get_operand_name(n)
                if operand_name:
                    operands.add(operand_name)
                    operand_counts[operand_name] = operand_counts.get(operand_name, 0) + 1

        n1 = len(operators)
        n2 = len(operands)
        N1 = sum(operator_counts.values())
        N2 = sum(operand_counts.values())

        program_length = N1 + N2
        program_vocabulary = n1 + n2

        if program_vocabulary == 0 or n2 == 0:  # Handle potential division by zero
            logger.warning("Program vocabulary or operands are zero, returning default Halstead metrics.")
            return {
                'program_length': program_length,
                'program_vocabulary': program_vocabulary,
                'program_volume': 0.0,
                'program_difficulty': 0.0,
                'program_effort': 0.0,
                'time_required_to_program': 0.0,
                'number_delivered_bugs': 0.0
            }

        program_volume = program_length * math.log2(program_vocabulary)
        program_difficulty = (n1 / 2) * (N2 / n2)
        program_effort = program_difficulty * program_volume
        time_required_to_program = program_effort / 18  # seconds
        number_delivered_bugs = program_volume / 3000

        metrics = {
            'program_length': program_length,
            'program_vocabulary': program_vocabulary,
            'program_volume': program_volume,
            'program_difficulty': program_difficulty,
            'program_effort': program_effort,
            'time_required_to_program': time_required_to_program,
            'number_delivered_bugs': number_delivered_bugs
        }
        logger.debug(f"Halstead metrics: {metrics}")
        return metrics

    def calculate_maintainability_index(self, node: ast.AST) -> float:
        """
        Calculates maintainability index.

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            float: The maintainability index as a float.
        """
        halstead = self.calculate_halstead_metrics(node)
        complexity = self.calculate_cyclomatic_complexity(node)
        sloc = self._count_source_lines(node)

        volume = halstead['program_volume']

        if volume == 0 or sloc == 0:  # Handle potential errors
            mi = 100.0  # Maximum value if no volume or lines of code
        else:
            mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(sloc)
            mi = max(0, mi)  # Ensure MI is not negative
            mi = min(100, mi * 100 / 171)  # Normalize to 0-100

        logger.debug(f"Maintainability index: {mi}")
        return round(mi, 2)

    def count_methods(self, node: ast.ClassDef) -> int:
        """
        Counts the number of methods in a class.

        Args:
            node (ast.ClassDef): The AST node representing a class.

        Returns:
            int: The number of methods in the class.
        """
        return len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])

    def _count_source_lines(self, node: ast.AST) -> int:
        """
        Counts source lines of code (excluding comments and blank lines).

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            int: The number of source lines as an integer.
        """
        if hasattr(ast, 'unparse'):
            source = ast.unparse(node)
        else:
            source = self._get_source_code(node)
        lines = [line.strip() for line in source.splitlines()]
        sloc = len([line for line in lines if line and not line.startswith('#')])
        logger.debug(f"Source lines of code: {sloc}")
        return sloc

    def _get_source_code(self, node: ast.AST) -> str:
        """
        Extracts source code for Python < 3.9.

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            str: The source code as a string.
        """
        return ast.dump(node)

    def _get_operand_name(self, node: ast.AST) -> str:
        """
        Gets the name of an operand node.

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            str: The operand name as a string.
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return ''
        

class MetricsCollector:
    """Collects and manages metrics for operations."""

    def __init__(self):
        """
        Initialize the MetricsCollector.

        Attributes:
            metrics_store (List[Dict[str, Any]]): List to store collected metrics.
        """
        self.metrics_store: List[Dict[str, Any]] = []

    async def track_operation(
        self,
        operation_type: str,
        success: bool,
        duration: float,
        usage: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track an operation's metrics.

        Args:
            operation_type (str): Type of the operation being tracked.
            success (bool): Whether the operation was successful.
            duration (float): Duration of the operation in seconds.
            usage (Optional[Dict[str, Any]]): Optional usage statistics.
            error (Optional[str]): Optional error message if the operation failed.
            metadata (Optional[Dict[str, Any]]): Optional additional metadata.

        Raises:
            Exception: If tracking the operation fails.
        """
        metric = {
            'operation_type': operation_type,
            'success': success,
            'duration': duration,
            'usage': usage or {},
            'error': error,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        self.metrics_store.append(metric)
        logger.debug(f"Tracked metrics: {metric}")

    def get_metrics(self) -> List[Dict[str, Any]]:
        """
        Retrieve all collected metrics.

        Returns:
            List[Dict[str, Any]]: List of collected metrics.
        """
        return self.metrics_store

    def clear_metrics(self):
        """
        Clear all collected metrics.

        Raises:
            Exception: If clearing the metrics fails.
        """
        self.metrics_store.clear()

    async def close(self):
        """
        Cleanup and close the metrics collector.

        Raises:
            Exception: If closing the metrics collector fails.
        """
        try:
            self.clear_metrics()  # Clear metrics before closing
            logger.info("MetricsCollector closed successfully")
        except Exception as e:
            logger.error(f"Error closing MetricsCollector: {e}")
            raise