"""
Metrics Collector Module

Provides functionality to collect and manage metrics for operations.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
import ast

from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)


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

    def calculate_complexity(self, node: ast.AST) -> int:
        """
        Calculate code complexity for an AST node.

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            int: The calculated complexity.

        Raises:
            ValueError: If the node is not a function, async function, or class definition.
        """
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            raise ValueError("Node must be a function, async function, or class definition")

        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try,
                                  ast.ExceptHandler, ast.With, ast.Assert)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

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
