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
        """Initialize the MetricsCollector."""
        self.metrics_store: List[Dict[str, Any]] = []

    async def track_operation(
        self,
        operation_type: str,
        success: bool,
        duration: float,
        usage: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Track an operation's metrics."""
        metric = {
            'operation_type': operation_type,
            'success': success,
            'duration': duration,
            'usage': usage or {},
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.metrics_store.append(metric)
        logger.debug(f"Tracked metrics: {metric}")

    def calculate_complexity(self, node: ast.AST) -> int:
        """Calculate code complexity for an AST node."""
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return 0

        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try,
                                  ast.ExceptHandler, ast.With, ast.Assert)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def get_metrics(self) -> List[Dict[str, Any]]:
        """Retrieve all collected metrics."""
        return self.metrics_store

    def clear_metrics(self):
        """Clear all collected metrics."""
        self.metrics_store.clear()

    async def close(self):
        """Cleanup and close the metrics collector."""
        try:
            self.clear_metrics()  # Clear metrics before closing
            logger.info("MetricsCollector closed successfully")
        except Exception as e:
            logger.error(f"Error closing MetricsCollector: {e}")
            raise
