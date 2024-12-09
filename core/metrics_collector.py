"""
Metrics Collector Module.

Collects and manages operation metrics.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from core.logger import LoggerSetup, log_debug, log_error, log_info

class MetricsCollector:
    """Collects and manages operation metrics."""

    def __init__(self) -> None:
        """Initialize MetricsCollector."""
        self.metrics: List[Dict[str, Any]] = []
        self.logger = LoggerSetup.get_logger(__name__)
        self.logger.info("MetricsCollector initialized.")

    async def track_operation(
        self,
        operation_type: str,
        success: bool,
        duration: float,
        usage: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Track an operation's metrics.

        Args:
            operation_type (str): Type of the operation (e.g., 'function_metrics').
            success (bool): Whether the operation was successful.
            duration (float): Duration of the operation in seconds.
            usage (Optional[Dict[str, Any]]): Additional usage data.
            error (Optional[str]): Error message if the operation failed.
            metadata (Optional[Dict[str, Any]]): Additional metadata.
        """
        metric = {
            "operation_type": operation_type,
            "success": success,
            "duration": duration,
            "usage": usage or {},
            "validation_success": metadata.get("validation_success", False) if metadata else False,
            "timestamp": datetime.utcnow().isoformat()
        }
        if error:
            metric["error"] = error
        self.metrics.append(metric)
        self.logger.debug(f"Tracked operation: {metric}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Retrieve collected metrics.

        Returns:
            Dict[str, Any]: A dictionary containing all tracked operations.
        """
        self.logger.debug("Retrieving collected metrics.")
        return {"operations": self.metrics}

    def clear_metrics(self) -> None:
        """
        Clear all collected metrics.
        """
        self.metrics.clear()
        self.logger.info("Cleared all metrics.")

    async def close(self) -> None:
        """
        Cleanup metrics collector resources (if any).
        """
        # If there are resources to clean up, handle them here.
        # For now, it's a placeholder.
        self.logger.info("MetricsCollector cleanup completed.")