"""
Metrics collection and storage module.
"""

from typing import Any, Union
from datetime import datetime
import json
import os
import uuid

from core.logger import LoggerSetup
from core.types.base import MetricData
from core.console import (
    print_error,
    print_info,
    print_status,
    print_section_break,
    display_metrics,
    print_warning,
)


class MetricsCollector:
    """Collects and stores metrics data for code analysis."""

    _instance = None
    _initialized = False

    def __new__(cls, correlation_id: str | None = None) -> "MetricsCollector":
        """Ensure only one instance exists (singleton pattern)."""
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, correlation_id: str | None = None) -> None:
        """Initialize the metrics collector.

        Args:
            correlation_id: Optional correlation ID for tracking metrics
        """
        if not self._initialized:
            self.logger = LoggerSetup.get_logger(__name__, correlation_id)
            self.correlation_id = correlation_id or str(uuid.uuid4())
            self.metrics_history: dict[str, list[dict[str, Any]]] = {}
            self.operations: list[dict[str, Any]] = []
            self.current_module_metrics: dict[str, Any] = {}
            self.accumulated_functions = 0
            self.accumulated_classes = 0
            self.has_metrics = False
            self._load_history()
            self.__class__._initialized = True

    def collect_metrics(self, module_name: str, metrics: MetricData) -> None:
        """Collect metrics for a module."""
        try:
            if not module_name or module_name == "default_module":
                if hasattr(metrics, "module_name") and metrics.module_name:
                    module_name = metrics.module_name

            if not module_name or not metrics:
                self.logger.warning(
                    f"Invalid metrics data received for module '{module_name}' "
                    f"with correlation ID: {self.correlation_id}"
                )
                return

            if module_name not in self.metrics_history:
                self.metrics_history[module_name] = []

            current_metrics = self._metrics_to_dict(metrics)
            if module_name in self.current_module_metrics:
                last_metrics = self._metrics_to_dict(
                    self.current_module_metrics[module_name]
                )
                if current_metrics == last_metrics:
                    return

            self.current_module_metrics[module_name] = metrics

            print_section_break()
            print_info(f"📊 Metrics Collected for Module: {module_name} 📊")
            display_metrics(current_metrics, title=f"Module: {module_name}")
            print_section_break()

            entry = {
                "timestamp": datetime.now().isoformat(),
                "metrics": current_metrics,
                "correlation_id": self.correlation_id,
            }

            if module_name in self.metrics_history:
                if self.metrics_history[module_name]:
                    last_entry = self.metrics_history[module_name][-1]
                    if last_entry.get("metrics", {}) != current_metrics:
                        self.metrics_history[module_name].append(entry)
                        self._save_history()
                else:
                    self.metrics_history[module_name] = [entry]
                    self._save_history()
            else:
                self.metrics_history[module_name] = [entry]
                self._save_history()

        except (IOError, ValueError) as e:
            print_error(
                f"🔥 Error collecting metrics for module '{module_name}': {e}",
                correlation_id=self.correlation_id,
            )

    def update_scan_progress(self, module_name: str, item_type: str, count: int = 0) -> None:
        """Update the scan progress for functions or classes."""
        try:
            if module_name == "default_module":
                return

            if item_type == 'function':
                self.accumulated_functions = 0  # Initialize to 0
                if self.current_module_metrics and self.current_module:
                    metrics = self.current_module_metrics[self.current_module]
                    metrics.total_functions = 0  # Initialize to 0
                    metrics.scanned_functions = 0  # Initialize to 0
            elif item_type == 'class':
                self.accumulated_classes = 0  # Initialize to 0
                if self.current_module_metrics and self.current_module:
                    metrics = self.current_module_metrics[self.current_module]
                    metrics.total_classes = 0  # Initialize to 0
                    metrics.scanned_classes = 0  # Initialize to 0

            self.has_metrics = True

        except Exception as e:
            self.logger.error(
                f"Error updating scan progress: {e} with correlation ID: {self.correlation_id}"
            )

    def _metrics_to_dict(self, metrics: MetricData) -> dict[str, Any]:
        """Convert MetricData to dictionary format."""
        try:
            return {
                "cyclomatic_complexity": getattr(metrics, "cyclomatic_complexity", 0),
                "cognitive_complexity": getattr(metrics, "cognitive_complexity", 0),
                "maintainability_index": getattr(metrics, "maintainability_index", 0.0),
                "halstead_metrics": getattr(metrics, "halstead_metrics", {}),
                "lines_of_code": getattr(metrics, "lines_of_code", 0),
                "total_functions": getattr(metrics, "total_functions", 0),
                "scanned_functions": getattr(metrics, "scanned_functions", 0),
                "function_scan_ratio": metrics.function_scan_ratio,
                "total_classes": getattr(metrics, "total_classes", 0),
                "scanned_classes": getattr(metrics, "scanned_classes", 0),
                "class_scan_ratio": metrics.class_scan_ratio,
                "complexity_graph": getattr(metrics, "complexity_graph", None),
            }
        except Exception as e:
            self.logger.error(
                f"Error converting metrics to dict: {e} with correlation ID: {self.correlation_id}"
            )
            return {}

    async def track_operation(
        self,
        operation_type: str,
        success: bool,
        duration: float,
        **kwargs: Any,
    ) -> None:
        metadata = kwargs.get("metadata")
        usage = kwargs.get("usage")
        """Track an operation with its metrics."""
        try:
            operation: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "operation_type": operation_type,
                "success": success,
                "duration": duration,
                "correlation_id": self.correlation_id,
            }
            if metadata:
                operation["metadata"] = metadata
            if usage:
                operation["usage"] = usage

            self.operations.append(operation)

            status_details = {"Success": "✅" if success else "❌", "Duration": f"{duration:.2f}s"}
            if metadata:
                status_details.update(metadata)

            print_status(
                f"🔄 Operation: {operation_type}",
                status_details,
            )

        except Exception as e:
            print_error(
                f"🔥 Error tracking operation '{operation_type}': {e}",
                correlation_id=self.correlation_id
            )

    async def close(self) -> None:
        """Clean up resources and save final state."""
        try:
            self._save_history()
        except Exception as e:
            self.logger.error(
                f"Error closing MetricsCollector: {e} with correlation ID: {self.correlation_id}"
            )

    def _load_history(self) -> None:
        """Load metrics history from storage."""
        try:
            if os.path.exists("metrics_history.json"):
                with open("metrics_history.json", "r", encoding="utf-8") as f:
                    self.metrics_history = json.load(f)
                self.logger.info("Metrics history loaded successfully.")  # Added info log
        except Exception as e:
            self.logger.error(
                f"Error loading metrics history: {str(e)} with correlation ID: {self.correlation_id}"
            )
            print_warning("⚠️ Could not load previous metrics history.")  # Added user warning
            self.metrics_history = {}

    def _save_history(self) -> None:
        """Save metrics history to storage."""
        try:
            with open("metrics_history.json", "w", encoding="utf-8") as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
            self.logger.info("Metrics history saved successfully.")  # Added info log
        except Exception as e:
            self.logger.error(
                f"Error saving metrics history: {str(e)} with correlation ID: {self.correlation_id}",
                exc_info=True
            )
            print_warning("⚠️ Could not save current metrics history.")  # Added user warning

    def clear_history(self) -> None:
        """Clear all metrics history."""
        self.metrics_history = {}
        if os.path.exists("metrics_history.json"):
            os.remove("metrics_history.json")

    def get_metrics(self) -> dict[str, Any]:
        """Get the current metrics data."""
        return {
            "current_metrics": self.current_module_metrics,
            "history": self.metrics_history,
            "operations": self.operations,
        }

    def get_metrics_history(self, module_name: str) -> list[dict[str, Any]]:
        """Get metrics history for a specific module."""
        return self.metrics_history.get(module_name, [])

    def collect_token_usage(self, prompt_tokens: int, completion_tokens: int, cost: float, model: str) -> None:
        """Collect metrics specifically for token usage."""
        try:
            self.operations.append({
                "timestamp": datetime.now().isoformat(),
                "operation_type": "token_usage",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "total_cost": cost,
                "model": model,
                "correlation_id": self.correlation_id,
            })
            self.logger.info(
                f"Token usage collected: {prompt_tokens + completion_tokens} tokens, ${cost:.4f}.",
                extra={"model": model, "correlation_id": self.correlation_id}
            )
        except Exception as e:
            self.logger.error(f"Error collecting token usage: {e}", exc_info=True)

    def collect_validation_metrics(self, success: bool) -> None:
        """Collect metrics for schema validation results."""
        try:
            if success:
                self._increment_metric("validation_success")
            else:
                self._increment_metric("validation_failure")
        except Exception as e:
            self.logger.error(f"Error collecting validation metrics: {e}", exc_info=True)

    def _increment_metric(self, metric_name: str) -> None:
        """Increment a specific metric."""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = 0
        self.metrics_history[metric_name] += 1

    def get_aggregated_token_usage(self) -> dict[str, Union[int, float]]:
        """Aggregate token usage statistics across operations."""
        total_prompt_tokens = sum(
            op.get("prompt_tokens", 0)
            for op in self.operations
            if op["operation_type"] == "token_usage"
        )
        total_completion_tokens = sum(
            op.get("completion_tokens", 0)
            for op in self.operations
            if op["operation_type"] == "token_usage"
        )
        total_cost = sum(
            op.get("total_cost", 0)
            for op in self.operations
            if op["operation_type"] == "token_usage"
        )

        return {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "total_cost": total_cost,
        }
