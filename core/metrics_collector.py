"""Metrics collection and storage module."""
import asyncio
from typing import Any, Optional
from datetime import datetime
import json
import os
import uuid

from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.types import MetricData
from core.console import (
    create_progress,
    display_metrics,
    print_error,
    print_info,
    print_warning
)

class MetricsCollector:
    """Collects and stores metrics data for code analysis."""

    # Class variables for singleton pattern
    _instance = None
    _initialized = False

    def __new__(cls, correlation_id: Optional[str] = None) -> 'MetricsCollector':
        """Ensure only one instance exists (singleton pattern)."""
        if not cls._instance:
            instance = super().__new__(cls)
            if not cls._initialized:
                instance.logger = CorrelationLoggerAdapter(LoggerSetup.get_logger(__name__))
                instance.correlation_id = correlation_id or str(uuid.uuid4())
                instance.metrics_history: dict[str, list[dict[str, Any]]] = {}
                instance.operations: list[dict[str, Any]] = []
                instance.current_module_metrics: dict[str, MetricData] = {}
                instance.accumulated_functions = 0
                instance.accumulated_classes = 0
                instance.progress = None
                instance.current_task_id = None
                instance.current_module = None
                instance.has_metrics = False
                instance._load_history()
                cls._initialized = True
            cls._instance = instance
        return cls._instance

    def _format_progress_desc(
        self,
        module_name: str,
        scanned_funcs: int,
        total_funcs: int,
        scanned_classes: int,
        total_classes: int
    ) -> str:
        """Format the progress description."""
        display_name = os.path.basename(module_name) if module_name else "unknown"
        func_ratio = scanned_funcs / total_funcs if total_funcs > 0 else 0
        class_ratio = scanned_classes / total_classes if total_classes > 0 else 0
        return (
            f"[cyan]{display_name:<20}[/cyan] "
            f"[green]Functions:[/green] {scanned_funcs}/{total_funcs} ({func_ratio:.0%}) "
            f"[blue]Classes:[/blue] {scanned_classes}/{total_classes} ({class_ratio:.0%})"
        )

    async def start_progress(self) -> None:
        """Initialize and start progress tracking."""
        async with self.semaphore:
            if self.progress is not None:
                await self.stop_progress()
            self.progress = create_progress()
            self.progress.start()
            self.current_task_id = None

    async def stop_progress(self) -> None:
        """Stop and cleanup progress tracking."""
        async with self.semaphore:
            if self.progress is not None:
                self.progress.stop()
                self.progress = None
                self.current_task_id = None

    async def _init_progress(self, module_name: str, total_items: int) -> None:
        """Initialize or update the progress tracking for a new module."""
        try:
            if self.progress is not None:
                await self.stop_progress()

            await self.start_progress()

            if self.current_task_id is not None:
                self.progress.remove_task(self.current_task_id)

            desc = self._format_progress_desc(module_name, 0, 0, 0, 0)
            self.current_task_id = self.progress.add_task(
                desc, total=max(1, total_items))
            self.current_module = module_name

            self.accumulated_functions = 0
            self.accumulated_classes = 0

        except Exception as e:
            print_error(f"Error initializing progress: {e} with correlation ID: {self.correlation_id}")
            self.current_task_id = None

    def collect_metrics(self, module_name: str, metrics: MetricData) -> None:
        """Collect metrics for a module."""
        try:
            if not module_name or not metrics:
                print_warning(f"Invalid metrics data received with correlation ID: {self.correlation_id}")
                return

            if module_name not in self.metrics_history:
                self.metrics_history[module_name] = []

            current_metrics = self._metrics_to_dict(metrics)
            if module_name in self.current_module_metrics:
                last_metrics = self._metrics_to_dict(
                    self.current_module_metrics[module_name])
                if current_metrics == last_metrics:
                    return

            self.current_module_metrics[module_name] = metrics

            entry = {
                'timestamp': datetime.now().isoformat(),
                'metrics': current_metrics,
                'correlation_id': self.correlation_id
            }

            if module_name in self.metrics_history:
                if self.metrics_history[module_name]:
                    last_entry = self.metrics_history[module_name][-1]
                    if last_entry.get('metrics', {}) != current_metrics:
                        self.metrics_history[module_name].append(entry)
                        self._save_history()
                else:
                    self.metrics_history[module_name] = [entry]
                    self._save_history()
            else:
                self.metrics_history[module_name] = [entry]
                self._save_history()

            total_items = metrics.total_functions + metrics.total_classes
            if total_items > 0:
                if self.current_module != module_name:
                    self._init_progress(module_name, total_items)
                    self._update_progress(
                        module_name,
                        (metrics.scanned_functions, metrics.total_functions),
                        (metrics.scanned_classes, metrics.total_classes)
                    )
        except Exception as e:
            print_error(f"Error collecting metrics: {e} with correlation ID: {self.correlation_id}")

    def update_scan_progress(self, module_name: str, item_type: str, name: str) -> None:
        """Update and log scan progress for a module."""
        try:
            if module_name in self.current_module_metrics:
                metrics = self.current_module_metrics[module_name]

                if item_type == 'function':
                    self.accumulated_functions += 1
                    metrics.scanned_functions = self.accumulated_functions
                    if self.current_task_id is not None and self.progress is not None:
                        self.progress.advance(self.current_task_id)
                        self._update_progress(
                            module_name,
                            (self.accumulated_functions, metrics.total_functions),
                            (self.accumulated_classes, metrics.total_classes)
                        )
                elif item_type == 'class':
                    self.accumulated_classes += 1
                    metrics.scanned_classes = self.accumulated_classes
                    if self.current_task_id is not None and self.progress is not None:
                        self.progress.advance(self.current_task_id)
                        self._update_progress(
                            module_name,
                            (self.accumulated_functions, metrics.total_functions),
                            (self.accumulated_classes, metrics.total_classes)
                        )

        except Exception as e:
            print_error(f"Error updating scan progress: {e} with correlation ID: {self.correlation_id}")

    def _update_progress(
        self,
        module_name: str,
        functions: tuple[int, int],
        classes: tuple[int, int]
    ) -> None:
        """Update the progress tracking with current counts."""
        try:
            if self.current_task_id is None or self.progress is None:
                return

            scanned_funcs, total_funcs = functions
            scanned_classes, total_classes = classes

            total_items = total_funcs + total_classes
            completed_items = scanned_funcs + scanned_classes

            desc = self._format_progress_desc(
                module_name,
                scanned_funcs,
                total_funcs,
                scanned_classes,
                total_classes
            )
            self.progress.update(
                self.current_task_id,
                description=desc,
                completed=completed_items,
                total=max(1, total_items)
            )

        except Exception as e:
            print_error(f"Error updating progress: {e} with correlation ID: {self.correlation_id}")

    def _metrics_to_dict(self, metrics: MetricData) -> dict[str, Any]:
        """Convert MetricData to dictionary format."""
        try:
            return {
                'cyclomatic_complexity': getattr(metrics, 'cyclomatic_complexity', 0),
                'cognitive_complexity': getattr(metrics, 'cognitive_complexity', 0),
                'maintainability_index': getattr(metrics, 'maintainability_index', 0.0),
                'halstead_metrics': getattr(metrics, 'halstead_metrics', {}),
                'lines_of_code': getattr(metrics, 'lines_of_code', 0),
                'total_functions': getattr(metrics, 'total_functions', 0),
                'scanned_functions': getattr(metrics, 'scanned_functions', 0),
                'function_scan_ratio': getattr(metrics, 'function_scan_ratio', 0.0),
                'total_classes': getattr(metrics, 'total_classes', 0),
                'scanned_classes': getattr(metrics, 'scanned_classes', 0),
                'class_scan_ratio': getattr(metrics, 'class_scan_ratio', 0.0),
                'complexity_graph': getattr(metrics, 'complexity_graph', None)
            }
        except Exception as e:
            print_error(f"Error converting metrics to dict: {e} with correlation ID: {self.correlation_id}")
            return {}

    async def track_operation(
        self,
        operation_type: str,
        success: bool,
        duration: float,
        metadata: Optional[dict[str, Any]] = None,
        usage: Optional[dict[str, Any]] = None
    ) -> None:
        """Track an operation with its metrics."""
        try:
            operation = {
                'timestamp': datetime.now().isoformat(),
                'operation_type': operation_type,
                'success': success,
                'duration': duration,
                'correlation_id': self.correlation_id
            }
            if metadata:
                operation['metadata'] = metadata
            if usage:
                operation['usage'] = usage

            self.operations.append(operation)

        except Exception as e:
            print_error(f"Error tracking operation: {e} with correlation ID: {self.correlation_id}")

    async def close(self) -> None:
        """Clean up resources and save final state."""
        try:
            self.stop_progress()
            self._save_history()
        except Exception as e:
            print_error(f"Error closing MetricsCollector: {e} with correlation ID: {self.correlation_id}")

    def _load_history(self) -> None:
        """Load metrics history from storage."""
        try:
            if os.path.exists('metrics_history.json'):
                with open('metrics_history.json', 'r') as f:
                    self.metrics_history = json.load(f)
        except Exception as e:
            print_error(f"Error loading metrics history: {str(e)} with correlation ID: {self.correlation_id}")
            self.metrics_history = {}

    def _save_history(self) -> None:
        """Save metrics history to storage."""
        try:
            with open('metrics_history.json', 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
        except Exception as e:
            print_error(f"Error saving metrics history: {str(e)} with correlation ID: {self.correlation_id}")

    def clear_history(self) -> None:
        """Clear all metrics history."""
        self.metrics_history = {}
        if os.path.exists('metrics_history.json'):
            os.remove('metrics_history.json')

    def get_metrics(self) -> dict[str, Any]:
        """Get the current metrics data."""
        return {
            'current_metrics': self.current_module_metrics,
            'history': self.metrics_history,
            'operations': self.operations
        }

    def get_metrics_history(self, module_name: str) -> list[dict[str, Any]]:
        """Get metrics history for a specific module."""
        return self.metrics_history.get(module_name, [])

    async def display_metrics(self) -> None:
        """Display collected metrics and system performance metrics."""
        try:
            print_info(f"Displaying metrics with correlation ID: {self.correlation_id}")
            collected_metrics = self.metrics_history

            if not collected_metrics:
                print_warning(f"No metrics collected with correlation ID: {self.correlation_id}")
                return

            for module_name, history in collected_metrics.items():
                if history:
                    latest = history[-1]['metrics']
                    display_metrics(
                        {
                            "Module": module_name,
                            "Scanned Functions": latest['scanned_functions'],
                            "Total Functions": latest['total_functions'],
                            "Scanned Classes": latest['scanned_classes'],
                            "Total Classes": latest['total_classes'],
                            "Complexity Score": latest['cyclomatic_complexity'],
                            "Maintainability": f"{latest['maintainability_index']:.2f}"
                        },
                        title=f"Metrics for {module_name}"
                    )

        except Exception as e:
            print_error(f"Error displaying metrics: {e} with correlation ID: {self.correlation_id}")
