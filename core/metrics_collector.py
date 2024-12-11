"""Metrics collection and storage module."""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import os
import sys
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.console import Console

from core.logger import LoggerSetup
from core.types import MetricData
from core.console import console


class MetricsCollector:
    """Collects and stores metrics data for code analysis."""

    # Class variables for singleton pattern
    _instance = None
    _initialized = False

    def __new__(cls, correlation_id: Optional[str] = None) -> 'MetricsCollector':
        """Ensure only one instance exists (singleton pattern).

        Args:
            correlation_id: Optional correlation ID for tracking related operations

        Returns:
            The singleton MetricsCollector instance
        """
        if not cls._instance:
            instance = super().__new__(cls)
            # Initialize here instead of in __init__ to avoid recursion
            if not cls._initialized:
                instance.logger = LoggerSetup.get_logger(__name__)
                instance.correlation_id = correlation_id
                instance.metrics_history = {}
                instance.operations = []
                instance.current_module_metrics = {}
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

    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """Initialize metrics collector.

        Args:
            correlation_id: Optional correlation ID for tracking related operations
        """
        # Skip initialization since it's done in __new__
        if MetricsCollector._initialized:
            return

    def _format_progress_desc(
        self,
        module_name: str,
        scanned_funcs: int,
        total_funcs: int,
        scanned_classes: int,
        total_classes: int
    ) -> str:
        """Format the progress description.

        Args:
            module_name: Name of the module
            scanned_funcs: Number of scanned functions
            total_funcs: Total number of functions
            scanned_classes: Number of scanned classes
            total_classes: Total number of classes

        Returns:
            Formatted description string
        """
        # Use just the filename from the module path
        display_name = os.path.basename(
            module_name) if module_name else "unknown"
        func_ratio = scanned_funcs / total_funcs if total_funcs > 0 else 0
        class_ratio = scanned_classes / total_classes if total_classes > 0 else 0
        return (
            f"[cyan]{display_name:<20}[/cyan] "
            f"[green]Functions:[/green] {scanned_funcs}/{total_funcs} ({func_ratio:.0%}) "
            f"[blue]Classes:[/blue] {scanned_classes}/{total_classes} ({class_ratio:.0%})"
        )

    def start_progress(self) -> None:
        """Initialize and start progress tracking."""
        if self.progress is None:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(complete_style="green"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
                expand=True
            )
            self.progress.start()

    def stop_progress(self) -> None:
        """Stop and cleanup progress tracking."""
        if self.progress is not None:
            self.progress.stop()
            self.progress = None
            self.current_task_id = None

    def _init_progress(self, module_name: str, total_items: int) -> None:
        """Initialize or update the progress tracking for a new module.

        Args:
            module_name: Name of the module being processed
            total_items: Total number of items to process
        """
        try:
            # Ensure progress is started
            if self.progress is None:
                self.start_progress()

            # Stop existing task if any
            if self.current_task_id is not None:
                self.progress.remove_task(self.current_task_id)
                self.current_task_id = None

            # Create new progress tracking with initial description
            desc = self._format_progress_desc(module_name, 0, 0, 0, 0)
            self.current_task_id = self.progress.add_task(
                desc, total=max(1, total_items))
            self.current_module = module_name

            # Reset accumulated counts
            self.accumulated_functions = 0
            self.accumulated_classes = 0

        except Exception as e:
            self.logger.error(f"Error initializing progress: {e}")

    def collect_metrics(self, module_name: str, metrics: MetricData) -> None:
        """Collect metrics for a module.

        Args:
            module_name: Name of the module being analyzed
            metrics: MetricData object containing the metrics
        """
        try:
            if not module_name or not metrics:
                self.logger.warning("Invalid metrics data received")
                return

            if module_name not in self.metrics_history:
                self.metrics_history[module_name] = []

            # Check if metrics have changed before storing
            current_metrics = self._metrics_to_dict(metrics)
            if module_name in self.current_module_metrics:
                last_metrics = self._metrics_to_dict(
                    self.current_module_metrics[module_name])
                if current_metrics == last_metrics:
                    return

            # Update current module metrics silently
            self.current_module_metrics[module_name] = metrics

            # Create metrics entry without output
            entry = {
                'timestamp': datetime.now().isoformat(),
                'metrics': current_metrics,
                'correlation_id': self.correlation_id
            }

            # Store metrics silently
            if module_name in self.metrics_history:
                # Only store if metrics have changed and history exists
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

            # Initialize progress for new module if it has items to process
            total_items = metrics.total_functions + metrics.total_classes
            if total_items > 0:
                if self.current_module != module_name:
                    self._init_progress(module_name, total_items)
                    # Update progress with initial counts
                    self._update_progress(
                        module_name,
                        (metrics.scanned_functions, metrics.total_functions),
                        (metrics.scanned_classes, metrics.total_classes)
                    )
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")

    def update_scan_progress(self, module_name: str, item_type: str, name: str) -> None:
        """Update and log scan progress for a module.

        Args:
            module_name: Name of the module being analyzed
            item_type: Type of item scanned ('function' or 'class')
            name: Name of the scanned item
        """
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
            self.logger.error(f"Error updating scan progress: {e}")

    def _update_progress(
        self,
        module_name: str,
        functions: Tuple[int, int],
        classes: Tuple[int, int]
    ) -> None:
        """Update the progress tracking with current counts.

        Args:
            module_name: Name of the module being processed
            functions: Tuple of (scanned, total) functions
            classes: Tuple of (scanned, total) classes
        """
        try:
            if self.current_task_id is None or self.progress is None:
                return

            scanned_funcs, total_funcs = functions
            scanned_classes, total_classes = classes

            # Calculate overall completion
            total_items = total_funcs + total_classes
            completed_items = scanned_funcs + scanned_classes

            # Update progress description and completion
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
                total=max(1, total_items)  # Ensure non-zero total
            )

        except Exception as e:
            self.logger.error(f"Error updating progress: {e}")

    def _metrics_to_dict(self, metrics: MetricData) -> Dict[str, Any]:
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
            self.logger.error(f"Error converting metrics to dict: {e}")
            return {}

    async def track_operation(
        self,
        operation_type: str,
        success: bool,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, Any]] = None
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

            # Silently track operation without output

        except Exception as e:
            self.logger.error(f"Error tracking operation: {e}")

    async def close(self) -> None:
        """Clean up resources and save final state."""
        try:
            self.stop_progress()
            self._save_history()
        except Exception as e:
            self.logger.error(f"Error closing MetricsCollector: {e}")

    def _load_history(self) -> None:
        """Load metrics history from storage."""
        try:
            if os.path.exists('metrics_history.json'):
                with open('metrics_history.json', 'r') as f:
                    self.metrics_history = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading metrics history: {str(e)}")
            self.metrics_history = {}

    def _save_history(self) -> None:
        """Save metrics history to storage."""
        try:
            with open('metrics_history.json', 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving metrics history: {str(e)}")

    def clear_history(self) -> None:
        """Clear all metrics history."""
        self.metrics_history = {}
        if os.path.exists('metrics_history.json'):
            os.remove('metrics_history.json')

    def get_metrics(self) -> Dict[str, Any]:
        """Get the current metrics data.

        Returns:
            Dictionary containing current metrics data and history
        """
        return {
            'current_metrics': self.current_module_metrics,
            'history': self.metrics_history,
            'operations': self.operations
        }

    def get_metrics_history(self, module_name: str) -> List[Dict[str, Any]]:
        """Get metrics history for a specific module.

        Args:
            module_name: Name of the module to get history for

        Returns:
            List of historical metrics entries for the module
        """
        return self.metrics_history.get(module_name, [])
