# monitoring.py
"""
Monitoring Module

Provides functionality to track and log metrics related to system performance,
operation success rates, and resource usage.
"""

import json
import logging
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, Any, List, Optional

import psutil  # Ensure this library is installed for system metrics

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class OperationMetrics:
    """Tracks metrics for a specific type of operation."""
    
    def __init__(self):
        self.total_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_duration = 0.0
        self.total_tokens = 0
        self.errors = Counter()

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of the operation."""
        return (self.success_count / self.total_count * 100) if self.total_count > 0 else 0.0

    @property
    def average_duration(self) -> float:
        """Calculate the average duration of the operation."""
        return (self.total_duration / self.total_count) if self.total_count > 0 else 0.0

class MetricsCollector:
    """Collects and tracks metrics for various operations."""
    
    def __init__(self):
        self.operations = defaultdict(OperationMetrics)
        self.cache_metrics = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0
        }
        self.start_time = datetime.now()
        logger.debug("Metrics collector initialized")

    def track_operation(
        self,
        operation_type: str,
        success: bool,
        duration: Optional[float] = None,
        tokens_used: Optional[int] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Track an operation's metrics.

        Args:
            operation_type: Type of operation being tracked
            success: Whether the operation was successful
            duration: Operation duration in seconds
            tokens_used: Number of tokens used
            error: Error message if operation failed
        """
        metrics = self.operations[operation_type]
        metrics.total_count += 1
        
        if success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1
            if error:
                metrics.errors[error] += 1

        if duration is not None:
            metrics.total_duration += duration

        if tokens_used is not None:
            metrics.total_tokens += tokens_used

        logger.debug(f"Tracked {operation_type} operation: success={success}, duration={duration}")

    def track_cache_hit(self) -> None:
        """Track a cache hit."""
        self.cache_metrics['hits'] += 1
        self.cache_metrics['total_requests'] += 1

    def track_cache_miss(self) -> None:
        """Track a cache miss."""
        self.cache_metrics['misses'] += 1
        self.cache_metrics['total_requests'] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.

        Returns:
            Dict[str, Any]: Collected metrics and statistics
        """
        metrics = {
            'operations': {},
            'cache': self._get_cache_metrics(),
            'system': SystemMonitor.get_system_metrics(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }

        for op_type, op_metrics in self.operations.items():
            metrics['operations'][op_type] = {
                'total_count': op_metrics.total_count,
                'success_count': op_metrics.success_count,
                'failure_count': op_metrics.failure_count,
                'success_rate': op_metrics.success_rate,
                'average_duration': op_metrics.average_duration,
                'total_tokens': op_metrics.total_tokens,
                'common_errors': dict(op_metrics.errors.most_common(5))
            }

        return metrics

    def _get_cache_metrics(self) -> Dict[str, Any]:
        """Get cache-related metrics."""
        total = self.cache_metrics['total_requests']
        hit_rate = (self.cache_metrics['hits'] / total * 100) if total > 0 else 0
        
        return {
            'hits': self.cache_metrics['hits'],
            'misses': self.cache_metrics['misses'],
            'total_requests': total,
            'hit_rate': round(hit_rate, 2)
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.operations.clear()
        self.cache_metrics = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0
        }
        self.start_time = datetime.now()
        logger.info("Metrics collector reset")

class SystemMonitor:
    """Monitors system resources and performance."""

    def __init__(self):
        """Initialize system monitor."""
        self.start_time = datetime.now()
        self._operation_times: Dict[str, List[float]] = defaultdict(list)
        self._error_counts: Counter = Counter()
        logger.debug("System monitor initialized")

    @staticmethod
    def get_system_metrics() -> Dict[str, Any]:
        """
        Get current system metrics.

        Returns:
            Dict[str, Any]: System resource usage metrics
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'disk_percent': disk.percent,
                'disk_used_gb': round(disk.used / (1024**3), 2),
                'disk_total_gb': round(disk.total / (1024**3), 2)
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {}

    def log_operation_complete(
        self,
        operation_name: str,
        execution_time: float,
        tokens_used: int,
        error: Optional[str] = None
    ) -> None:
        """
        Log completion of an operation.

        Args:
            operation_name: Name of the operation
            execution_time: Time taken to execute
            tokens_used: Number of tokens used
            error: Optional error message
        """
        self._operation_times[operation_name].append(execution_time)
        if error:
            self._error_counts[f"{operation_name}:{error}"] += 1
        logger.debug(f"Operation logged: {operation_name}, time={execution_time:.2f}s, tokens={tokens_used}")

    def log_api_request(self, success: bool) -> None:
        """
        Log an API request.

        Args:
            success: Whether the request was successful
        """
        if not success:
            self._error_counts['api_request_failure'] += 1

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.

        Returns:
            Dict[str, Any]: Summary of all collected metrics
        """
        summary = {
            'system': self.get_system_metrics(),
            'operations': {},
            'errors': dict(self._error_counts.most_common(10)),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }

        for op_name, times in self._operation_times.items():
            if times:
                summary['operations'][op_name] = {
                    'count': len(times),
                    'average_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }

        return summary

    def reset(self) -> None:
        """Reset all monitoring data."""
        self.start_time = datetime.now()
        self._operation_times.clear()
        self._error_counts.clear()
        logger.info("System monitor reset")

    def export_metrics(self, filepath: str) -> None:
        """
        Export metrics to a JSON file.

        Args:
            filepath: Path to save the metrics file
        """
        try:
            metrics = self.get_metrics_summary()
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.reset()