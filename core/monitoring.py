# monitoring.py

"""
Monitoring Module

Provides system monitoring and performance tracking for Azure OpenAI operations.
Focuses on essential metrics while maintaining efficiency.
"""

import psutil
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from collections import defaultdict
import ast

from core.logger import LoggerSetup
from core.token_management import TokenManager

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


class SystemMonitor:
    """Monitors system resources and performance metrics."""

    def __init__(self, check_interval: int = 60, token_manager: Optional[TokenManager] = None):
        """Initialize system monitor."""
        self.check_interval = check_interval
        self.token_manager = token_manager
        self.start_time = datetime.now()
        self._metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        logger.info("System monitor initialized")

    async def start(self) -> None:
        """Start monitoring system resources."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("System monitoring started")

    async def stop(self) -> None:
        """Stop monitoring system resources."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                metrics = self._collect_system_metrics()
                self._store_metrics(metrics)
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)  # Ensure we don't enter a tight loop on errors


    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {'percent': cpu_percent, 'count': psutil.cpu_count()},
                'memory': {'total': memory.total, 'available': memory.available, 'percent': memory.percent},
                'disk': {'total': disk.total, 'used': disk.used, 'free': disk.free, 'percent': disk.percent}
            }

            if self.token_manager:
                token_stats = self.token_manager.get_usage_stats()
                metrics['tokens'] = token_stats  # Include all token stats

            return metrics

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}


    def _store_metrics(self, metrics: Dict[str, Any]) -> None:
        """Store collected metrics."""
        for key, value in metrics.items():
            if key != 'timestamp':
                self._metrics[key].append({'timestamp': metrics['timestamp'], 'value': value})

        # Keep only the last hour of metrics
        cutoff_time = datetime.now() - timedelta(hours=1)
        for key in self._metrics:
            self._metrics[key] = [m for m in self._metrics[key] if datetime.fromisoformat(m['timestamp']) >= cutoff_time]


    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        try:
            current_metrics = self._collect_system_metrics()
            runtime = (datetime.now() - self.start_time).total_seconds()
            return {
                'current': current_metrics,
                'runtime_seconds': runtime,
                'averages': self._calculate_averages(),
                'status': self._get_system_status()
            }
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {'error': str(e)}

    def _calculate_averages(self) -> Dict[str, float]:
        """Calculate average values for metrics."""
        averages = {}
        for key, values in self._metrics.items():
            if values and key in ('cpu', 'memory', 'disk'):  # Calculate averages for these metrics
                averages[key] = sum(v['value']['percent'] for v in values) / len(values)
        return averages

    def _get_system_status(self) -> str:
        """Determine overall system status."""
        try:
            current = self._collect_system_metrics()

            # Define thresholds (adjust as needed)
            cpu_threshold = 90
            memory_threshold = 90
            disk_threshold = 90

            if (current.get('cpu', {}).get('percent', 0) > cpu_threshold or
                current.get('memory', {}).get('percent', 0) > memory_threshold or
                current.get('disk', {}).get('percent', 0) > disk_threshold):
                return 'critical'
            elif (current.get('cpu', {}).get('percent', 0) > cpu_threshold * 0.8 or
                  current.get('memory', {}).get('percent', 0) > memory_threshold * 0.8 or
                  current.get('disk', {}).get('percent', 0) > disk_threshold * 0.8):
                return 'warning'
            return 'healthy'

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return 'unknown'


    async def __aenter__(self) -> 'SystemMonitor':
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
