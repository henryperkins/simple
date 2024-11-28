"""
Monitoring Module

Provides system monitoring and performance tracking for Azure OpenAI operations.
Focuses on essential metrics while maintaining efficiency.
"""

import psutil
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict

from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class MetricsCollector:
    """Collects and manages metrics for operations."""

    def __init__(self):
        """Initialize the MetricsCollector with an empty metrics store."""
        self.metrics_store: List[Dict[str, Any]] = []

    async def track_operation(
        self, 
        operation_type: str, 
        success: bool, 
        duration: float, 
        usage: Dict[str, Any] = None, 
        error: str = None
    ):
        """
        Track an operation's metrics.

        Args:
            operation_type (str): The type of operation being tracked.
            success (bool): Whether the operation was successful.
            duration (float): The duration of the operation in seconds.
            usage (Dict[str, Any], optional): Additional usage metrics.
            error (str, optional): Error message if the operation failed.
        """
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

    def get_metrics(self) -> List[Dict[str, Any]]:
        """Retrieve all collected metrics."""
        return self.metrics_store

    def clear_metrics(self):
        """Clear all collected metrics."""
        self.metrics_store.clear()
        
    async def close(self):
        """Cleanup and close the metrics collector."""
        try:
            # Save any pending metrics or perform cleanup
            self.clear_metrics()
            logger.info("MetricsCollector closed successfully")
        except Exception as e:
            logger.error(f"Error closing MetricsCollector: {e}")
            raise
        
class SystemMonitor:
    """Monitors system resources and performance metrics."""

    def __init__(self, check_interval: int = 60):
        """
        Initialize system monitor.

        Args:
            check_interval: Interval between system checks in seconds
        """
        self.check_interval = check_interval
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
                await asyncio.sleep(self.check_interval)

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count()
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent
                }
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}

    def _store_metrics(self, metrics: Dict[str, Any]) -> None:
        """Store collected metrics."""
        for key, value in metrics.items():
            if key != 'timestamp':
                self._metrics[key].append({
                    'timestamp': metrics['timestamp'],
                    'value': value
                })

        # Keep only last hour of metrics
        max_entries = 3600 // self.check_interval
        for key in self._metrics:
            if len(self._metrics[key]) > max_entries:
                self._metrics[key] = self._metrics[key][-max_entries:]

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
            if values:
                if key == 'cpu':
                    averages[key] = sum(v['value']['percent'] for v in values) / len(values)
                elif key in ['memory', 'disk']:
                    averages[key] = sum(v['value']['percent'] for v in values) / len(values)
        return averages

    def _get_system_status(self) -> str:
        """Determine overall system status."""
        try:
            current = self._collect_system_metrics()
            
            # Define thresholds
            CPU_THRESHOLD = 90
            MEMORY_THRESHOLD = 90
            DISK_THRESHOLD = 90

            if (current.get('cpu', {}).get('percent', 0) > CPU_THRESHOLD or
                current.get('memory', {}).get('percent', 0) > MEMORY_THRESHOLD or
                current.get('disk', {}).get('percent', 0) > DISK_THRESHOLD):
                return 'critical'
            elif (current.get('cpu', {}).get('percent', 0) > CPU_THRESHOLD * 0.8 or
                  current.get('memory', {}).get('percent', 0) > MEMORY_THRESHOLD * 0.8 or
                  current.get('disk', {}).get('percent', 0) > DISK_THRESHOLD * 0.8):
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