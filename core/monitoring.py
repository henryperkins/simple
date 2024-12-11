"""
Monitoring Module.

Provides system monitoring and performance tracking for operations, integrating detailed logging.
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import psutil

from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.metrics_collector import MetricsCollector
from api.token_management import TokenManager


class SystemMonitor:
    """Monitors system resources and performance metrics."""

    def __init__(
        self,
        check_interval: int = 60,
        token_manager: Optional[TokenManager] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """
        Initialize system monitor.

        Args:
            check_interval: Interval in seconds between metric checks
            token_manager: Optional token manager for tracking token usage
            metrics_collector: Optional metrics collector for tracking metrics
            correlation_id: Optional correlation ID for tracking related operations
        """
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__),
            correlation_id=correlation_id
        )
        self.correlation_id = correlation_id
        self.check_interval = check_interval
        self.token_manager = token_manager
        self.start_time = datetime.now()
        self._metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.metrics_collector = metrics_collector or MetricsCollector(
            correlation_id=correlation_id)
        self.logger.info("System monitor initialized")

    async def start(self) -> None:
        """Start monitoring system resources."""
        if self._running:
            self.logger.warning("System monitoring is already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        self.logger.info("System monitoring started")

    async def stop(self) -> None:
        """Stop monitoring system resources."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                self.logger.debug("Monitoring task was cancelled")
        self.logger.info("System monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                metrics = self._collect_system_metrics()
                await self._store_metrics(metrics)
                self.logger.debug("System metrics collected and stored")
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(
                    "Error in monitoring loop: %s", e, exc_info=True)
                await asyncio.sleep(self.check_interval)

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """
        Collect current system metrics.

        Returns:
            Dictionary containing system metrics
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent,
                },
            }

            if self.token_manager:
                token_stats = self.token_manager.get_usage_stats()
                metrics["tokens"] = token_stats

            self.logger.debug("Collected system metrics",
                              extra={"metrics": metrics})
            return metrics

        except Exception as e:
            self.logger.error(
                "Error collecting system metrics: %s", e, exc_info=True)
            return {}

    async def _store_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Store collected metrics.

        Args:
            metrics: The metrics to store
        """
        try:
            for key, value in metrics.items():
                if key != "timestamp":
                    self._metrics[key].append({
                        "timestamp": metrics["timestamp"],
                        "value": value
                    })
                    await self.metrics_collector.track_operation(
                        operation_type=f"system_{key}",
                        success=True,
                        duration=self.check_interval,
                        usage=value
                    )

            # Clean up old metrics
            cutoff_time = datetime.now() - timedelta(hours=1)
            for key in self._metrics:
                self._metrics[key] = [
                    m for m in self._metrics[key]
                    if datetime.fromisoformat(m["timestamp"]) >= cutoff_time
                ]
            self.logger.info("Stored and cleaned up metrics")
        except Exception as e:
            self.logger.error("Error storing metrics: %s", e, exc_info=True)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics summary.

        Returns:
            Summary of current metrics
        """
        try:
            current_metrics = self._collect_system_metrics()
            runtime = (datetime.now() - self.start_time).total_seconds()
            collected_metrics = self.metrics_collector.get_metrics()

            self.logger.debug("Retrieved metrics summary")
            return {
                "current": current_metrics,
                "runtime_seconds": runtime,
                "averages": self._calculate_averages(),
                "status": self._get_system_status(),
                "collected_metrics": collected_metrics,
            }
        except Exception as e:
            self.logger.error(
                "Error getting metrics summary: %s", e, exc_info=True)
            return {"error": str(e)}

    def _calculate_averages(self) -> Dict[str, float]:
        """
        Calculate average values for metrics.

        Returns:
            Dictionary of average metric values
        """
        averages = {}
        for key, values in self._metrics.items():
            if values and key in ("cpu", "memory", "disk"):
                averages[key] = sum(
                    v["value"]["percent"] for v in values
                ) / len(values)
        self.logger.debug("Calculated averages", extra={"averages": averages})
        return averages

    def _get_system_status(self) -> str:
        """
        Determine overall system status.

        Returns:
            System status ('healthy', 'warning', 'critical', or 'unknown')
        """
        try:
            current = self._collect_system_metrics()

            cpu_threshold = 90
            memory_threshold = 90
            disk_threshold = 90

            cpu_value = current.get("cpu", {}).get("percent", 0)
            memory_value = current.get("memory", {}).get("percent", 0)
            disk_value = current.get("disk", {}).get("percent", 0)

            if (cpu_value > cpu_threshold or
                memory_value > memory_threshold or
                    disk_value > disk_threshold):
                return "critical"

            if (cpu_value > cpu_threshold * 0.8 or
                memory_value > memory_threshold * 0.8 or
                    disk_value > disk_threshold * 0.8):
                return "warning"

            return "healthy"
        except Exception as e:
            self.logger.error(
                "Error getting system status: %s", e, exc_info=True)
            return "unknown"

    async def __aenter__(self) -> "SystemMonitor":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
