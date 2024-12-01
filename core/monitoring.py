"""  
Monitoring Module  
  
Provides system monitoring and performance tracking for Azure OpenAI operations.  
Focuses on essential metrics while maintaining efficiency.  
  
Usage Example:
    ```python
    from core.monitoring import SystemMonitor
    from api.token_management import TokenManager

    async def main():
        token_manager = TokenManager()
        monitor = SystemMonitor(token_manager=token_manager)
        await monitor.start()
        await asyncio.sleep(60)  # Monitor for 60 seconds
        metrics = monitor.get_metrics()
        print(metrics)
        await monitor.stop()

    import asyncio
    asyncio.run(main())
    ```

Key Classes and Functions:
- SystemMonitor: Main class for monitoring system resources and performance metrics.
- start: Start monitoring system resources.
- stop: Stop monitoring system resources.
- get_metrics: Get current metrics summary.
- _collect_system_metrics: Collect current system metrics.
- _store_metrics: Store collected metrics.
- _calculate_averages: Calculate average values for metrics.
- _get_system_status: Determine overall system status.
"""

import psutil
import asyncio  
from datetime import datetime, timedelta  
from typing import Dict, Any, Optional, List  
from collections import defaultdict  
  
from core.logger import LoggerSetup  
from api.token_management import TokenManager  
  
class SystemMonitor:  
    """Monitors system resources and performance metrics."""  
  
    def __init__(self, check_interval: int = 60,  
                 token_manager: Optional[TokenManager] = None):  
        """Initialize system monitor.
        
        Args:
            check_interval (int): Interval in seconds between metric checks.
            token_manager (Optional[TokenManager]): Optional token manager for tracking token usage.
        
        Raises:
            Exception: If initialization fails.
        """  
        self.logger = LoggerSetup.get_logger(__name__)  # Initialize logger  
        self.check_interval = check_interval  
        self.token_manager = token_manager  
        self.start_time = datetime.now()  
        self._metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  
        self._running = False  
        self._task: Optional[asyncio.Task] = None  
        self.logger.info("System monitor initialized")  
  
    async def start(self) -> None:  
        """Start monitoring system resources.
        
        Raises:
            Exception: If starting the monitor fails.
        """  
        if self._running:  
            return  
  
        self._running = True  
        self._task = asyncio.create_task(self._monitor_loop())  
        self.logger.info("System monitoring started")  
  
    async def stop(self) -> None:  
        """Stop monitoring system resources.
        
        Raises:
            Exception: If stopping the monitor fails.
        """  
        self._running = False  
        if self._task:  
            self._task.cancel()  
            try:  
                await self._task  
            except asyncio.CancelledError:  
                pass  
        self.logger.info("System monitoring stopped")  
  
    async def _monitor_loop(self) -> None:  
        """Main monitoring loop.
        
        Raises:
            Exception: If an error occurs in the monitoring loop.
        """  
        while self._running:  
            try:  
                metrics = self._collect_system_metrics()  
                self._store_metrics(metrics)  
                await asyncio.sleep(self.check_interval)  
            except Exception as e:  
                self.logger.error(f"Error in monitoring loop: {e}")  
                await asyncio.sleep(self.check_interval)  
  
    def _collect_system_metrics(self) -> Dict[str, Any]:  
        """Collect current system metrics.
        
        Returns:
            Dict[str, Any]: Collected system metrics.
        
        Raises:
            Exception: If collecting system metrics fails.
        """  
        try:  
            cpu_percent = psutil.cpu_percent(interval=1)  
            memory = psutil.virtual_memory()  
            disk = psutil.disk_usage('/')  
  
            metrics = {  
                'timestamp': datetime.now().isoformat(),  
                'cpu': {'percent': cpu_percent, 'count': psutil.cpu_count()},  
                'memory': {'total': memory.total, 'available': memory.available,  
                           'percent': memory.percent},  
                'disk': {'total': disk.total, 'used': disk.used, 'free': disk.free,  
                         'percent': disk.percent}  
            }  
  
            if self.token_manager:  
                token_stats = self.token_manager.get_usage_stats()  
                metrics['tokens'] = token_stats  
  
            return metrics  
  
        except Exception as e:  
            self.logger.error(f"Error collecting system metrics: {e}")  
            return {}  
  
    def _store_metrics(self, metrics: Dict[str, Any]) -> None:  
        """Store collected metrics.
        
        Args:
            metrics (Dict[str, Any]): Collected system metrics.
        
        Raises:
            Exception: If storing metrics fails.
        """  
        for key, value in metrics.items():  
            if key != 'timestamp':  
                self._metrics[key].append({'timestamp': metrics['timestamp'],  
                                           'value': value})  
  
        cutoff_time = datetime.now() - timedelta(hours=1)  
        for key in self._metrics:  
            self._metrics[key] = [m for m in self._metrics[key]  
                                  if datetime.fromisoformat(m['timestamp']) >= cutoff_time]  
  
    def get_metrics(self) -> Dict[str, Any]:  
        """Get current metrics summary.
        
        Returns:
            Dict[str, Any]: Current metrics summary.
        
        Raises:
            Exception: If getting metrics summary fails.
        """  
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
            self.logger.error(f"Error getting metrics summary: {e}")  
            return {'error': str(e)}  
  
    def _calculate_averages(self) -> Dict[str, float]:  
        """Calculate average values for metrics.
        
        Returns:
            Dict[str, float]: Average values for metrics.
        
        Raises:
            Exception: If calculating averages fails.
        """  
        averages = {}  
        for key, values in self._metrics.items():  
            if values and key in ('cpu', 'memory', 'disk'):  
                averages[key] = sum(v['value']['percent'] for v in values) / len(values)  
        return averages  
  
    def _get_system_status(self) -> str:  
        """Determine overall system status.
        
        Returns:
            str: Overall system status.
        
        Raises:
            Exception: If determining system status fails.
        """  
        try:  
            current = self._collect_system_metrics()  
  
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
            self.logger.error(f"Error getting system status: {e}")  
            return 'unknown'  
  
    async def __aenter__(self) -> 'SystemMonitor':  
        """Async context manager entry.
        
        Returns:
            SystemMonitor: The instance of the monitor.
        
        Raises:
            Exception: If starting the monitor fails.
        """  
        await self.start()  
        return self  
  
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  
        """Async context manager exit.
        
        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.
        
        Raises:
            Exception: If stopping the monitor fails.
        """  
        await self.stop()  
