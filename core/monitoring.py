"""
Simplified system monitoring for tracking operations and performance.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import time
from core.logger import log_info, log_error, log_debug

@dataclass
class MetricsData:
    """Basic metrics data structure."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_tokens: int = 0
    total_time: float = 0
    cache_hits: int = 0
    cache_misses: int = 0
    api_requests: int = 0
    api_errors: int = 0
    start_time: float = field(default_factory=time.time)

class SystemMonitor:
    """Simple system monitoring and metrics tracking."""

    def __init__(self):
        """Initialize monitoring system."""
        self.metrics = MetricsData()
        log_info("System monitor initialized")

    def log_operation_complete(
        self,
        operation_name: str,
        execution_time: float,
        tokens_used: int,
        error: Optional[str] = None
    ) -> None:
        """Log completion of an operation."""
        self.metrics.total_operations += 1
        self.metrics.total_time += execution_time
        self.metrics.total_tokens += tokens_used
        
        if error:
            self.metrics.failed_operations += 1
            log_error(f"Operation failed: {operation_name} - {error}")
        else:
            self.metrics.successful_operations += 1
            log_info(f"Operation complete: {operation_name}")

    def log_api_request(self, success: bool = True) -> None:
        """Log an API request."""
        self.metrics.api_requests += 1
        if not success:
            self.metrics.api_errors += 1

    def log_cache_hit(self, key: str) -> None:
        """Log a cache hit."""
        self.metrics.cache_hits += 1
        log_debug(f"Cache hit: {key}")

    def log_cache_miss(self, key: str) -> None:
        """Log a cache miss."""
        self.metrics.cache_misses += 1
        log_debug(f"Cache miss: {key}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        runtime = time.time() - self.metrics.start_time
        success_rate = (self.metrics.successful_operations / 
                       max(self.metrics.total_operations, 1)) * 100
        cache_hit_rate = (self.metrics.cache_hits / 
                         max(self.metrics.cache_hits + self.metrics.cache_misses, 1)) * 100
        api_success_rate = ((self.metrics.api_requests - self.metrics.api_errors) / 
                           max(self.metrics.api_requests, 1)) * 100

        return {
            "runtime_seconds": round(runtime, 2),
            "total_operations": self.metrics.total_operations,
            "success_rate": round(success_rate, 2),
            "average_operation_time": round(
                self.metrics.total_time / max(self.metrics.total_operations, 1), 
                3
            ),
            "total_tokens": self.metrics.total_tokens,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "api_success_rate": round(api_success_rate, 2)
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = MetricsData()
        log_info("Metrics reset")