from dataclasses import dataclass, field, fields
from typing import List, Dict, Any, Optional
import time
import json
from datetime import datetime
from core.logger import log_info, log_error, log_debug

@dataclass
class DocstringChanges:
    """Data structure to track changes in function docstrings."""
    added: List[Dict[str, Any]] = field(default_factory=list)
    updated: List[Dict[str, Any]] = field(default_factory=list)
    retained: List[Dict[str, Any]] = field(default_factory=list)
    failed: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class CurrentBatch:
    """Data structure for tracking current batch processing metrics."""
    total_tokens: int = 0
    total_time: float = 0.0
    processed: int = 0
    failed: int = 0

@dataclass
class ModelMetrics:
    """Metrics structure for model operations."""
    model_type: str
    operation: str
    tokens_used: int
    response_time: float
    status: str
    cost: float
    error: Optional[str] = None

@dataclass
class APIMetrics:
    """Metrics data structure for API operations."""
    endpoint: str
    tokens: int
    response_time: float
    status: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class BatchMetrics:
    """Metrics data structure for batch operations."""
    total_functions: int
    successful: int
    failed: int
    total_tokens: int
    total_time: float
    average_time_per_function: float

class SystemMonitor:
    """
    System monitoring and metrics tracking.

    This class provides methods to log various system events, such as API requests,
    cache hits and misses, docstring changes, and model-specific metrics.
    """

    def __init__(self):
        """Initialize the monitoring system with default metrics and start time."""
        self.requests = []
        self.metrics = {}
        self.api_metrics = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.docstring_changes = DocstringChanges()
        self.current_batch = CurrentBatch()
        self.metrics_by_model = {
            "azure": {},
            "openai": {},
            "gemini": {},
            "claude": {}
        }
        self.start_time = time.time()

    def log_api_request(self, api_metrics: APIMetrics) -> None:
        """
        Log an API request with detailed metrics.

        Args:
            api_metrics (APIMetrics): Metrics data for the API request.
        """
        log_debug(f"Logging API request to endpoint: {api_metrics.endpoint}")
        self.api_metrics.append(api_metrics)
        log_info(
            f"API Request logged: {api_metrics.endpoint} - "
            f"Status: {api_metrics.status}, Tokens: {api_metrics.tokens}, "
            f"Cost: {api_metrics.estimated_cost}"
        )

    def log_cache_hit(self, function_name: str) -> None:
        """
        Log a cache hit event.

        Args:
            function_name (str): Name of the function for which the cache was hit.
        """
        self.cache_hits += 1
        log_info(f"Cache hit for function: {function_name}")

    def log_cache_miss(self, function_name: str) -> None:
        """
        Log a cache miss event.

        Args:
            function_name (str): Name of the function for which the cache was missed.
        """
        self.cache_misses += 1
        log_info(f"Cache miss for function: {function_name}")

    def log_docstring_changes(self, action: str, function_name: str) -> None:
        """
        Log changes to function docstrings.

        Args:
            action (str): The action performed on the docstring (e.g., 'added', 'updated').
            function_name (str): Name of the function whose docstring was changed.
        """
        log_debug(f"Logging docstring change: {action} for function: {function_name}")
        # Use fields() to get the list of valid actions
        valid_actions = {f.name for f in fields(DocstringChanges)}
        if action in valid_actions:
            getattr(self.docstring_changes, action).append({
                'function': function_name,
                'timestamp': datetime.now().isoformat()
            })
            log_info(f"Docstring {action} for function: {function_name}")
        else:
            log_error(f"Unknown docstring action: {action}")

    def log_operation_complete(self, function_name: str, execution_time: float, tokens_used: int) -> None:
        """
        Log completion of a function processing operation.

        Args:
            function_name (str): Name of the function processed.
            execution_time (float): Time taken to process the function.
            tokens_used (int): Number of tokens used in the operation.
        """
        log_debug(f"Logging operation completion for function: {function_name}")
        self.current_batch.total_tokens += tokens_used
        self.current_batch.total_time += execution_time
        self.current_batch.processed += 1
        log_info(f"Operation complete for function: {function_name}")

    def log_batch_completion(self, total_functions: int) -> BatchMetrics:
        """
        Log completion of a batch processing operation.

        Args:
            total_functions (int): Total number of functions in the batch.

        Returns:
            BatchMetrics: Metrics for the completed batch.
        """
        log_debug("Logging batch completion")
        metrics = BatchMetrics(
            total_functions=total_functions,
            successful=self.current_batch.processed,
            failed=self.current_batch.failed,
            total_tokens=self.current_batch.total_tokens,
            total_time=self.current_batch.total_time,
            average_time_per_function=self.current_batch.total_time / max(self.current_batch.processed, 1)
        )
        
        # Reset batch metrics
        self.current_batch = CurrentBatch()
        log_info(f"Batch processing completed: {metrics}")
        return metrics

    def log_model_metrics(self, model_type: str, metrics: ModelMetrics) -> None:
        """
        Log metrics for specific model.

        Args:
            model_type (str): Type of the AI model (azure/openai/gemini/claude)
            metrics (ModelMetrics): Metrics data for the model operation

        Raises:
            ValueError: If the model type is not supported
        """
        log_debug(f"Logging metrics for model type: {model_type}")
        if model_type not in self.metrics_by_model:
            error_msg = f"Unsupported model type: {model_type}"
            log_error(error_msg)
            raise ValueError(error_msg)
            
        self.metrics_by_model[model_type][metrics.operation] = metrics
        log_info(f"Model metrics logged: {model_type} - Operation: {metrics.operation}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive metrics summary.

        Returns:
            Dict[str, Any]: Complete metrics summary including runtime, API metrics,
                            cache metrics, docstring changes, and model-specific metrics.
        """
        log_debug("Generating metrics summary")
        runtime = time.time() - self.start_time
        total_requests = len(self.api_metrics)
        failed_requests = len([m for m in self.api_metrics if m.error])

        # Initialize 'model_metrics' as an empty dictionary
        summary = {
            'runtime_seconds': runtime,
            'api_metrics': {
                'total_requests': total_requests,
                'failed_requests': failed_requests,
                'error_rate': failed_requests / max(total_requests, 1),
                'average_response_time': sum(m.response_time for m in self.api_metrics) / max(total_requests, 1),
                'total_tokens_used': sum(m.tokens for m in self.api_metrics)
            },
            'cache_metrics': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            },
            'docstring_changes': {
                field.name: len(getattr(self.docstring_changes, field.name))
                for field in fields(DocstringChanges)
            },
            'model_metrics': {}  # Initialize as an empty dictionary
        }

        # Add model-specific metrics to the summary
        for model_type, operations in self.metrics_by_model.items():
            # Ensure 'model_metrics' is a dictionary
            assert isinstance(summary['model_metrics'], dict)
            summary['model_metrics'][model_type] = {
                'total_operations': len(operations),
                'total_tokens': sum(m.tokens_used for m in operations.values()),
                'total_cost': sum(m.cost for m in operations.values()),
                'average_response_time': sum(m.response_time for m in operations.values()) / max(len(operations), 1),
                'error_count': len([m for m in operations.values() if m.error])
            }

        log_info(f"Metrics summary generated: {summary}")
        return summary

    def export_metrics(self, filepath: str) -> None:
        """
        Export metrics to a JSON file.

        Args:
            filepath (str): Path to save the metrics file.
        """
        log_debug(f"Exporting metrics to file: {filepath}")
        try:
            metrics = self.get_metrics_summary()
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            log_info(f"Metrics exported to: {filepath}")
        except Exception as e:
            log_error(f"Failed to export metrics: {str(e)}")
