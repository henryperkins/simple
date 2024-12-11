"""
Function extraction module for Python source code analysis.

This module provides functionality to extract function definitions and related
metadata from Python source code using the Abstract Syntax Tree (AST).
"""

import ast
from typing import List, Any, Optional, Union
from core.logger import LoggerSetup, CorrelationLoggerAdapter, log_error
from core.metrics_collector import MetricsCollector
from core.docstring_processor import DocstringProcessor
from core.types import (
    ExtractedFunction,
    ExtractedArgument,
    ExtractionContext,
    MetricData,
)
from utils import get_source_segment, get_node_name, NodeNameVisitor
from core.types.base import Injector


class FunctionExtractor:
    """Handles extraction of functions from Python source code."""

    def __init__(
        self,
        context: "ExtractionContext",
        correlation_id: Optional[str] = None,
    ) -> None:
        """Initialize the function extractor.

        Args:
            context (ExtractionContext): The context for extraction, including settings and source code.
            correlation_id (Optional[str]): An optional correlation ID for logging purposes.
        """
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__), correlation_id=correlation_id)
        self.context = context
        # Get metrics calculator with fallback
        try:
            self.metrics_calculator = Injector.get('metrics_calculator')
        except KeyError:
            self.logger.warning(
                "Metrics calculator not registered, creating new instance")
            from core.metrics import Metrics
            metrics_collector = MetricsCollector(correlation_id=correlation_id)
            self.metrics_calculator = Metrics(
                metrics_collector=metrics_collector, correlation_id=correlation_id)

        # Get docstring parser with fallback
        try:
            self.docstring_parser = Injector.get('docstring_parser')
        except KeyError:
            self.logger.warning(
                "Docstring parser not registered, using default")
            self.docstring_parser = DocstringProcessor()
            Injector.register('docstring_parser', self.docstring_parser)
        self.errors: List[str] = []

    # ... rest of the class remains unchanged ...
