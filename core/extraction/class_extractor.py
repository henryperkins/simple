"""
Class extraction module for Python source code analysis.

This module provides functionality to extract class definitions and related
metadata from Python source code using the Abstract Syntax Tree (AST).
"""

import ast
from typing import Any, Optional, Dict, List, Union
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.metrics import Metrics
from core.metrics_collector import MetricsCollector
from core.docstring_processor import DocstringProcessor
from core.types import (
    ExtractionContext,
    ExtractedClass,
    ExtractedFunction,
    MetricData
)
from utils import handle_extraction_error, get_source_segment, NodeNameVisitor, get_node_name
from core.types.base import Injector


class ClassExtractor:
    """Handles extraction of classes from Python source code."""

    def __init__(
        self,
        context: "ExtractionContext",
        correlation_id: Optional[str] = None
    ) -> None:
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__), correlation_id)
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
        self.errors: list[str] = []

    # ... rest of the class remains unchanged ...
