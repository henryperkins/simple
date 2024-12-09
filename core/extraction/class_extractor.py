"""
Class extraction module for Python source code analysis.

This module provides functionality to extract class definitions and related
metadata from Python source code using the Abstract Syntax Tree (AST).
"""

import ast
from typing import Any, Optional, Dict
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.metrics import Metrics
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
        context: ExtractionContext,
        correlation_id: Optional[str] = None
    ) -> None:
        """Initialize the class extractor.

        Args:
            context (ExtractionContext): The context for extraction, including settings and source code.
            correlation_id (Optional[str]): An optional correlation ID for logging purposes.
        """
        self.logger = CorrelationLoggerAdapter(LoggerSetup.get_logger(__name__), correlation_id)
        self.context = context
        self.metrics_calculator = Injector.get('metrics_calculator')
        self.docstring_parser = Injector.get('docstring_parser')
        self.errors: list[str] = []

    async def extract_classes(self, tree: ast.AST) -> list[ExtractedClass]:
        """Extract class definitions from AST nodes.

        Args:
            tree (ast.AST): The AST tree to process.

        Returns:
            list[ExtractedClass]: A list of extracted class metadata.
        """
        classes: list[ExtractedClass] = []
        try:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if not self._should_process_class(node):
                        self.logger.debug(f"Skipping class: {node.name}")
                        continue

                    try:
                        extracted_class = await self._process_class(node)
                        if extracted_class:
                            classes.append(extracted_class)
                            # Update scan progress
                            if self.metrics_calculator.metrics_collector:
                                self.metrics_calculator.metrics_collector.update_scan_progress(
                                    self.context.module_name or "unknown",
                                    "class",
                                    node.name
                                )
                    except Exception as e:
                        handle_extraction_error(
                            self.logger,
                            self.errors,
                            f"Class {node.name}",
                            e,
                            extra={'class_name': node.name}
                        )

            return classes
        except Exception as e:
            self.logger.error(f"Error extracting classes: {e}", exc_info=True)
            return []

    async def _process_class(self, node: ast.ClassDef) -> Optional[ExtractedClass]:
        """Process a class node to extract information.

        Args:
            node (ast.ClassDef): The class node to process.

        Returns:
            Optional[ExtractedClass]: The extracted class metadata, or None if processing fails.
        """
        try:
            # Extract basic information
            docstring = ast.get_docstring(node) or ""
            source = get_source_segment(self.context.source_code or "", node) or ""

            # Calculate metrics
            metrics = MetricData()
            extracted_class = ExtractedClass(
                name=node.name,
                lineno=node.lineno,
                source=source,
                docstring=docstring,
                metrics=metrics,
                dependencies=self.context.dependency_analyzer.analyze_dependencies(node),
                decorators=self._extract_decorators(node),
                complexity_warnings=[],  # Populate as needed
                ast_node=node,
                methods=await self._extract_methods(node),
                attributes=self._extract_attributes(node),
                instance_attributes=self._extract_instance_attributes(node),
                bases=self._extract_bases(node),
                metaclass=self._extract_metaclass(node),
                is_exception=self._is_exception_class(node),
                docstring_info=self.docstring_parser(docstring)  # Use injected docstring parser
            )

            # Calculate and assign metrics
            extracted_class.metrics = self.metrics_calculator.calculate_metrics_for_class(extracted_class)

            return extracted_class

        except Exception as e:
            self.logger.error(
                f"Failed to process class {node.name}: {e}",
                exc_info=True,
                extra={'class_name': node.name}
            )
            return None

    # ... rest of the methods remain unchanged ...
