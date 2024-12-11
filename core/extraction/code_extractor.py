"""
Code Extractor Module.

This module provides functionality to extract various code elements from Python
source files using the ast module. It allows for the extraction of classes,
functions, variables, constants, imports, docstrings, and other relevant
information from the code.
"""

import ast
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.metrics import Metrics
from core.types import (
    ExtractionContext,
    ExtractionResult,
    ExtractedClass,
    ExtractedFunction,
    MetricData,
    DocstringData
)
from core.docstring_processor import DocstringProcessor
from core.extraction.function_extractor import FunctionExtractor
from core.extraction.class_extractor import ClassExtractor
from core.extraction.dependency_analyzer import DependencyAnalyzer
from core.metrics_collector import MetricsCollector
from core.types.base import Injector
from utils import (
    NodeNameVisitor,
    get_source_segment,
    handle_extraction_error
)
from core.console import (
    print_info,
    print_error,
    print_warning,
    display_metrics,
    create_progress,
    display_metrics
)

class CodeExtractor:
    """
    Extracts code elements from Python source files.
    """

    def __init__(self, context: Optional[ExtractionContext] = None, correlation_id: Optional[str] = None) -> None:
        """
        Initialize the CodeExtractor.

        Args:
            context: The extraction context.
            correlation_id: Optional correlation ID for logging.
        """
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__),
            {'correlation_id': self.correlation_id}
        )
        self.context = context or ExtractionContext()
        self.metrics_collector = self._get_metrics_collector()
        self.metrics = self._get_metrics_calculator()
        self.docstring_processor = DocstringProcessor()
        self.function_extractor = None
        self.class_extractor = None
        self.dependency_analyzer = None
        self.logger.info("Initializing code extractor")

    def _get_metrics_collector(self) -> MetricsCollector:
        """
        Get the metrics collector instance, with fallback if not registered.
        """
        try:
            return Injector.get('metrics_collector')
        except KeyError:
            self.logger.warning(
                "Metrics collector not registered, creating new instance"
            )
            metrics_collector = MetricsCollector(
                correlation_id=self.correlation_id)
            Injector.register('metrics_collector', metrics_collector)
            return metrics_collector

    def _get_metrics_calculator(self) -> Metrics:
        """
        Get the metrics calculator instance, with fallback if not registered.
        """
        try:
            return Injector.get('metrics_calculator')
        except KeyError:
            self.logger.warning(
                "Metrics calculator not registered, creating new instance"
            )
            metrics_calculator = Metrics(
                metrics_collector=self.metrics_collector, correlation_id=self.correlation_id)
            Injector.register('metrics_calculator', metrics_calculator)
            return metrics_calculator

    def _initialize_extractors(self) -> None:
        """Initialize the function and class extractors."""
        if self.context:
            self.function_extractor = FunctionExtractor(
                context=self.context,
                correlation_id=self.correlation_id
            )
            self.class_extractor = ClassExtractor(
                context=self.context,
                correlation_id=self.correlation_id,
                metrics_collector=self.metrics_collector,
                docstring_processor=self.docstring_processor
            )
            self.dependency_analyzer = DependencyAnalyzer(
                context=self.context,
                correlation_id=self.correlation_id
            )

    async def extract_code(self, source_code: str, context: Optional[ExtractionContext] = None) -> ExtractionResult:
        """
        Extract code elements and metadata from source code.

        Args:
            source_code: The source code to extract from.
            context: The extraction context.

        Returns:
            An ExtractionResult object containing extracted information.
        """
        self.context = context or self.context
        if not self.context:
            raise ValueError(
                "Extraction context is required for code extraction.")

        self._initialize_extractors()

        module_name = self.context.module_name or "unnamed_module"
        module_metrics = MetricData()
        module_metrics.module_name = module_name

        try:
            with create_progress() as progress:
                extraction_task = progress.add_task(
                    "Extracting code elements", total=100)

                progress.update(
                    extraction_task, advance=10, description="Parsing AST...")
                tree = ast.parse(source_code)

                progress.update(
                    extraction_task, advance=10, description="Extracting dependencies...")
                dependencies = self.dependency_analyzer.analyze_dependencies(
                    tree)

                progress.update(
                    extraction_task, advance=15, description="Extracting classes...")
                classes = await self.class_extractor.extract_classes(tree)
                for cls in classes:
                    module_metrics.scanned_classes += 1
                    module_metrics.total_classes += 1

                progress.update(
                    extraction_task, advance=15, description="Extracting functions...")
                functions = await self.function_extractor.extract_functions(tree)
                for func in functions:
                    module_metrics.scanned_functions += 1
                    module_metrics.total_functions += 1

                progress.update(
                    extraction_task, advance=20, description="Extracting variables...")
                variables = self._extract_variables(tree)

                progress.update(
                    extraction_task, advance=10, description="Extracting constants...")
                constants = self._extract_constants(tree)

                progress.update(
                    extraction_task, advance=10, description="Extracting docstrings...")
                module_docstring = self._extract_module_docstring(tree)

                progress.update(
                    extraction_task, advance=10, description="Calculating metrics...")
                module_metrics = self.metrics.calculate_metrics(
                    source_code, module_name)

                # Display extraction metrics
                metrics = {
                    "Classes": len(classes),
                    "Functions": len(functions),
                    "Variables": len(variables),
                    "Constants": len(constants),
                    "Lines of Code": len(source_code.splitlines()),
                    "Cyclomatic Complexity": module_metrics.cyclomatic_complexity,
                    "Maintainability Index": f"{module_metrics.maintainability_index:.2f}",
                    "Halstead Volume": f"{module_metrics.halstead_metrics.get('volume', 0):.2f}",
                    "Dependencies": len(dependencies)
                }
                display_metrics(
                    metrics, title=f"Code Extraction Results for {module_name}")

                return ExtractionResult(
                    module_docstring=module_docstring,
                    classes=classes,
                    functions=functions,
                    variables=variables,
                    constants=constants,
                    dependencies=dependencies,
                    metrics=module_metrics,
                    source_code=source_code,
                    module_name=module_name,
                    file_path=str(self.context.base_path) if self.context.base_path else ""
                )

        except Exception as e:
            handle_extraction_error(e, {
                "operation": "code_extraction",
                "context": str(context)
            }, self.correlation_id)
            raise

    def _extract_variables(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract variables from the AST."""
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append({
                            "name": target.id,
                            "type": "variable",
                            "value": self._get_value(node.value)
                        })
        return variables

    def _extract_constants(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract constants from the AST."""
        constants = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append({
                            "name": target.id,
                            "type": "constant",
                            "value": self._get_value(node.value)
                        })
        return constants

    def _get_value(self, node: Any) -> str:
        """Get the value of a node as a string."""
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return "N/A"

    def _extract_module_docstring(self, tree: ast.AST) -> Dict[str, Any]:
        """Extract the module-level docstring."""
        docstring_data = {}
        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            docstring_data = self.docstring_processor.parse(module_docstring)
        return docstring_data

    def _process_variable_node(self, node: ast.Assign) -> Optional[Dict[str, Any]]:
        """Process a variable node and extract its information."""
        try:
            if isinstance(node.targets[0], ast.Name):
                variable_name = node.targets[0].id
                source_segment = get_source_segment(
                    self.context.source_code, node)
                if source_segment:
                    sanitized_source = self._sanitize(source_segment)
                    return {
                        "name": variable_name,
                        "type": "variable",
                        "value": sanitized_source
                    }
        except Exception as e:
            handle_extraction_error(
                e,
                "Error processing variable node",
                self.context.module_name,
                self.correlation_id
            )
        return None

    def _process_constant_node(self, node: ast.Assign) -> Optional[Dict[str, Any]]:
        """Process a constant node and extract its information."""
        try:
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id.isupper():
                constant_name = node.targets[0].id
                source_segment = get_source_segment(
                    self.context.source_code, node)
                if source_segment:
                    sanitized_source = self._sanitize(source_segment)
                    return {
                        "name": constant_name,
                        "type": "constant",
                        "value": sanitized_source
                    }
        except Exception as e:
            handle_extraction_error(
                e,
                "Error processing constant node",
                self.context.module_name,
                self.correlation_id
            )
        return None

    def _sanitize(self, text: str) -> str:
        """Sanitize the given text to remove sensitive information."""
        # Basic example: replace paths with placeholders
        return re.sub(r'(/[\w\./]+)', '[SANITIZED_PATH]', text)
