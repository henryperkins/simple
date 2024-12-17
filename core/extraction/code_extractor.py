"""
Code Extractor Module.

This module provides functionality to extract various code elements from Python
source files using the ast module.
"""

import ast
import uuid
import time
from typing import Any, Dict, List
from pathlib import Path

from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.metrics import Metrics
from core.types.base import (
    ExtractionContext,
    ExtractionResult,
    MetricData,
    ExtractedClass,
    ExtractedFunction,
)
from core.docstring_processor import DocstringProcessor
from core.extraction.function_extractor import FunctionExtractor
from core.extraction.class_extractor import ClassExtractor
from core.extraction.dependency_analyzer import DependencyAnalyzer
from core.metrics_collector import MetricsCollector
from core.console import print_info
from core.exceptions import ProcessingError, ExtractionError
from core.extraction.extraction_utils import extract_attributes, extract_instance_attributes


class CodeExtractor:
    """
    Extracts code elements from Python source files.
    """

    def __init__(
        self, context: ExtractionContext, correlation_id: str | None = None
    ) -> None:
        """Initialize the CodeExtractor."""
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__),
            extra={"correlation_id": self.correlation_id},
        )
        self.context = context
        self.metrics_collector: MetricsCollector = MetricsCollector(
            correlation_id=self.correlation_id
        )
        self.metrics: Metrics = Metrics(
            metrics_collector=self.metrics_collector, correlation_id=self.correlation_id
        )
        self.docstring_processor: DocstringProcessor = DocstringProcessor()
        self.function_extractor = FunctionExtractor(context, correlation_id)
        # Assign function_extractor to the context
        self.context.function_extractor = self.function_extractor
        # Initialize ClassExtractor with the updated context
        self.class_extractor = ClassExtractor(
            context=self.context,
            correlation_id=correlation_id
        )
        self.dependency_analyzer = DependencyAnalyzer(context, correlation_id)

    async def extract_code(self, source_code: str) -> ExtractionResult:
        """Extract code elements from source code."""
        if not source_code or not source_code.strip():
            raise ExtractionError("Source code is empty or missing")

        module_name = self.context.module_name or "unnamed_module"

        start_time = time.time()

        try:
            # VALIDATE before setting the source code
            file_path = str(getattr(self.context, "base_path", "")) or ""
            self._validate_source_code(
                source_code,
                file_path,
                module_name,
                str(getattr(self.context, "base_path", "")),
            )

            # Update the context
            self.context.set_source_code(
                source_code, source="code_extractor.extract_code"
            )

            tree = ast.parse(source_code)

            dependencies = self.dependency_analyzer.analyze_dependencies(tree)

            # Calculate metrics only once at the module level
            module_metrics = self.metrics.calculate_metrics(source_code, module_name)

            # Extract classes and functions, passing the metrics
            classes: List[ExtractedClass] = await self.class_extractor.extract_classes(
                tree, module_metrics
            )
            functions: List[ExtractedFunction] = (
                await self.function_extractor.extract_functions(tree, module_metrics)
            )

            variables = self._extract_variables(tree)
            constants = self._extract_constants(tree)
            module_docstring = self._extract_module_docstring(tree)

            # Update module metrics with extraction results
            module_metrics.total_classes = len(classes)
            module_metrics.scanned_classes = len(
                [cls for cls in classes if cls.docstring_info]
            )
            module_metrics.total_functions = len(functions)
            module_metrics.scanned_functions = len(
                [func for func in functions if func.docstring_info]
            )

            self._display_metrics(
                self._get_metrics_display(
                    classes,
                    functions,
                    variables,
                    constants,
                    source_code,
                    dependencies,
                    module_metrics,
                ),
                title=f"Code Extraction Results for {module_name}",
            )

            processing_time = time.time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="code_extraction",
                success=True,
                duration=processing_time,
                metadata={
                    "classes_extracted": len(classes),
                    "functions_extracted": len(functions),
                    "variables_extracted": len(variables),
                    "constants_extracted": len(constants),
                },
            )
            self.logger.info(f"Code extraction completed in {processing_time:.2f}s.")

            # Collect metrics
            self.metrics_collector.collect_metrics(module_name, module_metrics)

            extraction_result = ExtractionResult(
                source_code=source_code,
                module_docstring=module_docstring,
                classes=classes,
                functions=functions,
                variables=variables,
                constants=constants,
                dependencies=dependencies,
                metrics=module_metrics.__dict__,
                module_name=module_name,
                file_path=file_path,
            )
            return extraction_result

        except ProcessingError as pe:
            self.logger.error(
                f"Processing error during code extraction: {pe}",
                extra={
                    "source_code_snippet": source_code[:50],
                    "module_name": module_name,
                    "file_path": file_path,
                },
                exc_info=True,
            )
            await self.metrics_collector.track_operation(
                operation_type="code_extraction",
                success=False,
                duration=time.time() - start_time,
                metadata={"error": str(pe)},
            )
            raise
        except ExtractionError as ee:
            self.logger.error(
                f"Extraction error during code extraction: {ee}",
                extra={
                    "source_code_snippet": source_code[:50],
                    "module_name": module_name,
                    "file_path": file_path,
                },
                exc_info=True,
            )
            await self.metrics_collector.track_operation(
                operation_type="code_extraction",
                success=False,
                duration=time.time() - start_time,
                metadata={"error": str(ee)},
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Unexpected error during code extraction: {e}",
                extra={
                    "source_code_snippet": source_code[:50],
                    "module_name": module_name,
                    "file_path": file_path,
                },
                exc_info=True,
            )
            await self.metrics_collector.track_operation(
                operation_type="code_extraction",
                success=False,
                duration=time.time() - start_time,
                metadata={"error": str(e)},
            )
            raise ExtractionError(f"Unexpected error during extraction: {e}") from e

    def _validate_source_code(
        self, source_code: str, file_path: str, module_name: str, project_root: str
    ) -> None:
        """Validate source code."""
        self.logger.info(f"Validating source code for file: {file_path}")
        try:
            ast.parse(source_code)
            self.logger.info(f"Syntax validation successful for: {file_path}")
        except SyntaxError as e:
            error_details = {
                "error_message": str(e),
                "line_number": e.lineno,
                "offset": e.offset,
                "text": e.text.strip() if e.text else "N/A",
            }
            self.logger.error(
                f"Syntax error during validation for {file_path}: {error_details}"
            )
            raise ProcessingError(f"Syntax error in source code: {e}") from e

        # Step 2: Check for module inclusion in __init__.py
        try:
            init_file_path = (
                Path(project_root) / module_name.split(".")[0] / "__init__.py"
            )
            if not init_file_path.exists():
                self.logger.warning(
                    f"No __init__.py found at {init_file_path}. Module '{module_name}' may not be importable."
                )
            else:
                with open(init_file_path, "r") as init_file:
                    init_content = init_file.read()
                    if module_name not in init_content:
                        self.logger.warning(
                            f"Module '{module_name}' is not explicitly referenced in {init_file_path}. "
                            "It may not be properly importable."
                        )
                    else:
                        self.logger.info(
                            f"Module '{module_name}' is explicitly referenced in {init_file_path}."
                        )
        except Exception as e:
            self.logger.error(f"Error reading {init_file_path}: {e}")

    def _extract_variables(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract variables from the AST."""
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append(
                            {
                                "name": target.id,
                                "type": "variable",
                                "value": ast.unparse(node.value),
                                "lineno": node.lineno,
                            }
                        )
        return variables

    def _extract_constants(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract constants from the AST."""
        constants = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append(
                            {
                                "name": target.id,
                                "type": "constant",
                                "value": ast.unparse(node.value),
                                "lineno": node.lineno,
                            }
                        )
        return constants

    def _extract_module_docstring(self, tree: ast.AST) -> Dict[str, Any]:
        """Extract module docstring."""
        if isinstance(tree, ast.Module):
            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                return self.docstring_processor.parse(module_docstring).__dict__
        return {}

    def _display_metrics(self, metrics: Dict[str, Any], title: str) -> None:
        """Display extracted metrics."""
        print_info(title)
        for metric_name, metric_value in metrics.items():
            print_info(f"{metric_name}: {metric_value}")

    def _get_metrics_display(
        self,
        classes: List[ExtractedClass],
        functions: List[ExtractedFunction],
        variables: List[Dict[str, Any]],
        constants: List[Dict[str, Any]],
        source_code: str,
        dependencies: Dict[str, set[str]],
        module_metrics: MetricData,
    ) -> Dict[str, Any]:
        """Prepare metrics for display."""
        return {
            "Classes": len(classes),
            "Functions": len(functions),
            "Variables": len(variables),
            "Constants": len(constants),
            "Lines of Code": len(source_code.splitlines()),
            "Cyclomatic Complexity": module_metrics.cyclomatic_complexity,
            "Maintainability Index": f"{module_metrics.maintainability_index:.2f}",
            "Halstead Volume": f"{module_metrics.halstead_metrics.get('volume', 0):.2f}",
            "Dependencies": len(dependencies),
        }
