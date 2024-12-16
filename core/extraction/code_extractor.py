"""
Code Extractor Module.

This module provides functionality to extract various code elements from Python
source files using the ast module.
"""

import ast
import uuid
import time
import os
from typing import Any, cast
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
from core.dependency_injection import Injector
from core.console import print_info, print_error, print_success
from core.exceptions import ProcessingError, ExtractionError


class CodeExtractor:
    """
    Extracts code elements from Python source files.
    """

    def __init__(
        self, context: ExtractionContext, correlation_id: str | None = None
    ) -> None:
        """
        Initialize the CodeExtractor.

        Args:
            context: Context for extraction operations.
            correlation_id: Unique identifier for tracking operations.
        """
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__),
            extra={"correlation_id": self.correlation_id},
        )
        self.context = context
        self.metrics_collector: MetricsCollector = MetricsCollector(correlation_id=self.correlation_id)
        self.metrics: Metrics = Injector.get("metrics_calculator")
        self.docstring_processor: DocstringProcessor = Injector.get("docstring_processor")
        self.function_extractor: FunctionExtractor = Injector.get("function_extractor")
        self.class_extractor: ClassExtractor = Injector.get("class_extractor")
        self.dependency_analyzer: DependencyAnalyzer = Injector.get("dependency_analyzer")
        self.logger.info(f"Initialising with dependency_analyzer: {self.context.dependency_analyzer}")
        if self.context.dependency_analyzer is None:
            self.context.dependency_analyzer = DependencyAnalyzer(self.context)
        print_info("CodeExtractor initialized.")


    async def extract_code(self, source_code: str) -> ExtractionResult:
        """
        Extract code elements and metadata from source code.

        Args:
            source_code: The source code to extract elements from.

        Returns:
            Result of the extraction process.

        Raises:
            ExtractionError: If there's an issue during the extraction process.
        """
        if not source_code or not source_code.strip():
            raise ExtractionError("Source code is empty or missing")

         # Update the existing context with new source code and validate
        self.context.set_source_code(source_code)

        
        module_name = self.context.module_name or "unnamed_module"
        module_metrics = MetricData()
        module_metrics.module_name = module_name
        start_time = time.time()

        try:
            print_info(f"Extracting code elements from {module_name}")

            tree = ast.parse(source_code)
            print_info("Validating source code...")
            file_path = str(getattr(self.context, 'base_path', '')) or "" # Use '' as default if base_path is not set
            self._validate_source_code(source_code, file_path, module_name, project_root=str(getattr(self.context, "base_path", "")))

            print_info("Analyzing dependencies...")
            dependencies = self.dependency_analyzer.analyze_dependencies(tree)
            
            print_info("Extracting classes...")
            classes = await self.class_extractor.extract_classes(tree)
            module_metrics.total_classes = len(classes)
            module_metrics.scanned_classes = len(
                [cls for cls in classes if hasattr(cls, 'docstring_info')]
            )

            print_info("Extracting functions...")
            functions = await self.function_extractor.extract_functions(tree)
            module_metrics.total_functions = len(functions)
            module_metrics.scanned_functions = len(
                [func for func in functions if hasattr(func, 'docstring_info')]
            )

            print_info("Extracting variables...")
            variables = self.extract_variables(tree)

            print_info("Extracting constants...")
            constants = self.extract_constants(tree)

            print_info("Extracting module docstring...")
            module_docstring = self.extract_module_docstring(tree)

            print_info("Calculating metrics...")
            module_metrics = self.metrics.calculate_metrics(source_code, module_name)

            # Display extraction metrics
            metrics_display = {
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
            self._display_metrics(
                metrics_display, title=f"Code Extraction Results for {module_name}"
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
            print_success(f"Code extraction completed in {processing_time:.2f}s.")

            # Convert classes and functions to dicts for ExtractionResult
            class_dicts = [cls.__dict__ for cls in classes]
            function_dicts = [func.__dict__ for func in functions]

            extraction_result = ExtractionResult(
                source_code=source_code,
                module_docstring=module_docstring,
                classes=class_dicts,
                functions=function_dicts,
                variables=variables,
                constants=constants,
                dependencies=dependencies,
                metrics=module_metrics,
                module_name=module_name,
                file_path=(
                    str(self.context.base_path) if self.context.base_path else ""
                ),
            )
            return extraction_result

        except ProcessingError as pe:
            self.logger.error(
                f"Processing error during code extraction: {pe}",
                extra={
                    "source_code_snippet": source_code[:50],
                    "module_name": module_name,
                    "file_path": str(getattr(self.context, 'base_path', ''))
                },
                exc_info=True)
            await self.metrics_collector.track_operation(
                operation_type="code_extraction",
                success=False,
                duration=time.time() - start_time,
                metadata={"error": str(pe)},
            )
            print_error(f"Code extraction failed: {pe}")
            raise
        except ExtractionError as ee:
             self.logger.error(
                f"Extraction error during code extraction: {ee}",
                extra={
                     "source_code_snippet": source_code[:50],
                    "module_name": module_name,
                    "file_path": str(getattr(self.context, 'base_path', ''))
                },
                exc_info=True
            )
             await self.metrics_collector.track_operation(
                operation_type="code_extraction",
                success=False,
                duration=time.time() - start_time,
                metadata={"error": str(ee)},
            )
             print_error(f"Code extraction failed: {ee}")
             raise
        except Exception as e:
             self.logger.error(
                f"Unexpected error during code extraction: {e}",
                extra={
                    "source_code_snippet": source_code[:50],
                    "module_name": module_name,
                    "file_path": str(getattr(self.context, 'base_path', ''))
                },
                exc_info=True
            )
             await self.metrics_collector.track_operation(
                operation_type="code_extraction",
                success=False,
                duration=time.time() - start_time,
                metadata={"error": str(e)},
            )
             print_error(f"Code extraction failed: {e}")
             raise ExtractionError(f"Unexpected error during extraction: {e}") from e

    def _validate_source_code(self, source_code: str, file_path: str, module_name: str, project_root: str) -> None:
        """
        Validate the provided source code and log issues with its representation in __init__.py.

        Args:
            source_code: The source code to validate.
            file_path: The file path of the source code being validated.
            module_name: The dotted module name (e.g., "my_project.utils").
            project_root: The root directory of the project.

        Raises:
            ProcessingError: If the source code contains syntax errors or the file path is invalid.
        """
        self.logger.info(f"Validating source code for file: {file_path}")

        # Step 1: Check file extension
        if not file_path.endswith(".py"):
            self.logger.error(f"File {file_path} is not a Python file.")
            raise ProcessingError(f"Invalid file type for {file_path}. Only Python files are supported.")

        # Step 2: Validate file contents
        try:
            with open(file_path, "r") as f:
                content = f.read()
                self.logger.info(f"First 50 chars of file contents:{content[:50]}...")
                if not content or not content.strip():
                    raise ProcessingError(f"File {file_path} is empty")
        except FileNotFoundError as e:
            self.logger.error(f"Error accessing file {file_path}: {e}")
            raise ProcessingError(f"Error accessing file {file_path}: {e}") from e
        except Exception as e:
            self.logger.error(f"Error reading file contents {file_path}: {e}")
            raise ProcessingError(f"Error reading file {file_path}: {e}") from e

        # Step 3: Validate syntax
        try:
            ast.parse(source_code)
            self.logger.info(f"Syntax validation successful for: {file_path}")
        except SyntaxError as e:
            error_details = {
                "error_message": str(e),
                "line_number": e.lineno,
                "offset": e.offset,
                "text": e.text.strip() if e.text else "N/A"
            }
            self.logger.error(f"Syntax error during validation for {file_path}: {error_details}")
            raise ProcessingError(f"Syntax error in source code: {e}") from e


        # Step 4: Check for module inclusion in __init__.py
        try:
            init_file_path = Path(project_root) / module_name.split('.')[0] / "__init__.py"
            if not init_file_path.exists():
                self.logger.warning(f"No __init__.py found at {init_file_path}. Module '{module_name}' may not be importable.")
            else:
                with open(init_file_path, "r") as init_file:
                    init_content = init_file.read()
                    if module_name not in init_content:
                        self.logger.warning(
                            f"Module '{module_name}' is not explicitly referenced in {init_file_path}. "
                            "It may not be properly importable."
                        )
                    else:
                        self.logger.info(f"Module '{module_name}' is explicitly referenced in {init_file_path}.")
        except Exception as e:
            self.logger.error(f"Error reading {init_file_path}: {e}")


    def extract_variables(self, tree: ast.AST) -> list[dict[str, Any]]:
        """
        Extract variables from the AST.

        Args:
            tree: The AST to traverse.

        Returns:
            A list of dictionaries containing variable information.
        """
        variables: list[dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append({
                            "name": target.id,
                            "type": "variable",
                            "value": self._get_value(node.value),
                        })
        return variables

    def extract_constants(self, tree: ast.AST) -> list[dict[str, Any]]:
        """
        Extract constants from the AST.

        Args:
            tree: The AST to traverse.

        Returns:
            A list of dictionaries containing constant information.
        """
        constants: list[dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append({
                            "name": target.id,
                            "type": "constant",
                            "value": self._get_value(node.value),
                        })
        return constants

    def _get_value(self, node: Any) -> str:
        """
        Get the value of a node as a string.

        Args:
            node: The AST node to get the value from.

        Returns:
            The value of the node.
        """
        try:
            if isinstance(node, ast.Constant):
                return str(node.value)
            elif isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.List):
                return f"[{', '.join(self._get_value(elt) for elt in node.elts)}]"
            elif isinstance(node, ast.Tuple):
                return f"({', '.join(self._get_value(elt) for elt in node.elts)})"
            elif isinstance(node, ast.Dict):
                return f"{{{', '.join(f'{self._get_value(k)}: {self._get_value(v)}' for k, v in zip(node.keys, node.values))}}}"
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return "-" + self._get_value(node.operand)
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                return self._get_value(node.left) + " + " + self._get_value(node.right)
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
                return self._get_value(node.left) + " - " + self._get_value(node.right)
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                return self._get_value(node.left) + " * " + self._get_value(node.right)
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                return self._get_value(node.left) + " / " + self._get_value(node.right)
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
                return self._get_value(node.left) + " % " + self._get_value(node.right)
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                return self._get_value(node.left) + " ** " + self._get_value(node.right)
            return "N/A"
        except Exception:
            return "N/A"

    def extract_module_docstring(self, tree: ast.AST) -> dict[str, Any]:
        """
        Extract the module-level docstring.

        Args:
            tree: The AST from which to extract the module docstring.

        Returns:
            The module docstring as a dictionary.
        """
        if isinstance(tree, ast.Module):
            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                return self.docstring_processor.parse(module_docstring).__dict__
        return {}

    def _display_metrics(self, metrics: dict[str, Any], title: str) -> None:
        """
        Display the extracted metrics.

        Args:
            metrics: The metrics to display.
            title: The title for the metrics display.
        """
        print_info(title)
        for metric_name, metric_value in metrics.items():
            print_info(f"{metric_name}: {metric_value}")
