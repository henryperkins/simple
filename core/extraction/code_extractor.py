"""
Code Extractor Module.

This module provides functionality to extract various code elements from Python
source files using the ast module.
"""

import ast
import uuid
from typing import Any, Dict, List, Optional

from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.metrics import Metrics
from core.types.base import (
    ExtractionContext,
    ExtractionResult,
    MetricData,
)
from core.docstring_processor import DocstringProcessor
from core.extraction.function_extractor import FunctionExtractor
from core.extraction.class_extractor import ClassExtractor
from core.extraction.dependency_analyzer import DependencyAnalyzer
from core.metrics_collector import MetricsCollector
from core.types.base import Injector
from utils import (
    get_source_segment,
    handle_extraction_error,
)
from core.console import display_metrics, create_progress
from core.exceptions import ProcessingError, ExtractionError
from rich.progress import Progress


class CodeExtractor:
    """
    Extracts code elements from Python source files.
    """

    def __init__(
        self, context: ExtractionContext, correlation_id: Optional[str] = None
    ) -> None:
        """
        Initialize the CodeExtractor.

        Args:
            context: Context for extraction operations.
            correlation_id: Unique identifier for tracking operations.
        """
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = CorrelationLoggerAdapter(
            Injector.get("logger"),
            extra={"correlation_id": self.correlation_id},
        )
        self.context = context
        self.metrics_collector: MetricsCollector = Injector.get("metrics_collector")
        self.metrics: Metrics = Injector.get("metrics_calculator")
        self.docstring_processor: DocstringProcessor = Injector.get("docstring_processor")
        self.function_extractor: FunctionExtractor = Injector.get("function_extractor")
        self.class_extractor: ClassExtractor = Injector.get("class_extractor")
        self.dependency_analyzer: DependencyAnalyzer = Injector.get("dependency_analyzer")
        self.progress: Optional[Progress] = None

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

        module_name = self.context.module_name or "unnamed_module"
        module_metrics = MetricData()
        module_metrics.module_name = module_name
        try:
            # Initialize progress tracking
            self.progress = create_progress()
            task_id = self.progress.add_task(
                f"Extracting code elements from {module_name}",
                total=len(source_code.splitlines()),
            )
            self.progress.start()

            tree = ast.parse(source_code)
            self.progress.update(task_id, advance=1, description="Validating source code...")
            self._validate_source_code(source_code)

            self.progress.update(task_id, advance=1, description="Analyzing dependencies...")
            dependencies = self.dependency_analyzer.analyze_dependencies(tree)
            
            self.progress.update(task_id, advance=1, description="Extracting classes...")
            classes = await self.class_extractor.extract_classes(
                tree
            )
            module_metrics.total_classes = len(classes)
            module_metrics.scanned_classes = len(
                [cls for cls in classes if cls.docstring_info]
            )

            self.progress.update(task_id, advance=1, description="Extracting functions...")
            functions = await self.function_extractor.extract_functions(
                tree
            )
            module_metrics.total_functions = len(functions)
            module_metrics.scanned_functions = len(
                [func for func in functions if func.docstring_info]
            )

            self.progress.update(task_id, advance=1, description="Extracting variables...")
            variables = self.extract_variables(tree)

            self.progress.update(task_id, advance=1, description="Extracting constants...")
            constants = self.extract_constants(tree)

            self.progress.update(task_id, advance=1, description="Extracting module docstring...")
            module_docstring = self.extract_module_docstring(tree)

            self.progress.update(task_id, advance=1, description="Calculating metrics...")
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
            display_metrics(
                metrics_display, title=f"Code Extraction Results for {module_name}"
            )

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
                file_path=(
                    str(self.context.base_path) if self.context.base_path else ""
                ),
            )

        except ProcessingError as pe:
            handle_extraction_error(
                self.logger,
                [],
                "code_extraction",
                correlation_id=self.correlation_id,
                e=pe,
            )
            raise
        except ExtractionError as ee:
            handle_extraction_error(
                self.logger,
                [],
                "code_extraction",
                correlation_id=self.correlation_id,
                e=ee,
            )
            raise
        except Exception as e:
            handle_extraction_error(
                self.logger,
                [],
                "code_extraction",
                correlation_id=self.correlation_id,
                e=e,
            )
            raise ExtractionError(f"Unexpected error during extraction: {e}") from e
        finally:
            if self.progress:
                self.progress.stop()
                self.progress = None

    def _validate_source_code(self, source_code: str) -> None:
        """
        Validate the provided source code before processing.

        Args:
            source_code: The source code to validate.

        Raises:
            ProcessingError: If the source code contains syntax errors.
        """
        try:
            ast.parse(source_code)
        except SyntaxError as e:
            raise ProcessingError(f"Syntax error in source code: {e}")

    def extract_variables(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Extract variables from the AST.

        Args:
            tree: The AST to traverse.

        Returns:
            A list of dictionaries containing variable information.
        """
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append(
                            {
                                "name": target.id,
                                "type": "variable",
                                "value": self._get_value(node.value),
                            }
                        )
        return variables

    def extract_constants(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Extract constants from the AST.

        Args:
            tree: The AST to traverse.

        Returns:
            A list of dictionaries containing constant information.
        """
        constants = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append(
                            {
                                "name": target.id,
                                "type": "constant",
                                "value": self._get_value(node.value),
                            }
                        )
        return constants

    def _get_value(self, node: Any) -> str:
        """
        Get the value of a node as a string.

        Args:
            node: The AST node to get the value from.

        Returns:
            The value of the node.
        """
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
        else:
            return "N/A"

    def extract_module_docstring(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Extract the module-level docstring.

        Args:
            tree: The AST from which to extract the module docstring.

        Returns:
            The module docstring as a dictionary.
        """
        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            return self.docstring_processor.parse(module_docstring)
        return {}