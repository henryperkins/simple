"""
Code extraction module for Python source code analysis.
"""

import ast
from typing import Any, Optional

from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.metrics import Metrics
from core.types import (
    ExtractionContext,
    ExtractionResult,
    DocstringData,
    MetricData
)
from core.extraction.function_extractor import FunctionExtractor
from core.extraction.class_extractor import ClassExtractor
from core.extraction.dependency_analyzer import DependencyAnalyzer
from utils import (
    get_source_segment,
    NodeNameVisitor
)
import uuid

class CodeExtractor:
    """Extracts code elements and metadata from Python source code."""

    def __init__(
        self,
        context: Optional[ExtractionContext] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        # Initialize logger with a correlation ID
        base_logger = LoggerSetup.get_logger(__name__)
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = CorrelationLoggerAdapter(base_logger, correlation_id=self.correlation_id)

        self.context = context or ExtractionContext()
        self.metrics_calculator = Metrics()

        # Initialize extractors
        if not self.context.function_extractor:
            self.context.function_extractor = FunctionExtractor(self.context, self.metrics_calculator)
        if not self.context.class_extractor:
            self.context.class_extractor = ClassExtractor(self.context, self.metrics_calculator)
        if not self.context.dependency_analyzer:
            self.context.dependency_analyzer = DependencyAnalyzer(self.context)

    async def extract_code(self, source_code: str, context: Optional[ExtractionContext] = None) -> ExtractionResult:
        """Extract all code elements and metadata."""
        self.logger.info("Starting code extraction", extra={'correlation_id': self.correlation_id})

        if context:
            self.context = context
        self.context.source_code = source_code

        try:
            tree = ast.parse(source_code)
            self.context.tree = tree

            # Extract module docstring
            docstring_info = self._extract_module_docstring(tree)

            # Calculate maintainability
            maintainability_index = self.metrics_calculator.calculate_maintainability_index(tree)

            # Extract all elements
            result = ExtractionResult(
                module_docstring=docstring_info.__dict__,
                module_name=self.context.module_name or "",
                file_path=str(self.context.base_path or ""),
                classes=await self.context.class_extractor.extract_classes(tree),
                functions=await self.context.function_extractor.extract_functions(tree),
                variables=self._extract_variables(tree),
                constants=self._extract_constants(tree),
                dependencies=self.context.dependency_analyzer.analyze_dependencies(tree),
                errors=[],
                maintainability_index=maintainability_index,
                source_code=source_code,
                imports=[],
                metrics=MetricData(
                    cyclomatic_complexity=0,
                    cognitive_complexity=0,
                    maintainability_index=maintainability_index,
                    halstead_metrics={},
                    lines_of_code=0,
                    complexity_graph=None
                )
            )

            self.logger.info("Code extraction completed successfully", extra={'correlation_id': self.correlation_id})
            return result

        except Exception as e:
            self.logger.error(f"Error during code extraction: {e}", exc_info=True, extra={'correlation_id': self.correlation_id})
            return ExtractionResult(
                module_docstring={},
                module_name="",
                file_path="",
                classes=[],
                functions=[],
                variables=[],
                constants=[],
                dependencies={},
                errors=[str(e)],
                maintainability_index=None,
                source_code=source_code,
                imports=[],
                metrics=MetricData(
                    cyclomatic_complexity=0,
                    cognitive_complexity=0,
                    maintainability_index=0,
                    halstead_metrics={},
                    lines_of_code=0,
                    complexity_graph=None
                )
            )

    def _extract_variables(self, tree: ast.AST) -> list[dict[str, Any]]:
        """Extract variables using NodeNameVisitor."""
        variables: list[dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                visitor = NodeNameVisitor()
                if isinstance(node, ast.AnnAssign) and node.annotation:
                    visitor.visit(node.annotation)
                var_info = self._process_variable_node(node, visitor)
                if var_info:
                    variables.append(var_info)
        return variables

    def _extract_constants(self, tree: ast.AST) -> list[dict[str, Any]]:
        """Extract constants (uppercase variables)."""
        constants: list[dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append(self._process_constant_node(target, node))
        return constants

    def _extract_module_docstring(self, tree: ast.Module) -> DocstringData:
        """Extract module-level docstring."""
        docstring = ast.get_docstring(tree) or ""
        return DocstringData(
            summary=docstring.split("\n\n")[0] if docstring else "",
            description=docstring,
            args=[],
            returns={"type": "None", "description": ""},
            raises=[],
            complexity=1
        )

    def _process_variable_node(self, node: ast.AST, visitor: NodeNameVisitor) -> Optional[dict[str, Any]]:
        """Process variable node to extract information."""
        try:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        return {
                            "name": target.id,
                            "type": visitor.name or "Any",
                            "value": get_source_segment(self.context.source_code or "", node.value)
                        }
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                return {
                    "name": node.target.id,
                    "type": visitor.name or "Any",
                    "value": get_source_segment(self.context.source_code or "", node.value) if node.value else None
                }
            return None
        except Exception as e:
            self.logger.error(f"Error processing variable node: {e}", extra={'correlation_id': self.correlation_id})
            return None

    def _process_constant_node(self, target: ast.Name, node: ast.Assign) -> dict[str, Any]:
        """Process constant node to extract information."""
        return {
            "name": target.id,
            "value": get_source_segment(self.context.source_code or "", node.value)
        }
