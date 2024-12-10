"""
Code extraction module for Python source code analysis.
"""

import ast
import uuid
import re
from typing import Any, Optional

from core.logger import LoggerSetup, CorrelationLoggerAdapter, log_info, log_error
from core.metrics import Metrics
from core.types import (
    ExtractionContext,
    ExtractionResult,
    DocstringData,
    MetricData
)
from core.types.base import Injector
from core.extraction.function_extractor import FunctionExtractor
from core.extraction.class_extractor import ClassExtractor
from core.extraction.dependency_analyzer import DependencyAnalyzer
# Since utils.py is in the project root, we need to use an absolute import
from utils import NodeNameVisitor, get_source_segment

class CodeExtractor:
    """Extracts code elements and metadata from Python source code."""

    def __init__(
        self,
        context: Optional[ExtractionContext] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        # Generate correlation ID if not provided
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = CorrelationLoggerAdapter(LoggerSetup.get_logger(__name__), correlation_id=self.correlation_id)

        self.context = context or ExtractionContext()
        self.metrics_calculator = Injector.get('metrics_calculator')
        
        # Initialize extractors
        self._initialize_extractors()

    def _initialize_extractors(self) -> None:
        """Initialize the extractors with the current context."""
        self.context.function_extractor = FunctionExtractor(self.context, self.metrics_calculator)
        self.context.class_extractor = ClassExtractor(self.context, self.metrics_calculator)
        self.context.dependency_analyzer = DependencyAnalyzer(self.context)

    def _count_code_elements(self, tree: ast.AST) -> tuple[int, int]:
        """Count total functions and classes in the AST.
        
        Args:
            tree: The AST to analyze
            
        Returns:
            tuple[int, int]: Total number of functions and classes
        """
        total_functions = 0
        total_classes = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if self.context.class_extractor._should_process_class(node):
                    total_classes += 1
                    # Count methods within classes
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if self.context.function_extractor._should_process_function(child):
                                total_functions += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only count top-level functions here
                if (self.context.function_extractor._should_process_function(node) and
                    not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree) 
                           if node in ast.walk(parent))):
                    total_functions += 1
                            
        return total_functions, total_classes

    async def extract_code(self, source_code: str, context: Optional[ExtractionContext] = None) -> ExtractionResult:
        """Extract all code elements and metadata."""
        if context:
            self.context = context
            # Re-initialize extractors with new context
            self._initialize_extractors()
            
        self.context.source_code = source_code

        try:
            tree = ast.parse(source_code)
            self.context.tree = tree

            log_info(
                "Starting code extraction",
                extra={'file_path': str(self.context.base_path or ""), 'module_name': self.context.module_name or ""}
            )

            # Count total functions and classes before extraction
            total_functions, total_classes = self._count_code_elements(tree)

            # Extract module docstring
            docstring_info = self._extract_module_docstring(tree)

            # Calculate maintainability using the public method with source_code
            maintainability_index = self.metrics_calculator.calculate_maintainability_index(source_code)

            # Initialize metrics with total counts
            metrics = MetricData(
                cyclomatic_complexity=0,
                cognitive_complexity=0,
                maintainability_index=maintainability_index,
                halstead_metrics={},
                lines_of_code=len(source_code.splitlines()),
                complexity_graph=None,
                total_functions=total_functions,
                scanned_functions=0,
                total_classes=total_classes,
                scanned_classes=0
            )

            # Extract all elements
            extracted_classes = await self.context.class_extractor.extract_classes(tree)
            extracted_functions = await self.context.function_extractor.extract_functions(tree)

            # Update scanned counts
            metrics.scanned_classes = len(extracted_classes)
            metrics.scanned_functions = len(extracted_functions)
            
            # Count methods from extracted classes
            for class_info in extracted_classes:
                metrics.scanned_functions += len(class_info.methods)

            result = ExtractionResult(
                module_docstring=docstring_info.__dict__,
                module_name=self.context.module_name or "",
                file_path=str(self.context.base_path or ""),
                classes=extracted_classes,
                functions=extracted_functions,
                variables=self._extract_variables(tree),
                constants=self._extract_constants(tree),
                dependencies=self.context.dependency_analyzer.analyze_dependencies(tree),
                errors=[],
                maintainability_index=maintainability_index,
                source_code=source_code,
                imports=[],
                metrics=metrics
            )

            # Log extraction statistics
            log_info(
                f"Code extraction completed. Functions: {metrics.scanned_functions}/{metrics.total_functions}, "
                f"Classes: {metrics.scanned_classes}/{metrics.total_classes}",
                extra={'correlation_id': self.correlation_id}
            )
            return result

        except Exception as e:
            log_error(f"Error during code extraction: {e}", exc_info=True, extra={'source_code': self._sanitize(source_code)})
            raise

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

    def _sanitize(self, text: str) -> str:
        """Sanitize text to remove sensitive information."""
        return re.sub(r'(/[a-zA-Z0-9_\-./]+)', '[SANITIZED_PATH]', text)
