"""
Function extraction module for Python source code analysis.

This module provides functionality to extract function definitions and related
metadata from Python source code using the Abstract Syntax Tree (AST).
"""

import ast
from typing import Optional, Union
from core.logger import CorrelationLoggerAdapter, set_correlation_id
from core.types import (
    ExtractedFunction,
    ExtractedArgument,
    ExtractionContext,
    MetricData,
)
from utils import get_source_segment, get_node_name, NodeNameVisitor, handle_extraction_error
from core.dependency_injection import Injector


class FunctionExtractor:
    """Handles extraction of functions from Python source code."""

    def __init__(
        self,
        context: ExtractionContext,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Initialize the function extractor.

        Args:
            context: The context for extraction, including settings and source code.
            correlation_id: An optional correlation ID for logging purposes.
        """
        if correlation_id:
            set_correlation_id(correlation_id)
        self.logger = CorrelationLoggerAdapter(
            Injector.get("logger")
        )
        self.context = context
        self.metrics_calculator = Injector.get("metrics_calculator")
        self.docstring_parser = Injector.get("docstring_processor")
        self.errors: list[str] = []

    def _should_process_function(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> bool:
        """Determine if a function should be processed based on context settings.

        Args:
            node: The function node to check

        Returns:
            True if the function should be processed, False otherwise
        """
        # Skip private functions if not included in settings
        if not self.context.include_private and node.name.startswith("_"):
            return False

        # Skip magic methods if not included in settings
        if (
            not self.context.include_magic
            and node.name.startswith("__")
            and node.name.endswith("__")
        ):
            return False

        # Skip nested functions if not included in settings
        if not self.context.include_nested and self.context.tree:
            for parent in ast.walk(self.context.tree):
                if isinstance(
                    parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ) and node in ast.walk(parent):
                    if parent != node:  # Don't count the node itself
                        return False

        return True

    async def extract_functions(
        self, nodes: Union[ast.AST, list[ast.AST]]
    ) -> list[ExtractedFunction]:
        """Extract function definitions from AST nodes.

        Args:
            nodes: The AST nodes to process.

        Returns:
            A list of extracted function metadata.
        """
        functions: list[ExtractedFunction] = []

        # Support for either a list of nodes or a single node
        nodes_to_process = [nodes] if isinstance(nodes, ast.AST) else nodes

        try:
            for node in nodes_to_process:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not self._should_process_function(node):
                        self.logger.debug(f"Skipping function: {node.name}")
                        continue

                    try:
                        extracted_function = await self._process_function(node)
                        if extracted_function:
                            functions.append(extracted_function)
                            # Update scan progress
                            if self.metrics_calculator.metrics_collector:
                                self.metrics_calculator.metrics_collector.update_scan_progress(
                                    self.context.module_name or "unknown",
                                    "function",
                                    node.name,
                                )
                    except Exception as e:
                        self.logger.error(
                            f"Error extracting function {node.name if hasattr(node, 'name') else 'unknown'}: {e}",
                            exc_info=True,
                            extra={
                                "function_name": (
                                    node.name if hasattr(node, "name") else "unknown"
                                )
                            },
                        )
                        self.errors.append(
                            f"Error extracting function {node.name if hasattr(node, 'name') else 'unknown'}: {e}"
                        )
                        continue

            if self.errors:
                self.logger.warning(
                    f"Encountered {len(self.errors)} errors during function extraction"
                )

            return functions

        except Exception as e:
            self.logger.error(f"Error extracting functions: {e}", exc_info=True)
            return []

    def _extract_decorators(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        """
        Extract decorator names from a function node.

        Args:
            node: The function node to process.

        Returns:
            A list of decorator names as strings.
        """
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                # Handle cases like `module.decorator`
                if hasattr(decorator.value, 'id'):
                    decorators.append(f"{decorator.value.id}.{decorator.attr}")
            elif isinstance(decorator, ast.Call):
                # Handle decorated calls like `@decorator(arg)`
                if isinstance(decorator.func, ast.Name):
                    decorators.append(decorator.func.id)
                elif isinstance(decorator.func, ast.Attribute):
                    if hasattr(decorator.func.value, 'id'):
                        decorators.append(f"{decorator.func.value.id}.{decorator.func.attr}")
        return decorators

    def _extract_arguments(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ExtractedArgument]:
        """
        Extract arguments from a function node.

        Args:
            node: The function node to process.

        Returns:
            A list of extracted arguments.
        """
        args = []
        for arg in node.args.args:
            args.append(ExtractedArgument(name=arg.arg, type=get_node_name(arg.annotation) or "Any", description=""))
        return args

    async def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[ExtractedFunction]:
        try:
            docstring = ast.get_docstring(node) or ""
            source = get_source_segment(self.context.source_code or "", node) or ""

            # Extract function components
            decorators = self._extract_decorators(node)
            args = self._extract_arguments(node)
            return_type = get_node_name(node.returns) or "Any"

            extracted_function = ExtractedFunction(
                name=node.name,
                lineno=node.lineno,
                source=source,
                docstring=docstring,
                metrics=self.metrics_calculator.calculate_metrics(source, self.context.module_name),
                dependencies=self.context.dependency_analyzer.analyze_dependencies(node),
                decorators=decorators,
                args=args,
                returns={"type": return_type, "description": ""},
                is_async=isinstance(node, ast.AsyncFunctionDef),
            )
            return extracted_function
        except Exception as e:
            handle_extraction_error(
                self.logger,
                self.errors,
                "function_extraction",
                e,
                function_name=node.name,
            )
            return None
