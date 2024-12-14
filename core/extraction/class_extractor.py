"""
Class extraction module for Python source code analysis.

This module provides functionality to extract class definitions and related
metadata from Python source code using the Abstract Syntax Tree (AST).
"""

import ast
import uuid
from typing import Any, TypeVar, cast

from core.logger import CorrelationLoggerAdapter
from core.types import ExtractionContext, ExtractedClass, ExtractedFunction, MetricData
from utils import (
    get_source_segment,
    handle_extraction_error,
    get_node_name,
)
from core.types.base import Injector


T = TypeVar('T')


class ClassExtractor:
    """Handles extraction of classes from Python source code."""

    def __init__(
        self,
        context: ExtractionContext,
        correlation_id: str | None = None,
    ) -> None:
        """Initialize the ClassExtractor.

        Args:
            context: The extraction context containing necessary information.
            correlation_id: Optional correlation ID for logging.
        """
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = CorrelationLoggerAdapter(
            Injector.get("logger"),
            extra={"correlation_id": self.correlation_id},
        )
        self.context = context
        self.metrics_collector = Injector.get("metrics_collector")
        self.metrics_calculator = Injector.get("metrics_calculator")
        self.docstring_parser = Injector.get("docstring_processor")
        self.function_extractor = Injector.get("function_extractor")
        self.errors: list[str] = []

    async def extract_classes(
        self, tree: ast.AST | ast.Module
    ) -> list[ExtractedClass]:
        """Extract class definitions from AST nodes.

        Args:
            tree: The AST tree to process.

        Returns:
            A list of extracted class metadata.
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
                            self.metrics_collector.update_scan_progress(
                                self.context.module_name or "unknown",
                                "class",
                                node.name,
                            )
                    except Exception as e:
                        handle_extraction_error(
                            self.logger,
                            self.errors,
                            f"Class {node.name}",
                            e,
                            extra={"class_name": node.name},
                        )

            return classes
        except Exception as e:
            self.logger.error(
                f"Error extracting classes: {e} with correlation ID: {self.correlation_id}",
                exc_info=True,
            )
            return []

    def _should_process_class(self, node: ast.ClassDef) -> bool:
        """Determine if a class should be processed based on context settings."""
        # Skip private classes if not included in settings
        if not self.context.include_private and node.name.startswith("_"):
            return False

        # Skip nested classes if not included in settings
        if not self.context.include_nested and hasattr(self.context, "tree") and self.context.tree is not None:
            tree_node = cast(ast.AST, self.context.tree)
            for parent in ast.walk(tree_node):
                if isinstance(parent, ast.ClassDef) and node in ast.walk(parent):
                    if parent != node:  # Don't count the node itself
                        return False

        return True

    def _extract_decorators(self, node: ast.ClassDef) -> list[str]:
        """Extract decorator names from a class node."""
        decorators: list[str] = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorators.append(decorator.func.id)
                elif isinstance(decorator.func, ast.Attribute):
                    if hasattr(decorator.func.value, 'id'):
                        decorators.append(
                            f"{decorator.func.value.id}.{decorator.func.attr}"
                        )
            elif isinstance(decorator, ast.Attribute):
                if hasattr(decorator.value, 'id'):
                    decorators.append(f"{decorator.value.id}.{decorator.attr}")
        return decorators

    async def _extract_methods(self, node: ast.ClassDef) -> list[ExtractedFunction]:
        """Extract method definitions from a class node."""
        methods: list[ExtractedFunction] = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not self.function_extractor._should_process_function(child):
                    self.logger.debug(f"Skipping method: {child.name}")
                    continue

                try:
                    extracted_method = await self.function_extractor._process_function(
                        child
                    )
                    if extracted_method:
                        extracted_method.is_method = True
                        extracted_method.parent_class = node.name
                        methods.append(extracted_method)
                except Exception as e:
                    self.logger.error(
                        f"Failed to process method {child.name}: {e} with correlation ID: {self.correlation_id}",
                        exc_info=True,
                        extra={"method_name": child.name},
                    )
        return methods

    def _extract_attributes(self, node: ast.ClassDef) -> list[dict[str, Any]]:
        """Extract class-level attributes from a class node."""
        attributes: list[dict[str, Any]] = []
        source_code = getattr(self.context, "source_code", "") or ""
        
        for child in node.body:
            try:
                if isinstance(child, ast.AnnAssign) and isinstance(
                    child.target, ast.Name
                ):
                    attr_value = None
                    if child.value:
                        attr_value = get_source_segment(source_code, child.value)

                    attributes.append(
                        {
                            "name": child.target.id,
                            "type": get_node_name(child.annotation),
                            "value": attr_value,
                        }
                    )
                elif isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            attr_value = get_source_segment(source_code, child.value)
                            attributes.append(
                                {
                                    "name": target.id,
                                    "type": "Any",
                                    "value": attr_value,
                                }
                            )
            except Exception as e:
                handle_extraction_error(
                    self.logger,
                    self.errors,
                    f"Class {node.name}",
                    e,
                    extra={"attribute_name": getattr(child, "name", "unknown")},
                )
                continue

        return attributes

    def _extract_bases(self, node: ast.ClassDef) -> list[str]:
        """Extract base class names from a class definition."""
        bases: list[str] = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                if hasattr(base.value, 'id'):
                    bases.append(f"{base.value.id}.{base.attr}")
        return bases

    def _extract_metaclass(self, node: ast.ClassDef) -> str | None:
        """Extract metaclass name from class keywords if present."""
        for keyword in node.keywords:
            if keyword.arg == "metaclass":
                if isinstance(keyword.value, ast.Name):
                    return keyword.value.id
                elif isinstance(keyword.value, ast.Attribute):
                    if hasattr(keyword.value.value, 'id'):
                        return f"{keyword.value.value.id}.{keyword.value.attr}"
        return None

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """Check if the class is an exception class."""
        exception_bases = {"Exception", "BaseException"}
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in exception_bases:
                return True
        return False

    def _extract_instance_attributes(self, node: ast.ClassDef) -> list[dict[str, Any]]:
        """Extract instance attributes from a class node."""
        instance_attributes: list[dict[str, Any]] = []
        source_code = getattr(self.context, "source_code", "") or ""
        
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        instance_attributes.append(
                            {
                                "name": target.attr,
                                "type": "Any",
                                "value": get_source_segment(source_code, child.value),
                            }
                        )
            elif isinstance(child, ast.AnnAssign):
                if (
                    isinstance(child.target, ast.Attribute)
                    and isinstance(child.target.value, ast.Name)
                    and child.target.value.id == "self"
                ):
                    instance_attributes.append(
                        {
                            "name": child.target.attr,
                            "type": get_node_name(child.annotation),
                            "value": (
                                get_source_segment(source_code, child.value)
                                if child.value
                                else None
                            ),
                        }
                    )
        return instance_attributes

    async def _process_class(self, node: ast.ClassDef) -> ExtractedClass | None:
        """Process a class node to extract information."""
        try:
            source_code = getattr(self.context, "source_code", "") or ""
            docstring = ast.get_docstring(node) or ""
            source = get_source_segment(source_code, node) or ""

            # Initialize dependencies
            dependencies: dict[str, set[str]] = {}
            if hasattr(self.context, "dependency_analyzer") and self.context.dependency_analyzer:
                analyzer = self.context.dependency_analyzer
                if hasattr(analyzer, "analyze_dependencies"):
                    deps = analyzer.analyze_dependencies(node)
                    if isinstance(deps, dict):
                        dependencies = deps

            # Create the extracted class
            extracted_class = ExtractedClass(
                name=node.name,
                lineno=node.lineno,
                source=source,
                docstring=docstring,
                metrics={},  # Initialize as empty dict, will be populated below
                dependencies=dependencies,
                decorators=self._extract_decorators(node),
                complexity_warnings=[],
                ast_node=node,
                methods=await self._extract_methods(node),
                attributes=self._extract_attributes(node),
                instance_attributes=self._extract_instance_attributes(node),
                bases=self._extract_bases(node),
                metaclass=self._extract_metaclass(node),
                is_exception=self._is_exception_class(node),
            )

            # Ensure docstring_info is set correctly
            if isinstance(docstring, str):
                extracted_class.docstring_info = self.docstring_parser.parse(docstring)
            elif isinstance(docstring, dict):
                extracted_class.docstring_info = DocstringData(**docstring)

            # Calculate metrics using the metrics calculator
            if self.metrics_calculator:
                metrics = self.metrics_calculator.calculate_metrics(
                    source, self.context.module_name
                )
                if isinstance(metrics, dict):
                    extracted_class.metrics = metrics

            return extracted_class
        except Exception as e:
            handle_extraction_error(
                self.logger,
                self.errors,
                f"Class {node.name}",
                e,
                extra={"class_name": node.name},
            )
            return None
