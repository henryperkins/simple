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

    async def extract_classes(self, tree: Union[ast.AST, ast.Module]) -> List[ExtractedClass]:
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

    def _should_process_class(self, node: ast.ClassDef) -> bool:
        """Determine if a class should be processed based on context settings.

        Args:
            node: The class node to check

        Returns:
            bool: True if the class should be processed, False otherwise
        """
        # Skip private classes if not included in settings
        if not self.context.include_private and node.name.startswith('_'):
            return False

        # Skip nested classes if not included in settings
        if not self.context.include_nested:
            for parent in ast.walk(self.context.tree):
                if isinstance(parent, ast.ClassDef) and node in ast.walk(parent):
                    if parent != node:  # Don't count the node itself
                        return False

        return True

    def _extract_decorators(self, node: ast.ClassDef) -> List[str]:
        """Extract decorator names from a class node.

        Args:
            node (ast.ClassDef): The class node to process.

        Returns:
            List[str]: List of decorator names.
        """
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorators.append(decorator.func.id)
                elif isinstance(decorator.func, ast.Attribute):
                    decorators.append(
                        f"{decorator.func.value.id}.{decorator.func.attr}")
            elif isinstance(decorator, ast.Attribute):
                decorators.append(f"{decorator.value.id}.{decorator.attr}")
        return decorators

    async def _extract_methods(self, node: ast.ClassDef) -> List[ExtractedFunction]:
        """Extract method definitions from a class node.

        Args:
            node (ast.ClassDef): The class node to process.

        Returns:
            List[ExtractedFunction]: List of extracted method information.
        """
        methods = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not self.context.function_extractor._should_process_function(child):
                    self.logger.debug(f"Skipping method: {child.name}")
                    continue

                try:
                    extracted_method = await self.context.function_extractor._process_function(child)
                    if extracted_method:
                        # Mark as method and set parent class
                        extracted_method.is_method = True
                        extracted_method.parent_class = node.name
                        methods.append(extracted_method)
                except Exception as e:
                    self.logger.error(
                        f"Failed to process method {child.name}: {e}",
                        exc_info=True,
                        extra={'method_name': child.name}
                    )
        return methods

    def _extract_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract class-level attributes from a class node.

        Args:
            node (ast.ClassDef): The class node to process.

        Returns:
            List[Dict[str, Any]]: List of extracted class attributes.
        """
        attributes = []
        for child in node.body:
            try:
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    # Handle annotated assignments (e.g., x: int = 1)
                    attr_value = None
                    if child.value:
                        attr_value = get_source_segment(
                            self.context.source_code or "", child.value)

                    attributes.append({
                        'name': child.target.id,
                        'type': get_node_name(child.annotation),
                        'value': attr_value
                    })
                elif isinstance(child, ast.Assign):
                    # Handle regular assignments (e.g., x = 1)
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            attr_value = get_source_segment(
                                self.context.source_code or "", child.value)
                            attributes.append({
                                'name': target.id,
                                'type': 'Any',  # Type not explicitly specified
                                'value': attr_value
                            })
            except Exception as e:
                self.logger.warning(
                    f"Error extracting attribute from {getattr(child, 'name', 'unknown')}: {e}",
                    exc_info=True
                )
                continue

        return attributes

    def _extract_bases(self, node: ast.ClassDef) -> List[str]:
        """Extract base class names from a class definition.

        Args:
            node (ast.ClassDef): The class node to process.

        Returns:
            List[str]: List of base class names.
        """
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{base.value.id}.{base.attr}")
        return bases

    def _extract_metaclass(self, node: ast.ClassDef) -> Optional[str]:
        """Extract metaclass name from class keywords if present.

        Args:
            node (ast.ClassDef): The class node to process.

        Returns:
            Optional[str]: Metaclass name if present, None otherwise.
        """
        for keyword in node.keywords:
            if keyword.arg == 'metaclass':
                if isinstance(keyword.value, ast.Name):
                    return keyword.value.id
                elif isinstance(keyword.value, ast.Attribute):
                    return f"{keyword.value.value.id}.{keyword.value.attr}"
        return None

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """Check if the class is an exception class.

        Args:
            node (ast.ClassDef): The class node to process.

        Returns:
            bool: True if the class is an exception class, False otherwise.
        """
        exception_bases = {'Exception', 'BaseException'}
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in exception_bases:
                return True
        return False

    def _extract_instance_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract instance attributes from a class node.

        Args:
            node (ast.ClassDef): The class node to process.

        Returns:
            List[Dict[str, Any]]: List of extracted instance attributes.
        """
        instance_attributes = []
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                        instance_attributes.append({
                            'name': target.attr,
                            'type': 'Any',  # Type not explicitly specified
                            'value': get_source_segment(self.context.source_code or "", child.value)
                        })
            elif isinstance(child, ast.AnnAssign):
                if isinstance(child.target, ast.Attribute) and isinstance(child.target.value, ast.Name) and child.target.value.id == 'self':
                    instance_attributes.append({
                        'name': child.target.attr,
                        'type': get_node_name(child.annotation),
                        'value': get_source_segment(self.context.source_code or "", child.value) if child.value else None
                    })
        return instance_attributes

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
            source = get_source_segment(
                self.context.source_code or "", node) or ""

            # Create the extracted class
            extracted_class = ExtractedClass(
                name=node.name,
                lineno=node.lineno,
                source=source,
                docstring=docstring,
                metrics=MetricData(),  # Will be populated below
                dependencies=self.context.dependency_analyzer.analyze_dependencies(
                    node),
                decorators=self._extract_decorators(node),
                complexity_warnings=[],
                ast_node=node,
                methods=await self._extract_methods(node),
                attributes=self._extract_attributes(node),
                instance_attributes=self._extract_instance_attributes(node),
                bases=self._extract_bases(node),
                metaclass=self._extract_metaclass(node),
                is_exception=self._is_exception_class(node)
            )

            # Calculate metrics using the metrics calculator
            metrics = self.metrics_calculator.calculate_metrics(
                source, self.context.module_name)
            extracted_class.metrics = metrics

            return extracted_class

        except Exception as e:
            self.logger.error(
                f"Failed to process class {node.name}: {e}",
                exc_info=True,
                extra={'class_name': node.name}
            )
            return None

    # ... rest of the methods remain unchanged ...
