"""
Class extraction module for Python source code analysis.

This module provides functionality to extract class definitions and related
metadata from Python source code using the Abstract Syntax Tree (AST).
"""

import ast
from typing import Any, Optional, List, Dict
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.metrics import Metrics
from core.types import (
    ExtractionContext,
    ExtractedClass,
    ExtractedFunction,
    MetricData
)
from utils import handle_extraction_error, get_source_segment, NodeNameVisitor, get_node_name


class ClassExtractor:
    """Handles extraction of classes from Python source code."""

    def __init__(
        self,
        context: ExtractionContext,
        metrics_calculator: Metrics,
        correlation_id: Optional[str] = None
    ) -> None:
        """Initialize the class extractor.

        Args:
            context (ExtractionContext): The context for extraction, including settings and source code.
            metrics_calculator (Metrics): The metrics calculator for analyzing class complexity.
            correlation_id (Optional[str]): An optional correlation ID for logging purposes.
        """
        self.logger = CorrelationLoggerAdapter(LoggerSetup.get_logger(__name__), correlation_id)
        self.context = context
        self.metrics_calculator = metrics_calculator
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
                            self.logger.info(
                                f"Successfully extracted class: {node.name}",
                                extra={'class_name': node.name}
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
                docstring_info=None  # Set if available
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

    def _should_process_class(self, node: ast.ClassDef) -> bool:
        """Determine whether the class should be processed.

        Args:
            node (ast.ClassDef): The class node to check.

        Returns:
            bool: True if the class should be processed, False otherwise.
        """
        return not (
            (not self.context.include_private and node.name.startswith('_')) or
            (not self.context.include_magic and node.name.startswith('__'))
        )

    def _extract_bases(self, node: ast.ClassDef) -> list[str]:
        """Extract base classes using NodeNameVisitor.

        Args:
            node (ast.ClassDef): The class node to process.

        Returns:
            list[str]: A list of base class names.
        """
        bases: list[str] = []
        for base in node.bases:
            try:
                visitor = NodeNameVisitor()
                visitor.visit(base)
                if visitor.name:
                    bases.append(visitor.name)
            except Exception as e:
                self.logger.debug(f"Error extracting base class: {e}")
                bases.append("unknown_base")
        return bases

    async def _extract_methods(self, node: ast.ClassDef) -> list[ExtractedFunction]:
        """Extract methods using FunctionExtractor.

        Args:
            node (ast.ClassDef): The class node to process.

        Returns:
            list[ExtractedFunction]: A list of extracted methods.
        """
        methods: list[ExtractedFunction] = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                try:
                    extracted_methods = await self.context.function_extractor.extract_functions(child)
                    methods.extend(extracted_methods)
                except Exception as e:
                    self.logger.error(f"Error extracting method {child.name}: {e}")
        return methods

    def _extract_attributes(self, node: ast.ClassDef) -> list[Dict[str, Any]]:
        """Extract class attributes.

        Args:
            node (ast.ClassDef): The class node to process.

        Returns:
            list[Dict[str, Any]]: A list of class attributes.
        """
        attributes: list[Dict[str, Any]] = []
        for child in node.body:
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                attr_info = self._process_attribute(child)
                if attr_info:
                    attributes.append(attr_info)
        return attributes

    def _extract_instance_attributes(self, node: ast.ClassDef) -> list[Dict[str, Any]]:
        """Extract instance attributes from __init__ method.

        Args:
            node (ast.ClassDef): The class node to process.

        Returns:
            list[Dict[str, Any]]: A list of instance attributes.
        """
        instance_attrs: list[Dict[str, Any]] = []
        init_method = next(
            (m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == '__init__'),
            None
        )

        if init_method:
            for stmt in init_method.body:
                if isinstance(stmt, ast.Assign):
                    attr_info = self._process_self_attribute(stmt)
                    if attr_info:
                        instance_attrs.append(attr_info)

        return instance_attrs

    def _extract_metaclass(self, node: ast.ClassDef) -> Optional[str]:
        """Extract metaclass information.

        Args:
            node (ast.ClassDef): The class node to process.

        Returns:
            Optional[str]: The metaclass name, if any.
        """
        for keyword in node.keywords:
            if keyword.arg == 'metaclass':
                visitor = NodeNameVisitor()
                visitor.visit(keyword.value)
                return visitor.name
        return None

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """Check if the class is an exception class.

        Args:
            node (ast.ClassDef): The class node to check.

        Returns:
            bool: True if the class is an exception class, False otherwise.
        """
        exception_bases = ('Exception', 'BaseException')
        for base in self._extract_bases(node):
            if base in exception_bases:
                return True
        return False

    def _extract_decorators(self, node: ast.ClassDef) -> list[str]:
        """Extract decorator information.

        Args:
            node (ast.ClassDef): The class node to process.

        Returns:
            list[str]: A list of decorator names.
        """
        decorators: list[str] = []
        for decorator in node.decorator_list:
            visitor = NodeNameVisitor()
            visitor.visit(decorator)
            if visitor.name:
                decorators.append(visitor.name)
        return decorators

    def _process_self_attribute(self, node: ast.Assign) -> Optional[Dict[str, Any]]:
        """Process self attribute assignment in __init__ method.

        Args:
            node (ast.Assign): The assignment node to process.

        Returns:
            Optional[Dict[str, Any]]: The attribute information, or None if processing fails.
        """
        try:
            if (isinstance(node.targets[0], ast.Attribute) and
                isinstance(node.targets[0].value, ast.Name) and
                node.targets[0].value.id == 'self'):

                attr_name = node.targets[0].attr

                # Get type information if available
                value_type = "Any"
                if isinstance(node, ast.AnnAssign) and node.annotation:
                    value_type = get_node_name(node.annotation)

                # Get value if available
                value = None
                if node.value:
                    value = get_source_segment(self.context.source_code or "", node.value)

                return {
                    "name": attr_name,
                    "type": value_type,
                    "value": value,
                    "defined_in": "__init__"
                }

        except Exception as e:
            self.logger.debug(f"Error processing self attribute: {e}")
        return None

    def _process_attribute(self, node: ast.AST) -> Optional[Dict[str, Any]]:
        """Process class-level attribute assignment.

        Args:
            node (ast.AST): The assignment node to process.

        Returns:
            Optional[Dict[str, Any]]: The attribute information, or None if processing fails.
        """
        try:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        value = get_source_segment(self.context.source_code or "", node.value)
                        return {
                            "name": target.id,
                            "type": "Any",
                            "value": value
                        }
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                value = None
                if node.value:
                    value = get_source_segment(self.context.source_code or "", node.value)

                return {
                    "name": node.target.id,
                    "type": get_node_name(node.annotation),
                    "value": value
                }
        except Exception as e:
            self.logger.debug(f"Error processing attribute: {e}")
        return None
