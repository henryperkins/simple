"""
Class extraction module for Python source code analysis.

This module provides functionality to extract class definitions and related
metadata from Python source code using the Abstract Syntax Tree (AST).
It identifies class attributes, methods, base classes, decorators, and
instance attributes. Additionally, it determines if a class is an exception
and handles errors during extraction.

Classes:
    ClassExtractor: Extracts class-related information from Python source code.

Example usage:
    extractor = ClassExtractor(context, metrics_calculator)
    classes = await extractor.extract_classes(ast_tree)
"""

import ast
from typing import List, Dict, Any, Optional

from core.logger import LoggerSetup
from core.metrics import Metrics
from core.types import ExtractedClass, ExtractedFunction, ExtractionContext
from core.extraction.function_extractor import FunctionExtractor
from core.extraction.dependency_analyzer import extract_dependencies_from_node
from core.utils import handle_extraction_error, get_source_segment, NodeNameVisitor
from core.docstringutils import DocstringUtils

logger = LoggerSetup.get_logger(__name__)

class ClassExtractor:
    """Handles extraction of classes from Python source code."""

    def __init__(self, context: ExtractionContext, metrics_calculator: Metrics):
        """Initialize the ClassExtractor."""
        self.logger = logger
        self.context = context
        self.metrics_calculator = metrics_calculator
        self.function_extractor = FunctionExtractor(context, metrics_calculator)
        self.errors: List[Dict[str, Any]] = []

    async def extract_classes(self, tree: ast.AST) -> List[ExtractedClass]:
        """Extract classes from the AST."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                try:
                    if self._should_process_class(node):
                        extracted_class = await self._process_class(node)
                        classes.append(extracted_class)
                        self.logger.debug("Extracted class: %s", node.name)
                except Exception as e:
                    handle_extraction_error(self.logger, self.errors, node.name, e)
        return classes

    def _should_process_class(self, node: ast.ClassDef) -> bool:
        """Determine if a class should be processed."""
        return not (not self.context.include_private and node.name.startswith("_"))

    async def _process_class(self, node: ast.ClassDef) -> ExtractedClass:
        """Process a class node to extract information."""
        self.logger.debug("Processing class: %s", node.name)
        try:
            metadata = DocstringUtils.extract_metadata(node)
            base_classes = self._extract_bases(node)
            
            # Get source segment only once for the entire class
            class_source = get_source_segment(self.context.source_code, node)
            if not class_source:
                self.logger.warning(f"Could not extract source for class {node.name}")
                class_source = ""
            
            if "description" not in metadata["docstring_info"]:
                metadata["docstring_info"]["description"] = ""
            if base_classes:
                metadata["docstring_info"]["description"] += f"\n\nThis class inherits from: {', '.join(base_classes)}."
            
            metrics = self.metrics_calculator.calculate_class_metrics(node)
            
            extracted_class = ExtractedClass(
                name=metadata["name"],
                docstring=metadata["docstring_info"]["docstring"],
                lineno=metadata["lineno"],
                source=class_source,
                metrics=metrics,
                dependencies=extract_dependencies_from_node(node),
                bases=base_classes,
                methods=await self._extract_methods(node),
                attributes=self._extract_attributes(node),
                is_exception=self._is_exception_class(node),
                decorators=self._extract_decorators(node),
                instance_attributes=self._extract_instance_attributes(node),
                metaclass=self._extract_metaclass(node),
                complexity_warnings=[],
                ast_node=node,
            )
            return extracted_class
        except Exception as e:
            self.logger.error("Failed to process class %s: %s", node.name, e, exc_info=True)
            self.errors.append({
                'name': node.name,
                'lineno': getattr(node, 'lineno', 'Unknown'),
                'error': str(e)
            })
            raise

    def _extract_bases(self, node: ast.ClassDef) -> List[str]:
        """Extract base classes from a class definition."""
        bases = []
        for base in node.bases:
            try:
                visitor = NodeNameVisitor()
                visitor.visit(base)
                bases.append(visitor.name)
            except Exception as e:
                self.logger.error("Error extracting base class: %s", e, exc_info=True)
                bases.append("unknown")
        return bases

    async def _extract_methods(self, node: ast.ClassDef) -> List[ExtractedFunction]:
        """Extract methods from a class definition."""
        methods = []
        for n in node.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                try:
                    extracted_methods = await self.function_extractor.extract_functions([n])
                    methods.extend(extracted_methods)
                    for method in extracted_methods:
                        self.logger.debug("Extracted method: %s", method.name)
                except Exception as e:
                    self.logger.error("Error extracting method %s: %s", n.name, e, exc_info=True)
        return methods

    def _extract_attributes(self, node: ast.ClassDef) -> List[str]:
        """Extract class attributes from a class definition."""
        attributes = []
        for child in node.body:
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                attr_info = self._process_attribute(child)
                if attr_info and attr_info["name"]:
                    attributes.append(attr_info["name"])
                    self.logger.debug("Extracted attribute: %s", attr_info["name"])
        return attributes

    def _process_attribute(self, node: ast.AST) -> Optional[Dict[str, Any]]:
        """Process a class-level attribute assignment."""
        try:
            if isinstance(node, ast.Assign):
                targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
                value = get_source_segment(self.context.source_code, node.value) if node.value else None
                visitor = NodeNameVisitor()
                visitor.visit(node.value)
                return {
                    "name": targets[0] if targets else None,
                    "value": value,
                    "type": visitor.name if node.value else "Any",
                }
            return None
        except Exception as e:
            self.logger.error("Error processing attribute: %s", e, exc_info=True)
            return None

    def _extract_decorators(self, node: ast.ClassDef) -> List[str]:
        """Extract decorator names from a class definition."""
        decorators = []
        for decorator in node.decorator_list:
            try:
                visitor = NodeNameVisitor()
                visitor.visit(decorator)
                decorators.append(visitor.name)
                self.logger.debug("Extracted decorator: %s", visitor.name)
            except Exception as e:
                self.logger.error("Error extracting decorator: %s", e, exc_info=True)
                decorators.append("unknown_decorator")
        return decorators

    def _extract_instance_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract instance attributes from class methods."""
        instance_attributes = []
        processed_attrs = set()

        def process_assign(stmt: ast.Assign, class_name: str) -> Optional[Dict[str, Any]]:
            try:
                if isinstance(stmt.targets[0], ast.Attribute) and isinstance(stmt.targets[0].value, ast.Name):
                    if stmt.targets[0].value.id == "self":
                        attr_name = stmt.targets[0].attr
                        if attr_name not in processed_attrs:
                            processed_attrs.add(attr_name)
                            value_visitor = NodeNameVisitor()
                            if stmt.value:
                                value_visitor.visit(stmt.value)
                            return {
                                "name": attr_name,
                                "type": value_visitor.name if stmt.value else "Any",
                                "value": value_visitor.name if stmt.value else None,
                                "defined_in": class_name,
                            }
                return None
            except Exception as e:
                self.logger.error(f"Error processing assignment: {e}")
                return None

        def extract_from_class(class_node: ast.ClassDef, is_parent: bool = False) -> None:
            for child in class_node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "__init__":
                    for stmt in child.body:
                        attr_info = None
                        if isinstance(stmt, ast.Assign):
                            attr_info = process_assign(stmt, class_node.name)
                        elif isinstance(stmt, ast.AnnAssign):
                            # Handle annotated assignments
                            if isinstance(stmt.target, ast.Attribute) and isinstance(stmt.target.value, ast.Name):
                                if stmt.target.value.id == "self":
                                    attr_name = stmt.target.attr
                                    if attr_name not in processed_attrs:
                                        processed_attrs.add(attr_name)
                                        type_visitor = NodeNameVisitor()
                                        value_visitor = NodeNameVisitor()
                                        
                                        if stmt.annotation:
                                            type_visitor.visit(stmt.annotation)
                                        if stmt.value:
                                            value_visitor.visit(stmt.value)
                                            
                                        attr_info = {
                                            "name": attr_name,
                                            "type": type_visitor.name or "Any",
                                            "value": value_visitor.name if stmt.value else None,
                                            "defined_in": class_node.name,
                                        }

                        if attr_info:
                            if is_parent:
                                attr_info["inherited"] = True
                            instance_attributes.append(attr_info)
                            self.logger.debug(f"Extracted instance attribute: {attr_info['name']}")

        # Process the current class
        extract_from_class(node)

        # Process parent classes
        for base in node.bases:
            try:
                visitor = NodeNameVisitor()
                visitor.visit(base)
                base_name = visitor.name
                # Find and process parent class if it's in the same module
                for parent_node in ast.walk(self.context.tree):
                    if isinstance(parent_node, ast.ClassDef) and parent_node.name == base_name:
                        extract_from_class(parent_node, is_parent=True)
                        break
            except Exception as e:
                self.logger.error(f"Error processing parent class: {e}")

        return instance_attributes

    def _extract_metaclass(self, node: ast.ClassDef) -> Optional[str]:
        """Extract the metaclass if specified in the class definition."""
        for keyword in node.keywords:
            if keyword.arg == "metaclass":
                visitor = NodeNameVisitor()
                visitor.visit(keyword.value)
                return visitor.name
        return None

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is an exception class."""
        for base in node.bases:
            visitor = NodeNameVisitor()
            visitor.visit(base)
            base_name = visitor.name
            if base_name in {"Exception", "BaseException"}:
                return True
        return False