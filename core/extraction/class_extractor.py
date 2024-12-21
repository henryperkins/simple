"""
Class extraction module for Python source code analysis.
"""

import ast
import uuid
from typing import Any, Optional, List, Dict

from core.logger import CorrelationLoggerAdapter
from core.types import ExtractionContext, ExtractedClass
from core.exceptions import ExtractionError
from utils import handle_extraction_error, log_and_raise_error
from core.extraction.extraction_utils import (
    extract_decorators,
    extract_attributes,
    extract_instance_attributes,
    extract_bases,
    get_node_name,
)


class ClassExtractor:
    """Handles extraction of classes from Python source code."""

    def __init__(
        self,
        context: Optional[ExtractionContext],
        correlation_id: str | None = None,
    ) -> None:
        """Initialize the ClassExtractor."""
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = CorrelationLoggerAdapter(
            context.logger,
            extra={"correlation_id": self.correlation_id},
        )
        self.context = context
        self.function_extractor = self.context.function_extractor
        self.errors: List[str] = []
        from core.dependency_injection import Injector  # Local import

        self.docstring_parser = Injector.get("docstring_processor")

    async def extract_classes(
        self, tree: ast.AST, module_metrics: Any
    ) -> List[ExtractedClass]:
        """Extract class definitions from AST nodes."""
        classes: List[ExtractedClass] = []
        self.logger.info("Starting class extraction.")
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                try:
                    extracted_class = await self._process_class(node, module_metrics)
                    if extracted_class:
                        classes.append(extracted_class)
                        self.logger.debug(
                            f"Extracted class: {extracted_class.name}, Methods: {[method.name for method in extracted_class.methods]}, Attributes: {extracted_class.attributes}",
                            extra={
                                "class_name": extracted_class.name,
                                "correlation_id": self.correlation_id,
                            },
                        )
                        # Update scan progress
                        if self.context.metrics_collector and self.context.module_name:
                            self.context.metrics_collector.update_scan_progress(
                                self.context.module_name,
                                "class",
                                node.name,
                            )
                except Exception as e:
                    log_and_raise_error(
                        self.logger,
                        e,
                        ExtractionError,
                        f"Error extracting class {node.name}",
                        self.correlation_id,
                        class_name=node.name,
                    )
        self.logger.info(
            f"Class extraction completed. Total classes extracted: {len(classes)}"
        )
        return classes

    def _should_process_class(self, node: ast.ClassDef) -> bool:
        """Check if a class should be processed based on context."""
        if not self.context.include_private and node.name.startswith("_"):
            self.logger.debug(
                f"Skipping private class: {node.name}",
                extra={"class_name": node.name, "correlation_id": self.correlation_id},
            )
            return False
        if not self.context.include_nested and self._is_nested_class(node):
            self.logger.debug(
                f"Skipping nested class: {node.name}",
                extra={"class_name": node.name, "correlation_id": self.correlation_id},
            )
            return False
        return True

    def _is_nested_class(self, node: ast.ClassDef) -> bool:
        """Check if the class is nested within another class."""
        if not hasattr(self.context, "tree") or self.context.tree is None:
            return False  # Cannot determine without the full tree
        for parent in ast.walk(self.context.tree):
            if isinstance(parent, ast.ClassDef) and node in ast.walk(parent):
                if parent != node:  # Don't count the node itself
                    return True
        return False

    def _extract_metaclass(self, node: ast.ClassDef) -> Optional[str]:
        """Extract metaclass name."""
        for keyword in node.keywords:
            if keyword.arg == "metaclass":
                return get_node_name(keyword.value)
        return None

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """Check if the class inherits from a known Exception class."""
        return any(
            get_node_name(base) in ("Exception", "BaseException") for base in node.bases
        )

    def _extract_abstract_methods(self, node: ast.ClassDef) -> List[str]:
        """Extract abstract method names from a class node."""
        abstract_methods = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(
                    isinstance(d, ast.Name) and d.id == "abstractmethod"
                    for d in child.decorator_list
                ):
                    abstract_methods.append(child.name)
        return abstract_methods

    def _extract_properties(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract property methods with their getter/setter pairs."""
        properties = []
        for method in node.body:
            if isinstance(method, ast.FunctionDef):
                if any(
                    isinstance(d, ast.Name) and d.id == "property"
                    for d in method.decorator_list
                ):
                    properties.append(
                        {
                            "name": method.name,
                            "type": (
                                get_node_name(method.returns)
                                if method.returns
                                else "Any"
                            ),
                            "has_setter": any(
                                m.name == f"{method.name}.setter"
                                for m in node.body
                                if isinstance(m, ast.FunctionDef)
                            ),
                        }
                    )
        return properties

    def _extract_class_variables(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract class variables and their types from a class definition."""
        class_variables = []

        # Get source code context
        source_code = self.context.get_source_code()

        for child in node.body:
            try:
                # Handle annotated class variables
                if isinstance(child, ast.AnnAssign) and isinstance(
                    child.target, ast.Name
                ):
                    attr_value = ast.unparse(child.value) if child.value else None
                    class_variables.append(
                        {
                            "name": child.target.id,
                            "type": get_node_name(child.annotation),
                            "value": attr_value,
                            "lineno": child.lineno,
                        }
                    )
                # Handle regular assignments
                elif isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            attr_value = ast.unparse(child.value)
                            class_variables.append(
                                {
                                    "name": target.id,
                                    "type": "Any",
                                    "value": attr_value,
                                    "lineno": child.lineno,
                                }
                            )
            except Exception as e:
                log_and_raise_error(
                    self.logger,
                    e,
                    ExtractionError,
                    f"Error extracting class variable in class {node.name}",
                    self.correlation_id,
                    class_name=node.name,
                    attribute_name=getattr(child, "name", "unknown"),
                )

        return class_variables

    def _group_methods_by_access(self, node: ast.ClassDef) -> Dict[str, List[str]]:
        """Group methods by their access modifiers."""
        method_groups = {
            "public": [],
            "private": [],
            "protected": [],
        }
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name.startswith("__") and not child.name.endswith("__"):
                    method_groups["private"].append(child.name)
                elif child.name.startswith("_"):
                    method_groups["protected"].append(child.name)
                else:
                    method_groups["public"].append(child.name)
        return method_groups

    def _get_inheritance_chain(self, node: ast.ClassDef) -> List[str]:
        """Get the inheritance chain for a class."""
        chain = []
        current = node
        while current:
            if current.name:
                chain.append(current.name)
            if isinstance(current, ast.ClassDef) and current.bases:
                for base in current.bases:
                    base_name = get_node_name(base)
                    if base_name in chain:
                        break  # Avoid infinite loop in case of circular inheritance
                    try:
                        if self.context.tree is None:
                            current = None
                            break  # Exit if no tree
                        base_node = next(
                            n
                            for n in ast.walk(self.context.tree)
                            if isinstance(n, ast.ClassDef) and n.name == base_name
                        )
                        current = base_node
                        break
                    except StopIteration:
                        current = None  # Base class not found in the current module
                        break
            else:
                current = None
        return chain[::-1]  # Reverse the chain to show from base to derived

    async def _process_class(
        self, node: ast.ClassDef, module_metrics: Any
    ) -> Optional[ExtractedClass]:
        """Process a single class definition."""
        try:
            source_code = self.context.get_source_code()
            if not source_code:
                log_and_raise_error(
                    self.logger,
                    ExtractionError("Source code is not available in the context"),
                    ExtractionError,
                    "Source code is not available in the context",
                    self.correlation_id,
                    class_name=node.name,
                )

            docstring = ast.get_docstring(node) or ""
            decorators = extract_decorators(node)
            bases = extract_bases(node)
            methods = await self.function_extractor.extract_functions(
                node.body, module_metrics
            )
            attributes = extract_attributes(node, source_code)
            instance_attributes = extract_instance_attributes(node, source_code)
            metaclass = self._extract_metaclass(node)
            is_exception = self._is_exception_class(node)

            extracted_class = ExtractedClass(
                name=node.name,
                lineno=node.lineno,
                source=ast.unparse(node),
                docstring=docstring,
                decorators=decorators,
                bases=bases,
                methods=methods,
                attributes=attributes,
                instance_attributes=instance_attributes,
                metaclass=metaclass,
                is_exception=is_exception,
                ast_node=node,
                dependencies=(
                    self.context.dependency_analyzer.analyze_dependencies(node)
                    if self.context.dependency_analyzer
                    else {}
                ),
                complexity_warnings=[],
                is_dataclass=any(
                    d.id == "dataclass" if isinstance(d, ast.Name) else d == "dataclass"
                    for d in decorators
                ),
                is_abstract=any(
                    base == "ABC" for base in bases if isinstance(base, str)
                ),
                abstract_methods=self._extract_abstract_methods(node),
                property_methods=self._extract_properties(node),
                class_variables=self._extract_class_variables(node),
                method_groups=self._group_methods_by_access(node),
                inheritance_chain=self._get_inheritance_chain(node),
            )

            if docstring:
                extracted_class.docstring_info = self.docstring_parser.parse(docstring)

            # Use module-level metrics for class-level metrics
            extracted_class.metrics = module_metrics.__dict__.copy()
            extracted_class.metrics["total_classes"] = 1
            extracted_class.metrics["scanned_classes"] = (
                1 if extracted_class.docstring_info else 0
            )

            return extracted_class

        except Exception as e:
            log_and_raise_error(
                self.logger,
                e,
                ExtractionError,
                f"Error processing class {node.name}",
                self.correlation_id,
                class_name=node.name,
            )
            return None
