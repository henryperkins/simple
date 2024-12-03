"""Class extraction module.

This module provides functionality to extract class definitions from Python source code.
It uses the Abstract Syntax Tree (AST) to parse and analyze the code, extracting information
about classes, their methods, attributes, and other metadata.
"""

import ast
from typing import List, Dict, Any, Optional, Set
from core.logger import LoggerSetup
from core.metrics import Metrics
from .types import ExtractedClass, ExtractedFunction, ExtractionContext
from .utils import ASTUtils
from .function_extractor import FunctionExtractor

logger = LoggerSetup.get_logger(__name__)

class ClassExtractor:
    """Handles extraction of classes from Python source code.

    Attributes:
        context (ExtractionContext): The context for extraction, including configuration options.
        metrics_calculator (Metrics): An instance of the Metrics class for calculating code metrics.
        errors (List[str]): A list to store error messages encountered during extraction.
    """

    def __init__(self, context: ExtractionContext, metrics_calculator: Metrics):
        """Initialize the ClassExtractor.

        Args:
            context (ExtractionContext): The extraction context containing settings and configurations.
            metrics_calculator (Metrics): An instance for calculating metrics related to code complexity.
        """
        self.logger = logger
        self.context = context
        self.metrics_calculator = metrics_calculator
        self.ast_utils = ASTUtils()
        self.function_extractor = FunctionExtractor(context, metrics_calculator)
        self.errors: List[str] = []
        self._current_class: Optional[ast.ClassDef] = None
        self.logger.debug("Initialized ClassExtractor")

    def extract_classes(self, tree: ast.AST) -> List[ExtractedClass]:
        """Extract all classes from the AST.

        Args:
            tree (ast.AST): The root of the AST representing the parsed Python source code.

        Returns:
            List[ExtractedClass]: A list of ExtractedClass objects containing information about each class.
        """
        self.logger.info("Starting class extraction")
        classes = []
        try:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if not self.context.include_private and node.name.startswith('_'):
                        self.logger.debug(f"Skipping private class: {node.name}")
                        continue
                    try:
                        self._current_class = node
                        extracted_class = self._process_class(node)
                        classes.append(extracted_class)
                        self.logger.debug(f"Extracted class: {extracted_class.name}")
                    except Exception as e:
                        self._handle_extraction_error(node.name, e)
                    finally:
                        self._current_class = None
            self.logger.info(f"Class extraction completed: {len(classes)} classes extracted")
        except Exception as e:
            self.logger.error(f"Error in extract_classes: {str(e)}", exc_info=True)
        return classes

    def _process_class(self, node: ast.ClassDef) -> ExtractedClass:
        """Process a class definition node.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            ExtractedClass: An object containing detailed information about the class.
        """
        self.logger.debug(f"Processing class: {node.name}")
        try:
            metrics = self._calculate_class_metrics(node)
            complexity_warnings = self._get_complexity_warnings(metrics)

            source = None
            if getattr(self.context, 'include_source', True):  # Safe access
                source = self.ast_utils.get_source_segment(node)

            extracted_class = ExtractedClass(
                name=node.name,
                docstring=ast.get_docstring(node),
                lineno=node.lineno,
                source=source,
                metrics=metrics,
                dependencies=self._extract_dependencies(node),
                bases=self._extract_bases(node),
                methods=self._extract_methods(node),
                attributes=self._extract_attributes(node),
                is_exception=self._is_exception_class(node),
                decorators=self._extract_decorators(node),
                instance_attributes=self._extract_instance_attributes(node),
                metaclass=self._extract_metaclass(node),
                complexity_warnings=complexity_warnings,
                ast_node=node
            )
            self.logger.debug(f"Completed processing class: {node.name}")
            return extracted_class
        except Exception as e:
            self.logger.error(f"Failed to process class {node.name}: {e}", exc_info=True)
            raise

    def _process_attribute(self, node: ast.AST) -> Optional[Dict[str, Any]]:
        """Process a class-level attribute assignment.

        Args:
            node (ast.AST): The AST node representing the attribute assignment.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with attribute details or None if processing fails.
        """
        try:
            if isinstance(node, ast.Assign):
                targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
                value = self.ast_utils.get_source_segment(node.value) if node.value else None
                return {
                    "name": targets[0] if targets else None,
                    "value": value,
                    "type": self.ast_utils.get_name(node.value) if node.value else 'Any'
                }
            return None
        except Exception as e:
            self.logger.error(f"Error processing attribute: {e}")
            return None

    def _process_instance_attribute(self, stmt: ast.Assign) -> Optional[Dict[str, Any]]:
        """Process an instance attribute assignment statement.

        Args:
            stmt (ast.Assign): The AST node representing the assignment statement.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with instance attribute details or None if processing fails.
        """
        try:
            if isinstance(stmt.targets[0], ast.Attribute) and isinstance(stmt.targets[0].value, ast.Name):
                if stmt.targets[0].value.id == 'self':
                    return {
                        'name': stmt.targets[0].attr,
                        'type': self.ast_utils.get_name(stmt.value) if stmt.value else 'Any',
                        'value': self.ast_utils.get_source_segment(stmt.value) if stmt.value else None
                    }
            return None
        except Exception as e:
            self.logger.error(f"Error processing instance attribute: {e}")
            return None

    def _extract_bases(self, node: ast.ClassDef) -> List[str]:
        """Extract base classes from a class definition.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            List[str]: A list of base class names.
        """
        self.logger.debug(f"Extracting bases for class: {node.name}")
        bases = []
        for base in node.bases:
            try:
                base_name = self.ast_utils.get_name(base)
                bases.append(base_name)
            except Exception as e:
                self.logger.error(f"Error extracting base class: {e}", exc_info=True)
                bases.append('unknown')
        return bases

    def _extract_methods(self, node: ast.ClassDef) -> List[ExtractedFunction]:
        """Extract methods from a class definition.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            List[ExtractedFunction]: A list of ExtractedFunction objects representing the methods.
        """
        self.logger.debug(f"Extracting methods for class: {node.name}")
        methods = []
        for n in node.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                try:
                    method = self.function_extractor._process_function(n)
                    methods.append(method)
                    self.logger.debug(f"Extracted method: {method.name}")
                except Exception as e:
                    self.logger.error(f"Error extracting method {n.name}: {e}", exc_info=True)
        return methods

    def _extract_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract class attributes from a class definition.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing attribute details.
        """
        self.logger.debug(f"Extracting attributes for class: {node.name}")
        attributes = []
        for child in node.body:
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                attr_info = self._process_attribute(child)
                if attr_info:
                    attributes.append(attr_info)
                    self.logger.debug(f"Extracted attribute: {attr_info['name']}")
        return attributes

    def _extract_decorators(self, node: ast.ClassDef) -> List[str]:
        """Extract decorator names from a class definition.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            List[str]: A list of decorator names.
        """
        self.logger.debug(f"Extracting decorators for class: {node.name}")
        decorators = []
        for decorator in node.decorator_list:
            try:
                decorator_name = self.ast_utils.get_name(decorator)
                decorators.append(decorator_name)
                self.logger.debug(f"Extracted decorator: {decorator_name}")
            except Exception as e:
                self.logger.error(f"Error extracting decorator: {e}")
                decorators.append("unknown_decorator")
        return decorators

    def _extract_instance_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract instance attributes from the __init__ method of a class.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing instance attribute details.
        """
        self.logger.debug(f"Extracting instance attributes for class: {node.name}")
        instance_attributes = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == '__init__':
                for stmt in child.body:
                    if isinstance(stmt, ast.Assign):
                        attr_info = self._process_instance_attribute(stmt)
                        if attr_info:
                            instance_attributes.append(attr_info)
                            self.logger.debug(f"Extracted instance attribute: {attr_info['name']}")
        return instance_attributes

    def _extract_metaclass(self, node: ast.ClassDef) -> Optional[str]:
        """Extract the metaclass if specified in the class definition.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            Optional[str]: The name of the metaclass or None if not specified.
        """
        self.logger.debug(f"Extracting metaclass for class: {node.name}")
        for keyword in node.keywords:
            if keyword.arg == 'metaclass':
                return self.ast_utils.get_name(keyword.value)
        return None

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is an exception class.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            bool: True if the class is an exception class, False otherwise.
        """
        self.logger.debug(f"Checking if class is an exception: {node.name}")
        for base in node.bases:
            base_name = self.ast_utils.get_name(base)
            if base_name in {'Exception', 'BaseException'}:
                return True
        return False

    def _calculate_class_metrics(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Calculate metrics for a class.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            Dict[str, Any]: A dictionary containing metrics such as method count, complexity, and inheritance depth.
        """
        self.logger.debug(f"Calculating metrics for class: {node.name}")
        try:
            metrics = {
                'method_count': len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]),
                'complexity': self.metrics_calculator.calculate_complexity(node),
                'maintainability': self.metrics_calculator.calculate_maintainability_index(node),
                'inheritance_depth': self._calculate_inheritance_depth(node)
            }
            self.logger.debug(f"Metrics for class {node.name}: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating class metrics: {e}", exc_info=True)
            return {}

    def _calculate_inheritance_depth(self, node: ast.ClassDef) -> int:
        """Calculate the inheritance depth of a class.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            int: The inheritance depth of the class.
        """
        self.logger.debug(f"Calculating inheritance depth for class: {node.name}")
        try:
            depth = 0
            bases = node.bases
            while bases:
                depth += 1
                new_bases = []
                for base in bases:
                    base_class = self._resolve_base_class(base)
                    if base_class and base_class.bases:
                        new_bases.extend(base_class.bases)
                bases = new_bases
            self.logger.debug(f"Inheritance depth for class {node.name}: {depth}")
            return depth
        except Exception as e:
            self.logger.error(f"Error calculating inheritance depth: {e}", exc_info=True)
            return 0

    def _resolve_base_class(self, base: ast.expr) -> Optional[ast.ClassDef]:
        """Resolve a base class node to its class definition.

        Args:
            base (ast.expr): The AST node representing the base class.

        Returns:
            Optional[ast.ClassDef]: The AST node of the resolved base class or None if not found.
        """
        self.logger.debug(f"Resolving base class: {ast.dump(base)}")
        try:
            if isinstance(base, ast.Name):
                if self._current_class and self._current_class.parent:
                    for node in ast.walk(self._current_class.parent):
                        if isinstance(node, ast.ClassDef) and node.name == base.id:
                            return node
            elif isinstance(base, ast.Attribute):
                base_name = self.ast_utils.get_name(base)
                self.logger.debug(f"Complex base class name: {base_name}")
                return None
            return None
        except Exception as e:
            self.logger.error(f"Error resolving base class: {e}")
            return None

    def _get_complexity_warnings(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate warnings based on complexity metrics.

        Args:
            metrics (Dict[str, Any]): A dictionary containing class metrics.

        Returns:
            List[str]: A list of warnings related to class complexity.
        """
        self.logger.debug("Generating complexity warnings")
        warnings = []
        try:
            if metrics.get('complexity', 0) > 10:
                warnings.append("High class complexity")
            if metrics.get('method_count', 0) > 20:
                warnings.append("High method count")
            if metrics.get('inheritance_depth', 0) > 3:
                warnings.append("Deep inheritance hierarchy")
            self.logger.debug(f"Complexity warnings: {warnings}")
        except Exception as e:
            self.logger.error(f"Error generating complexity warnings: {e}", exc_info=True)
        return warnings

    def _handle_extraction_error(self, class_name: str, error: Exception) -> None:
        """Handle class extraction errors.

        Args:
            class_name (str): The name of the class being processed when the error occurred.
            error (Exception): The exception that was raised.
        """
        error_msg = f"Failed to extract class {class_name}: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        self.errors.append(error_msg)

    def _extract_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Extract dependencies from a node.

        Args:
            node (ast.AST): The AST node representing the class definition.

        Returns:
            Dict[str, Set[str]]: A dictionary containing sets of dependencies categorized by type.
        """
        self.logger.debug(f"Extracting dependencies for class: {node.name}")
        # This would typically call into the DependencyAnalyzer
        # Simplified version for class-level dependencies
        return {'imports': set(), 'calls': set(), 'attributes': set()}
