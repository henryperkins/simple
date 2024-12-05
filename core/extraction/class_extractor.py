import ast
from typing import List, Dict, Any, Optional
from core.logger import LoggerSetup
from core.metrics import Metrics
from core.types import ExtractedClass, ExtractedFunction, ExtractionContext
from core.function_extractor import FunctionExtractor
from utils import handle_extraction_error, get_source_segment
from docstringutils import DocstringUtils
from core.dependency_analyzer import extract_dependencies_from_node

logger = LoggerSetup.get_logger(__name__)

class ClassExtractor:
    """Handles extraction of classes from Python source code."""

    def __init__(self, context: ExtractionContext, metrics_calculator: Metrics):
        """Initialize the ClassExtractor.

        Args:
            context (ExtractionContext): The extraction context containing settings and configurations.
            metrics_calculator (Metrics): An instance for calculating metrics related to code complexity.
        """
        self.logger = logger
        self.context = context
        self.metrics_calculator = metrics_calculator
        self.function_extractor = FunctionExtractor(context, metrics_calculator)
        self.errors: List[str] = []
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
                        extracted_class = self._process_class(node)
                        classes.append(extracted_class)
                        self.logger.debug(f"Extracted class: {extracted_class.name}")
                    except Exception as e:
                        handle_extraction_error(self.logger, self.errors, node.name, e)
            self.logger.info(f"Class extraction completed: {len(classes)} classes extracted")
        except Exception as e:
            self.logger.error(f"Error in extract_classes: {str(e)}", exc_info=True)
        return classes

    def _process_class(self, node: ast.ClassDef) -> ExtractedClass:
        """
        Process a class definition node to extract relevant information and metrics.

        Args:
            node (ast.ClassDef): The AST node representing the class.

        Returns:
            ExtractedClass: An object containing the extracted class details and metrics.
        """
        self.logger.debug(f"Processing class: {node.name}")
        try:
            # Extract metadata and docstring info
            metadata = DocstringUtils.extract_metadata(node)
            docstring_info = metadata['docstring_info']

            # Calculate metrics for the class
            metrics = self.metrics_calculator.calculate_class_metrics(node)
            cognitive_complexity = metrics.get('cognitive_complexity')
            halstead_metrics = metrics.get('halstead_metrics')

            # Extract source if needed
            source = get_source_segment(self.context.source_code, node) if getattr(self.context, 'include_source', True) else None

            # Create the ExtractedClass object
            extracted_class = ExtractedClass(
                name=metadata['name'],
                docstring=docstring_info['docstring'],
                lineno=metadata['lineno'],
                source=source,
                metrics=metrics,
                cognitive_complexity=cognitive_complexity,
                halstead_metrics=halstead_metrics,
                dependencies=extract_dependencies_from_node(node),
                bases=self._extract_bases(node),
                methods=self._extract_methods(node),
                attributes=self._extract_attributes(node),
                is_exception=self._is_exception_class(node),
                decorators=self._extract_decorators(node),
                instance_attributes=self._extract_instance_attributes(node),
                metaclass=self._extract_metaclass(node),
                complexity_warnings=[],
                ast_node=node
            )
            self.logger.debug(f"Completed processing class: {node.name}")
            return extracted_class
        except Exception as e:
            self.logger.error(f"Failed to process class {node.name}: {e}", exc_info=True)
            raise

    def _extract_bases(self, node: ast.ClassDef) -> List[str]:
        """Extract base classes from a class definition."""
        self.logger.debug(f"Extracting bases for class: {node.name}")
        bases = []
        for base in node.bases:
            try:
                base_name = DocstringUtils.get_node_name(base)
                bases.append(base_name)
            except Exception as e:
                self.logger.error(f"Error extracting base class: {e}", exc_info=True)
                bases.append('unknown')
        return bases

    def _extract_methods(self, node: ast.ClassDef) -> List[ExtractedFunction]:
        """Extract methods from a class definition."""
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
        """Extract class attributes from a class definition."""
        self.logger.debug(f"Extracting attributes for class: {node.name}")
        attributes = []
        for child in node.body:
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                attr_info = self._process_attribute(child)
                if attr_info:
                    attributes.append(attr_info)
                    self.logger.debug(f"Extracted attribute: {attr_info['name']}")
        return attributes

    def _process_attribute(self, node: ast.AST) -> Optional[Dict[str, Any]]:
        """Process a class-level attribute assignment."""
        try:
            if isinstance(node, ast.Assign):
                targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
                value = get_source_segment(self.context.source_code, node.value) if node.value else None
                return {
                    "name": targets[0] if targets else None,
                    "value": value,
                    "type": DocstringUtils.get_node_name(node.value) if node.value else 'Any'
                }
            return None
        except Exception as e:
            self.logger.error(f"Error processing attribute: {e}")
            return None

    def _extract_decorators(self, node: ast.ClassDef) -> List[str]:
        """Extract decorator names from a class definition."""
        self.logger.debug(f"Extracting decorators for class: {node.name}")
        decorators = []
        for decorator in node.decorator_list:
            try:
                decorator_name = DocstringUtils.get_node_name(decorator)
                decorators.append(decorator_name)
                self.logger.debug(f"Extracted decorator: {decorator_name}")
            except Exception as e:
                self.logger.error(f"Error extracting decorator: {e}", exc_info=True)
                decorators.append("unknown_decorator")
        return decorators

    def _extract_instance_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract instance attributes from the __init__ method of a class."""
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

    def _process_instance_attribute(self, stmt: ast.Assign) -> Optional[Dict[str, Any]]:
        """Process an instance attribute assignment statement."""
        try:
            if isinstance(stmt.targets[0], ast.Attribute) and isinstance(stmt.targets[0].value, ast.Name):
                if stmt.targets[0].value.id == 'self':
                    return {
                        'name': stmt.targets[0].attr,
                        'type': DocstringUtils.get_node_name(stmt.value) if stmt.value else 'Any',
                        'value': get_source_segment(self.context.source_code, stmt.value) if stmt.value else None
                    }
            return None
        except Exception as e:
            self.logger.error(f"Error processing instance attribute: {e}")
            return None

    def _extract_metaclass(self, node: ast.ClassDef) -> Optional[str]:
        """Extract the metaclass if specified in the class definition."""
        self.logger.debug(f"Extracting metaclass for class: {node.name}")
        for keyword in node.keywords:
            if keyword.arg == 'metaclass':
                return DocstringUtils.get_node_name(keyword.value)
        return None

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is an exception class."""
        self.logger.debug(f"Checking if class is an exception: {node.name}")
        for base in node.bases:
            base_name = DocstringUtils.get_node_name(base)
            if base_name in {'Exception', 'BaseException'}:
                return True
        return False