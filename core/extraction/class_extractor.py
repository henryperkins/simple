"""Class extraction module for Python source code analysis."""

import ast
from typing import List, Dict, Any, Optional

from core.logger import LoggerSetup
from core.metrics import Metrics
from core.types import ExtractedClass, ExtractedFunction, ExtractionContext
from core.extraction.function_extractor import FunctionExtractor
from core.extraction.dependency_analyzer import extract_dependencies_from_node
from core.utils import handle_extraction_error, get_source_segment
from core.docstringutils import DocstringUtils, get_node_name  # Ensure correct import

logger = LoggerSetup.get_logger(__name__)

class ClassExtractor:
    """Handles extraction of classes from Python source code."""

    def __init__(self, context: ExtractionContext, metrics_calculator: Metrics, function_extractor: FunctionExtractor):
        self.logger = logger
        self.context = context
        self.metrics_calculator = metrics_calculator
        self.function_extractor = function_extractor
        self.errors: List[str] = []

    async def extract_classes(self, tree: ast.AST) -> List[ExtractedClass]:
        """Extract all classes from the AST."""
        classes = []
        try:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self.logger.debug(f"Found class: {node.name}")
                    try:
                        extracted_class = await self._process_class(node)
                        classes.append(extracted_class)
                    except Exception as e:
                        handle_extraction_error(self.logger, self.errors, node.name, e)
            return classes
        except Exception as e:
            self.logger.error("Error extracting classes: %s", e)
            raise

    async def _process_class(self, node: ast.ClassDef) -> ExtractedClass:
        """Process and extract information from a class AST node."""
        try:
            metadata = DocstringUtils.extract_metadata(node)
            metrics = self.metrics_calculator.calculate_class_metrics(node)
            methods = await self._extract_methods(node)
            return ExtractedClass(
                name=metadata["name"],
                lineno=metadata["lineno"],
                source=get_source_segment(self.context.source_code, node),
                docstring=metadata["docstring_info"]["docstring"],
                metrics=metrics,
                methods=methods,
                decorators=metadata.get("decorators", []),
                bases=self._extract_bases(node),
                attributes=self._extract_attributes(node),
                ast_node=node,
            )
        except Exception as e:
            self.logger.error(f"Failed to process class {node.name}: {e}")
            raise

    async def _extract_methods(self, node: ast.ClassDef) -> List[ExtractedFunction]:
        """Extract methods from a class definition."""
        methods = []
        for n in node.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                try:
                    extracted_function = self.function_extractor._process_function(n)
                    methods.append(extracted_function)
                except Exception as e:
                    self.logger.error(f"Error extracting method {n.name}: {e}")
        return methods

    def _extract_bases(self, node: ast.ClassDef) -> List[str]:
        """Extract base classes from a class definition."""
        return [get_node_name(base) for base in node.bases if base]

    def _extract_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract class attributes from a class definition."""
        attributes = []
        for child in node.body:
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                attributes.append(self._process_attribute(child))
        return attributes

    def _process_attribute(self, node: ast.AST) -> Dict[str, Any]:
        """Process a class-level attribute assignment."""
        if isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            return {
                "name": targets[0] if targets else None,
                "type": get_node_name(node.value) if node.value else "Any",
                "value": get_source_segment(self.context.source_code, node.value) if node.value else None
            }
        return {}
