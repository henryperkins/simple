"""Refactored AST extraction modules to enhance metadata integration."""

import ast
from typing import List, Optional, Dict, Any
from core.logger import LoggerSetup
from core.metrics import Metrics
from core.types import ExtractionContext, ExtractionResult
from core.extraction.function_extractor import FunctionExtractor
from core.extraction.class_extractor import ClassExtractor
from core.extraction.dependency_analyzer import DependencyAnalyzer
from core.utils import handle_extraction_error, get_source_segment
from core.docstringutils import DocstringUtils, get_node_name

logger = LoggerSetup.get_logger(__name__)


class CodeExtractor:
    """Extracts code elements and metadata from Python source code."""

    def __init__(self, context: Optional[ExtractionContext] = None) -> None:
        """Initialize the CodeExtractor."""
        self.logger = logger
        self.context = context or ExtractionContext()
        self.errors: List[str] = []
        self.metrics_calculator = Metrics()
        self.function_extractor = FunctionExtractor(
            self.context, self.metrics_calculator
        )
        self.class_extractor = ClassExtractor(
            self.context, self.metrics_calculator, self.function_extractor
        )
        self.dependency_analyzer = DependencyAnalyzer(self.context)

    async def extract_code(
        self, source_code: str, context: Optional[ExtractionContext] = None
    ) -> Optional[ExtractionResult]:
        """Extract all code elements and metadata from source code."""
        if context:
            self.context = context
        self.context.source_code = source_code

        try:
            tree = ast.parse(source_code)
            self._add_parent_references(tree)

            # Extract module-level docstring info
            docstring_info = DocstringUtils.extract_docstring_info(tree)
            maintainability_index = self.metrics_calculator.calculate_maintainability_index(
                tree
            )

            result = ExtractionResult(
                module_docstring=docstring_info,
                maintainability_index=maintainability_index,
                classes=[],
                functions=[],
                variables=[],
                constants=[],
                dependencies={},
                errors=[],
            )

            await self._extract_elements(tree, result)

            # Ensure all attributes have default values
            result.classes = result.classes or []
            result.functions = result.functions or []
            result.variables = result.variables or []
            result.constants = result.constants or []
            result.dependencies = result.dependencies or {}
            result.errors = result.errors or []

            return result

        except SyntaxError as e:
            self.logger.error("Syntax error in source code: %s", e)
            return ExtractionResult(
                module_docstring={}, errors=[f"Syntax error: {str(e)}"]
            )
        except Exception as e:
            self.logger.error("Error extracting code: %s", e)
            return ExtractionResult(
                module_docstring={}, errors=[f"Error extracting code: {str(e)}"]
            )

    async def _extract_elements(
        self, tree: ast.AST, result: ExtractionResult
    ) -> None:
        """
        Extract different code elements from the AST.

        Args:
            tree (ast.AST): The AST to extract elements from
            result (ExtractionResult): Result object to store extracted elements
        """
        try:
            result.classes = await self.class_extractor.extract_classes(tree)
            self.logger.debug(f"Extracted {len(result.classes)} classes.")
        except Exception as e:
            handle_extraction_error(
                self.logger, self.errors, "Class extraction", e
            )
            result.errors.extend(self.errors)

        try:
            result.functions = await self.function_extractor.extract_functions(
                tree
            )
            self.logger.debug(f"Extracted {len(result.functions)} functions.")
        except Exception as e:
            handle_extraction_error(
                self.logger, self.errors, "Function extraction", e
            )
            result.errors.extend(self.errors)

        try:
            result.variables = self._extract_variables(tree)
            self.logger.debug(f"Extracted {len(result.variables)} variables.")
        except Exception as e:
            handle_extraction_error(
                self.logger, self.errors, "Variable extraction", e
            )
            result.errors.extend(self.errors)

        try:
            result.constants = self._extract_constants(tree)
            self.logger.debug(f"Extracted {len(result.constants)} constants.")
        except Exception as e:
            handle_extraction_error(
                self.logger, self.errors, "Constant extraction", e
            )
            result.errors.extend(self.errors)

        try:
            result.dependencies = self.dependency_analyzer.analyze_dependencies(
                tree
            )
            self.logger.debug(
                f"Extracted {len(result.dependencies)} dependencies."
            )
        except Exception as e:
            handle_extraction_error(
                self.logger, self.errors, "Dependency extraction", e
            )
            result.errors.extend(self.errors)

    def _extract_variables(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract variables from the AST."""
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        var_info = {
                            "name": target.id,
                            "type": get_node_name(node.annotation)
                            if isinstance(node, ast.AnnAssign)
                            else "Any",
                            "value": get_source_segment(
                                self.context.source_code or "", node.value
                            )
                            if node.value
                            else None,
                        }
                        variables.append(var_info)
        return variables

    def _extract_constants(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract constants from the AST."""
        constants = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        const_info = {
                            "name": target.id,
                            "value": get_source_segment(
                                self.context.source_code or "", node.value
                            )
                            if node.value
                            else None,
                        }
                        constants.append(const_info)
        return constants

    def _add_parent_references(self, node: ast.AST) -> None:
        """Add parent references to AST nodes for accurate context extraction."""
        for child in ast.walk(node):
            for child_node in ast.iter_child_nodes(child):
                setattr(child_node, "parent", child)
                setattr(
                    child_node, "module", getattr(node, "name", None)
                )  # Track module for accurate context

    def _extract_raises(self, node: ast.AST) -> List[Dict[str, str]]:
            """Extract raise statements from function."""
            raises = []
            return raises