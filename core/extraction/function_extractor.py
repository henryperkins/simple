"""Function extraction module for Python source code analysis."""

from typing import List, Union
import ast

from core.logger import LoggerSetup
from core.metrics import Metrics
from core.types import ExtractedFunction, ExtractionContext
from core.utils import handle_extraction_error, get_source_segment
from core.docstringutils import DocstringUtils  # Removed unused import

logger = LoggerSetup.get_logger(__name__)


class FunctionExtractor:
    """Handles extraction of functions from Python source code."""

    def __init__(self, context: ExtractionContext, metrics_calculator: Metrics) -> None:
        """Initialize the function extractor."""
        self.logger = logger
        self.context = context
        self.metrics_calculator = metrics_calculator
        self.errors: List[Union[str, dict]] = []

    async def extract_functions(self, tree: ast.AST) -> List[ExtractedFunction]:
        """Extract top-level functions and async functions from the AST."""
        self.logger.info("Starting function extraction")
        functions = []
        try:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.logger.debug(f"Found function: {node.name}")
                    if not self.context.include_private and node.name.startswith("_"):
                        continue
                    try:
                        extracted_function = self._process_function(node)
                        functions.append(extracted_function)
                    except Exception as e:
                        handle_extraction_error(self.logger, self.errors, node.name, e)

            self.logger.info(f"Function extraction completed: {len(functions)} functions extracted")
            return functions
        except Exception as e:
            self.logger.error(f"Error in extract_functions: {str(e)}", exc_info=True)
            return functions

    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> ExtractedFunction:
        """Process and extract information from a function AST node."""
        try:
            metadata = DocstringUtils.extract_metadata(node)
            metrics = self.metrics_calculator.calculate_function_metrics(node)
            source_code = self.context.source_code or ""
            return ExtractedFunction(
                name=metadata["name"],
                lineno=metadata["lineno"],
                source=get_source_segment(source_code, node),
                docstring=metadata["docstring_info"]["docstring"],
                metrics=metrics,
                decorators=metadata.get("decorators", []),
                body_summary=get_source_segment(source_code, node) or "",
                raises=metadata["docstring_info"]["raises"],
                ast_node=node,
            )
        except Exception as e:
            self.logger.error(f"Failed to process function {node.name}: {e}")
            self.errors.append({
                'name': node.name,
                'lineno': getattr(node, 'lineno', 'Unknown'),
                'error': str(e)
            })
            return ExtractedFunction(
                name=node.name,
                lineno=getattr(node, 'lineno', 0),
                source="",
                docstring="",
                metrics={},
                decorators=[],
                body_summary="",
                raises=[],
                ast_node=node,
            )
