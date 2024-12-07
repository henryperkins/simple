"""
Function extraction module for Python source code analysis.

This module provides functionality to extract function definitions and related
metadata from Python source code using the Abstract Syntax Tree (AST). It
processes both synchronous and asynchronous functions, capturing metadata such
as docstrings, metrics, decorators, and exceptions raised.

Classes:
    FunctionExtractor: Handles extraction of functions from Python source code.

Example usage:
    extractor = FunctionExtractor(context, metrics_calculator)
    functions = await extractor.extract_functions(ast_tree)
"""

import ast
from typing import List, Union, Optional

from core.logger import LoggerSetup
from core.metrics import Metrics
from core.types import ExtractedFunction, ExtractionContext
from core.utils import handle_extraction_error, get_source_segment
from core.docstringutils import DocstringUtils

logger = LoggerSetup.get_logger(__name__)

class FunctionExtractor:
    """Handles extraction of functions from Python source code.

    Attributes:
        context (ExtractionContext): The context for extraction operations.
        metrics_calculator (Metrics): The metrics calculator for evaluating function metrics.
        errors (List[Union[str, dict]]): List of errors encountered during extraction.
    """

    def __init__(self, context: ExtractionContext, metrics_calculator: Metrics) -> None:
        """Initialize the function extractor.

        Args:
            context (ExtractionContext): The context for extraction operations.
            metrics_calculator (Metrics): The metrics calculator for evaluating function metrics.
        """
        self.logger = logger
        self.context = context
        self.metrics_calculator = metrics_calculator
        self.errors: List[Union[str, dict]] = []

    async def extract_functions(self, tree: Union[ast.AST, List]) -> List[ExtractedFunction]:
        """Extract function definitions from AST nodes."""
        functions = []
        try:
            if isinstance(tree, list):
                nodes_to_process = tree
            else:
                nodes_to_process = list(ast.walk(tree))

            for node in nodes_to_process:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.logger.debug(f"Found function: {node.name}")
                    if not self.context.include_private and node.name.startswith("_"):
                        continue
                    try:
                        extracted_function = self._process_function(node)
                        if extracted_function:  # Only append if extraction was successful
                            functions.append(extracted_function)
                    except Exception as e:
                        handle_extraction_error(self.logger, self.errors, node.name, e)

            return functions
        except Exception as e:
            self.logger.error(f"Error in extract_functions: {str(e)}", exc_info=True)
            return functions

    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[ExtractedFunction]:
        """Process a function node to extract information."""
        try:
            metadata = DocstringUtils.extract_metadata(node)
            metrics = self.metrics_calculator.calculate_function_metrics(node)
            source_code = self.context.source_code or ""
            
            # Get source segment only once
            function_source = get_source_segment(source_code, node)
            if not function_source:
                self.logger.warning(f"Could not extract source for function {node.name}")
                function_source = ""
            
            return ExtractedFunction(
                name=metadata["name"],
                lineno=metadata["lineno"],
                source=function_source,
                docstring=metadata["docstring_info"]["docstring"],
                metrics=metrics,
                decorators=metadata.get("decorators", []),
                body_summary=function_source,
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
            return None