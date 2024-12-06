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
from typing import List, Union

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
        """Extract top-level functions and async functions from the AST.

        Args:
            tree: Either an AST node or a list of nodes to extract functions from.

        Returns:
            List[ExtractedFunction]: A list of extracted function information.
        """
        self.logger.info("Starting function extraction")
        functions = []
        try:
            # Handle both single nodes and lists of nodes
            if isinstance(tree, list):
                nodes_to_process = tree  # Use the list directly
            else:
                nodes_to_process = list(ast.walk(tree))  # Convert AST walk iterator to list

            # Process each node
            for node in nodes_to_process:
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
        """Process and extract information from a function AST node.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function node to process.

        Returns:
            ExtractedFunction: The extracted function information.

        Raises:
            Exception: If an error occurs during function processing.
        """
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