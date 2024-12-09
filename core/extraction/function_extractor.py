"""
Function extraction module for Python source code analysis.

This module provides functionality to extract function definitions and related
metadata from Python source code using the Abstract Syntax Tree (AST).
"""

import ast
from typing import List, Dict, Any, Optional, Union
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.metrics import Metrics
from core.types import ExtractedFunction, ExtractedArgument, ExtractionContext, MetricData
from utils import handle_extraction_error, get_source_segment, get_node_name, NodeNameVisitor


class FunctionExtractor:
    """Handles extraction of functions from Python source code."""

    def __init__(
        self,
        context: ExtractionContext,
        metrics_calculator: Metrics,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Initialize the function extractor.

        Args:
            context (ExtractionContext): The context for extraction, including settings and source code.
            metrics_calculator (Metrics): The metrics calculator for analyzing function complexity.
            correlation_id (Optional[str]): An optional correlation ID for logging purposes.
        """
        self.logger = CorrelationLoggerAdapter(LoggerSetup.get_logger(__name__), correlation_id)
        self.context = context
        self.metrics_calculator = metrics_calculator
        self.errors: List[str] = []  # Changed to List[str]

    async def extract_functions(
        self,
        nodes: Union[ast.AST, List[ast.AST]]
    ) -> List[ExtractedFunction]:
        """Extract function definitions from AST nodes.

        Args:
            nodes (Union[ast.AST, List[ast.AST]]): The AST nodes to process.

        Returns:
            List[ExtractedFunction]: A list of extracted function metadata.
        """
        functions: List[ExtractedFunction] = []

        # Support for either a list of nodes or a single node
        nodes_to_process = [nodes] if isinstance(nodes, ast.AST) else nodes

        try:
            for node in nodes_to_process:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not self._should_process_function(node):
                        self.logger.debug(f"Skipping function: {node.name}")
                        continue

                    try:
                        extracted_function = await self._process_function(node)
                        if extracted_function:
                            functions.append(extracted_function)
                            self.logger.info(f"Successfully extracted function: {node.name}")
                    except Exception as e:
                        handle_extraction_error(self.logger, self.errors, f"Function {node.name}", e)

            return functions
        except Exception as e:
            self.logger.error(f"Error extracting functions: {e}", exc_info=True)
            return []

    async def _process_function(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> Optional[ExtractedFunction]:
        """Process a function node to extract information.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function node to process.

        Returns:
            Optional[ExtractedFunction]: The extracted function metadata, or None if processing fails.
        """
        try:
            # Extract basic information
            docstring = ast.get_docstring(node) or ""
            source = get_source_segment(self.context.source_code or "", node) or ""

            # Extract function components
            args = [
                ExtractedArgument(
                    name=arg.arg,
                    type=get_node_name(arg.annotation),
                    default_value="None" if arg.arg in (a.arg for a in node.args.defaults) else None,
                    is_required=arg.arg not in (a.arg for a in node.args.defaults)
                ) for arg in node.args.args if isinstance(arg, ast.arg)
            ]
            return_type = get_node_name(node.returns) or "Any"
            decorators = [NodeNameVisitor().visit(decorator) for decorator in node.decorator_list]

            # Calculate metrics
            metrics = MetricData()
            extracted_function = ExtractedFunction(
                name=node.name,
                lineno=node.lineno,
                source=source,
                docstring=docstring,
                metrics=metrics,
                dependencies=self.context.dependency_analyzer.extract_dependencies(node),
                decorators=decorators,
                complexity_warnings=[],  # Populate as needed
                ast_node=node,
                args=args,
                returns={"type": return_type, "description": ""},
                is_async=isinstance(node, ast.AsyncFunctionDef),
                docstring_info=None  # Set if available
            )

            # Calculate and assign metrics
            extracted_function.metrics = self.metrics_calculator.calculate_metrics_for_function(extracted_function)

            return extracted_function
        except Exception as e:
            self.logger.error(f"Failed to process function {node.name}: {e}", exc_info=True)
            return None

    def _should_process_function(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> bool:
        """Determine whether the function should be processed.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function node to check.

        Returns:
            bool: True if the function should be processed, False otherwise.
        """
        return not (
            (not self.context.include_private and node.name.startswith('_')) or
            (not self.context.include_magic and node.name.startswith('__'))
        )
