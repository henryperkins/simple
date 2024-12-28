"""
Function extraction module for Python source code analysis.
"""

import ast
import uuid
from typing import Optional, List, Dict, Union, Any
from pathlib import Path

from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.types.base import (
    ExtractedFunction,
    ExtractedArgument,
)
from core.extraction.extraction_utils import extract_decorators, get_node_name
from core.exceptions import ExtractionError
from utils import log_and_raise_error


class BaseExtractor:
    """Base class for shared extraction methods."""

    def __init__(self, context, correlation_id: Optional[str] = None) -> None:
        """Initialize the base extractor."""
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = LoggerSetup.get_logger(
            __name__, correlation_id=self.correlation_id
        )
        self.context = context
        self.errors: List[str] = []

    def _should_process_function(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> bool:
        """Determine if a function should be processed based on context settings."""
        if not self.context.include_private and node.name.startswith("_"):
            self.logger.debug(
                f"Skipping private function: {node.name} (include_private=False)"
            )
            return False
        if (
            not self.context.include_magic
            and node.name.startswith("__")
            and node.name.endswith("__")
        ):
            self.logger.debug(
                f"Skipping magic function: {node.name} (include_magic=False)"
            )
            return False

        if not self.context.include_nested and self._is_nested_function(node):
            self.logger.debug(
                f"Skipping nested function: {node.name} (include_nested=False)"
            )
            return False
        return True

    def _is_nested_function(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> bool:
        """Check if a function is nested inside another function."""
        if not hasattr(self.context, "tree") or self.context.tree is None:
            return False

        for parent in ast.walk(self.context.tree):
            if isinstance(
                parent, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and node in ast.walk(parent):
                if parent != node:
                    return True
        return False

    def _extract_arguments(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> List[ExtractedArgument]:
        """Extract argument details from a function definition."""
        args = []
        for arg in node.args.args:
            arg_type = "typing.Any"
            if arg.annotation and isinstance(arg.annotation, ast.AST):
                arg_type = get_node_name(arg.annotation) or "typing.Any"
            args.append(
                ExtractedArgument(
                    name=arg.arg,
                    type=arg_type,
                    description="",  # Add description extraction if needed
                )
            )
        return args

    def _extract_type_hints(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> Dict[str, str]:
        """Extract type hints from function parameters and return value."""
        type_hints = {}
        for arg in node.args.args:
            if arg.annotation and isinstance(arg.annotation, ast.AST):
                type_hints[arg.arg] = get_node_name(arg.annotation)
        if node.returns and isinstance(node.returns, ast.AST):
            type_hints["return"] = get_node_name(node.returns)
        return type_hints

    def _analyze_complexity_warnings(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> List[str]:
        """Analyze and return specific complexity warnings."""
        warnings = []
        # Check nesting depth
        max_depth = 0
        current_depth = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While, ast.If, ast.With)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                current_depth = 0

        if max_depth > 3:
            warnings.append(f"High nesting depth ({max_depth} levels)")

        # Count number of branches
        branch_count = sum(
            1 for _ in ast.walk(node) if isinstance(_, (ast.If, ast.For, ast.While))
        )
        if branch_count > 10:
            warnings.append(f"High number of branches ({branch_count})")

        return warnings

    def _extract_examples_from_docstring(self, docstring: str) -> List[str]:
        """Extract usage examples from a docstring."""
        examples = []
        if docstring:
            lines = docstring.splitlines()
            example_start = False
            for line in lines:
                if line.strip().startswith("Example:"):
                    example_start = True
                    continue
                if example_start and line.strip():
                    examples.append(line.strip())
                else:
                    example_start = False
        return examples

    def _extract_dependencies(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> Dict[str, set[str]]:
        """Extract dependencies from a function node."""
        dependencies = {}
        if self.context.dependency_analyzer:
            dependencies = self.context.dependency_analyzer.analyze_dependencies(node)
        return dependencies

    def _extract_imports(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> List[str]:
        """Extract imports from a function node."""
        imports = []
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                if isinstance(child, ast.Import):
                    for alias in child.names:
                        imports.append(alias.name)
                elif child.module:
                    imports.append(child.module)
        return imports


class FunctionExtractor(BaseExtractor):
    """Handles extraction of functions from Python source code."""

    def __init__(
        self,
        context,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Initialize the function extractor."""
        super().__init__(context, correlation_id)

    async def extract_functions(
        self, nodes: Union[ast.AST, List[ast.AST]], module_metrics: Any
    ) -> List[ExtractedFunction]:
        """Extract function definitions from AST nodes."""
        functions: List[ExtractedFunction] = []

        # Ensure we process all nodes
        nodes_to_process = [nodes] if isinstance(nodes, ast.AST) else nodes
        self.logger.info("Starting function extraction.")
        for node in ast.walk(
            nodes_to_process[0] if nodes_to_process else ast.Module(body=[])
        ):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self._should_process_function(node):
                    try:
                        extracted_function = await self._process_function(
                            node, module_metrics
                        )
                        if extracted_function:
                            functions.append(extracted_function)
                            self.logger.debug(
                                f"Extracted function: {extracted_function.name}, Arguments: {[arg.name for arg in extracted_function.args]}, Return Type: {extracted_function.returns['type']}"
                            )
                            if self.context.metrics_collector:
                                self.context.metrics_collector.update_scan_progress(
                                    self.context.module_name
                                    or Path(
                                        getattr(self.context.base_path, "name", "")
                                    ).stem,
                                    "function",
                                    node.name,
                                )
                    except Exception as e:
                        log_and_raise_error(
                            self.logger,
                            e,
                            ExtractionError,
                            f"Error extracting function {node.name}",
                            self.correlation_id,
                            function_name=node.name,
                        )

        self.logger.info(
            f"Function extraction completed. Total functions extracted: {len(functions)}"
        )
        return functions

    async def _process_function(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], module_metrics: Any
    ) -> Optional[ExtractedFunction]:
        """Process a single function definition."""
        try:
            source_code = self.context.get_source_code()
            if not source_code:
                log_and_raise_error(
                    self.logger,
                    ExtractionError("Source code is not available in the context"),
                    ExtractionError,
                    "Source code is not available in the context",
                    self.correlation_id,
                    function_name=node.name,
                )

            docstring = ast.get_docstring(node) or ""
            decorators = extract_decorators(node)
            arguments = self._extract_arguments(node)
            return_type = (
                get_node_name(node.returns)
                if node.returns and isinstance(node.returns, ast.AST)
                else "typing.Any"
            )
            is_async = isinstance(node, ast.AsyncFunctionDef)

            extracted_fn = ExtractedFunction(
                name=node.name,
                lineno=node.lineno,
                source=ast.unparse(node),
                docstring=docstring,
                decorators=decorators,
                args=arguments,
                returns={"type": return_type, "description": ""},
                is_async=is_async,
                ast_node=node,
                dependencies=self._extract_dependencies(node),
                complexity_warnings=self._analyze_complexity_warnings(node),
            )

            if docstring and hasattr(self.context, "docstring_processor"):
                extracted_fn._docstring_info = self.context.docstring_processor.parse(
                    docstring
                )

            # Use module-level metrics for function-level metrics
            extracted_fn.metrics = module_metrics.__dict__.copy()
            extracted_fn.metrics["total_functions"] = 1
            extracted_fn.metrics["scanned_functions"] = (
                1 if extracted_fn.get_docstring_info() else 0
            )

            return extracted_fn

        except Exception as e:
            log_and_raise_error(
                self.logger,
                e,
                ExtractionError,
                f"Error processing function {node.name}",
                self.correlation_id,
                function_name=node.name,
            )
            return None
