import ast
from typing import List, Union, Dict, Any
from core.logger import LoggerSetup
from core.metrics import Metrics
from core.types import ExtractedFunction, ExtractionContext
from utils import handle_extraction_error, get_source_segment
from docstringutils import DocstringUtils
from core.dependency_analyzer import extract_dependencies_from_node

logger = LoggerSetup.get_logger(__name__)

class FunctionExtractor:
    """Handles extraction of functions from Python source code."""

    def __init__(self, context: ExtractionContext, metrics_calculator: Metrics):
        """Initialize the FunctionExtractor.

        Args:
            context (ExtractionContext): The extraction context containing settings and configurations.
            metrics_calculator (Metrics): An instance for calculating metrics related to code complexity.
        """
        self.logger = logger
        self.context = context
        self.metrics_calculator = metrics_calculator
        self.errors: List[str] = []
        self.logger.debug("Initialized FunctionExtractor")

    def extract_functions(self, tree: ast.AST) -> List[ExtractedFunction]:
        """Extract top-level functions and async functions from the AST.

        Args:
            tree (ast.AST): The root of the AST representing the parsed Python source code.

        Returns:
            List[ExtractedFunction]: A list of ExtractedFunction objects containing information about each function.
        """
        self.logger.info("Starting function extraction")
        functions = []
        try:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.logger.debug(f"Found {'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}function: {node.name}")

                    parent = getattr(node, 'parent', None)
                    if isinstance(parent, ast.Module):
                        if not self.context.include_private and node.name.startswith('_'):
                            self.logger.debug(f"Skipping private function: {node.name}")
                            continue

                        try:
                            extracted_function = self._process_function(node)
                            functions.append(extracted_function)
                            self.logger.debug(f"Extracted function: {extracted_function.name}")
                        except Exception as e:
                            handle_extraction_error(self.logger, self.errors, node.name, e)

            self.logger.info(f"Function extraction completed: {len(functions)} functions extracted")
            return functions

        except Exception as e:
            self.logger.error(f"Error in extract_functions: {str(e)}", exc_info=True)
            return functions

    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> ExtractedFunction:
        """
        Process a function definition node to extract relevant information and metrics.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function.

        Returns:
            ExtractedFunction: An object containing the extracted function details and metrics.
        """
        self.logger.debug(f"Processing function: {node.name}")
        try:
            # Extract metadata and docstring info
            metadata = DocstringUtils.extract_metadata(node)
            docstring_info = metadata['docstring_info']

            # Calculate metrics for the function
            metrics = self.metrics_calculator.calculate_function_metrics(node)
            cognitive_complexity = metrics.get('cognitive_complexity')
            halstead_metrics = metrics.get('halstead_metrics')

            # Create the ExtractedFunction object
            extracted_function = ExtractedFunction(
                name=metadata['name'],
                lineno=metadata['lineno'],
                source=get_source_segment(self.context.source_code, node),
                docstring=docstring_info['docstring'],
                metrics=metrics,
                cognitive_complexity=cognitive_complexity,
                halstead_metrics=halstead_metrics,
                dependencies=extract_dependencies_from_node(node),
                args=docstring_info['args'],
                return_type=docstring_info['returns']['type'],
                is_method=self._is_method(node),
                is_async=isinstance(node, ast.AsyncFunctionDef),
                is_generator=self._is_generator(node),
                is_property=self._is_property(node),
                body_summary=self._get_body_summary(node),
                raises=docstring_info['raises'],
                ast_node=node
            )
            self.logger.debug(f"Completed processing function: {node.name}")
            return extracted_function
        except Exception as e:
            self.logger.error(f"Failed to process function {node.name}: {e}", exc_info=True)
            raise

    def _get_body_summary(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        """Generate a summary of the function body."""
        self.logger.debug(f"Generating body summary for function: {node.name}")
        return get_source_segment(self.context.source_code, node) or "No body summary available"

    def _is_method(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if a function is a method."""
        self.logger.debug(f"Checking if function is a method: {node.name}")
        return isinstance(getattr(node, 'parent', None), ast.ClassDef)

    def _is_generator(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if a function is a generator."""
        self.logger.debug(f"Checking if function is a generator: {node.name}")
        return any(isinstance(child, (ast.Yield, ast.YieldFrom)) for child in ast.walk(node))

    def _is_property(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if a function is a property."""
        self.logger.debug(f"Checking if function is a property: {node.name}")
        return any(
            isinstance(decorator, ast.Name) and decorator.id == 'property'
            for decorator in node.decorator_list
        )