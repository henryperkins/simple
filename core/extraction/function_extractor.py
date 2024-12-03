"""Function extraction module.

This module provides functionality to extract function definitions from Python source code.
It uses the Abstract Syntax Tree (AST) to parse and analyze the code, extracting information
about functions, their arguments, return types, and other metadata.
"""

import ast
from typing import List, Optional, Dict, Any, Union, Set
from core.logger import LoggerSetup
from core.metrics import Metrics
from .types import ExtractedFunction, ExtractedArgument, ExtractionContext
from .utils import ASTUtils

logger = LoggerSetup.get_logger(__name__)

class FunctionExtractor:
    """Handles extraction of functions from Python source code.

    Attributes:
        context (ExtractionContext): The context for extraction, including configuration options.
        metrics_calculator (Metrics): An instance of the Metrics class for calculating code metrics.
        errors (List[str]): A list to store error messages encountered during extraction.
    """

    def __init__(self, context: ExtractionContext, metrics_calculator: Metrics):
        """Initialize the FunctionExtractor.

        Args:
            context (ExtractionContext): The extraction context containing settings and configurations.
            metrics_calculator (Metrics): An instance for calculating metrics related to code complexity.
        """
        self.logger = logger
        self.context = context
        self.metrics_calculator = metrics_calculator
        self.ast_utils = ASTUtils()
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
                            self._handle_extraction_error(node.name, e)

            self.logger.info(f"Function extraction completed: {len(functions)} functions extracted")
            return functions

        except Exception as e:
            self.logger.error(f"Error in extract_functions: {str(e)}", exc_info=True)
            return functions

    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> ExtractedFunction:
        """Process a function definition node.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            ExtractedFunction: An object containing detailed information about the function.
        """
        self.logger.debug(f"Processing function: {node.name}")
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(f"Expected FunctionDef or AsyncFunctionDef, got {type(node)}")

        try:
            metrics = self._calculate_function_metrics(node)
            docstring = ast.get_docstring(node)

            extracted_function = ExtractedFunction(
                name=node.name,
                lineno=node.lineno,
                source=self.ast_utils.get_source_segment(node, self.context.include_source),
                docstring=docstring,
                metrics=metrics,
                dependencies=self._extract_dependencies(node),
                args=self._get_function_args(node),
                return_type=self._get_return_type(node),
                is_method=self._is_method(node),
                is_async=isinstance(node, ast.AsyncFunctionDef),
                is_generator=self._is_generator(node),
                is_property=self._is_property(node),
                body_summary=self._get_body_summary(node),
                raises=self._extract_raises(node),
                ast_node=node
            )
            self.logger.debug(f"Completed processing function: {node.name}")
            return extracted_function
        except Exception as e:
            self.logger.error(f"Failed to process function {node.name}: {e}", exc_info=True)
            raise

    def _get_function_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[ExtractedArgument]:
        """Extract function arguments.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            List[ExtractedArgument]: A list of ExtractedArgument objects representing the function's arguments.
        """
        self.logger.debug(f"Extracting arguments for function: {node.name}")
        args = []
        try:
            for arg in node.args.args:
                arg_name = arg.arg
                type_hint = self.ast_utils.get_name(arg.annotation) if arg.annotation else None
                default_value = None
                is_required = True

                if node.args.defaults:
                    default_index = len(node.args.args) - len(node.args.defaults)
                    if node.args.args.index(arg) >= default_index:
                        default_value = self.ast_utils.get_name(
                            node.args.defaults[node.args.args.index(arg) - default_index]
                        )
                        is_required = False

                args.append(ExtractedArgument(
                    name=arg_name,
                    type_hint=type_hint,
                    default_value=default_value,
                    is_required=is_required
                ))
                self.logger.debug(f"Extracted argument: {arg_name}, type_hint: {type_hint}, default_value: {default_value}")

            # Handle keyword-only and positional-only arguments
            # Additional logic can be added here to handle these cases

        except Exception as e:
            self.logger.error(f"Error extracting arguments for function {node.name}: {e}", exc_info=True)

        return args

    def _get_return_type(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[str]:
        """Get the return type annotation.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            Optional[str]: The return type annotation as a string, or None if not specified.
        """
        self.logger.debug(f"Extracting return type for function: {node.name}")
        if node.returns:
            try:
                return_type = self.ast_utils.get_name(node.returns)
                if isinstance(node, ast.AsyncFunctionDef) and not return_type.startswith('Coroutine'):
                    return_type = f'Coroutine[Any, Any, {return_type}]'
                self.logger.debug(f"Return type for function {node.name}: {return_type}")
                return return_type
            except Exception as e:
                self.logger.error(f"Error getting return type for function {node.name}: {e}", exc_info=True)
                return 'Unknown'
        return None

    def _get_body_summary(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        """Generate a summary of the function body.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            str: A summary of the function body or a placeholder if unavailable.
        """
        self.logger.debug(f"Generating body summary for function: {node.name}")
        return self.ast_utils.get_source_segment(node) or "No body summary available"

    def _extract_raises(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
        """Extract raised exceptions from function body.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            List[str]: A list of exception names that the function raises.
        """
        self.logger.debug(f"Extracting raised exceptions for function: {node.name}")
        raises = set()
        try:
            for child in ast.walk(node):
                if isinstance(child, ast.Raise) and child.exc:
                    exc_name = self._get_exception_name(child.exc)
                    if exc_name:
                        raises.add(exc_name)
            self.logger.debug(f"Raised exceptions for function {node.name}: {raises}")
        except Exception as e:
            self.logger.error(f"Error extracting raises: {e}", exc_info=True)
        return list(raises)

    def _get_exception_name(self, node: ast.AST) -> Optional[str]:
        """Get the name of an exception node.

        Args:
            node (ast.AST): The AST node representing the exception.

        Returns:
            Optional[str]: The name of the exception or None if it cannot be determined.
        """
        try:
            if isinstance(node, ast.Call):
                return self.ast_utils.get_name(node.func)
            elif isinstance(node, (ast.Name, ast.Attribute)):
                return self.ast_utils.get_name(node)
            return "Exception"
        except Exception:
            return None

    def _is_method(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if a function is a method.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            bool: True if the function is a method, False otherwise.
        """
        self.logger.debug(f"Checking if function is a method: {node.name}")
        return isinstance(getattr(node, 'parent', None), ast.ClassDef)

    def _is_generator(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if a function is a generator.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            bool: True if the function is a generator, False otherwise.
        """
        self.logger.debug(f"Checking if function is a generator: {node.name}")
        return any(isinstance(child, (ast.Yield, ast.YieldFrom)) for child in ast.walk(node))

    def _is_property(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if a function is a property.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            bool: True if the function is a property, False otherwise.
        """
        self.logger.debug(f"Checking if function is a property: {node.name}")
        return any(
            isinstance(decorator, ast.Name) and decorator.id == 'property'
            for decorator in node.decorator_list
        )

    def _calculate_function_metrics(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        """Calculate metrics for a function.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            Dict[str, Any]: A dictionary containing metrics such as cyclomatic complexity and parameter count.
        """
        self.logger.debug(f"Calculating metrics for function: {node.name}")
        try:
            metrics = {
                'cyclomatic_complexity': self.metrics_calculator.calculate_cyclomatic_complexity(node),
                'cognitive_complexity': self.metrics_calculator.calculate_cognitive_complexity(node),
                'maintainability_index': self.metrics_calculator.calculate_maintainability_index(node),
                'parameter_count': len(node.args.args),
                'return_complexity': self._calculate_return_complexity(node),
                'is_async': isinstance(node, ast.AsyncFunctionDef)
            }
            self.logger.debug(f"Metrics for function {node.name}: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating metrics for function {node.name}: {e}", exc_info=True)
            return {}

    def _calculate_return_complexity(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
        """Calculate the complexity of return statements.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            int: The number of return statements in the function.
        """
        self.logger.debug(f"Calculating return complexity for function: {node.name}")
        try:
            return_complexity = sum(1 for _ in ast.walk(node) if isinstance(_, ast.Return))
            self.logger.debug(f"Return complexity for function {node.name}: {return_complexity}")
            return return_complexity
        except Exception as e:
            self.logger.error(f"Error calculating return complexity: {e}", exc_info=True)
            return 0

    def _extract_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Extract dependencies from a node.

        Args:
            node (ast.AST): The AST node representing the function definition.

        Returns:
            Dict[str, Set[str]]: A dictionary containing sets of dependencies categorized by type.
        """
        self.logger.debug(f"Extracting dependencies for function: {node.name}")
        # This would typically call into the DependencyAnalyzer
        # Simplified version for function-level dependencies
        return {'imports': set(), 'calls': set(), 'attributes': set()}

    def _handle_extraction_error(self, function_name: str, error: Exception) -> None:
        """Handle function extraction errors.

        Args:
            function_name (str): The name of the function being processed when the error occurred.
            error (Exception): The exception that was raised.
        """
        error_msg = f"Failed to process function {function_name}: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        self.errors.append(error_msg)
