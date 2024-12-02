"""
code_extraction.py - Unified code extraction module

Provides comprehensive code analysis and extraction functionality for Python
source code, including class and function extraction, metrics calculation, and
dependency analysis.
"""

import ast
import re
import sys
import importlib
import importlib.util
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Dict, Any, List, Optional, Set, Union
import types

from core.logger import LoggerSetup
from core.metrics import Metrics, MetricsError
from exceptions import ExtractionError

# Initialize the logger
logger = LoggerSetup.get_logger(__name__)

# Simple caching mechanism using a dictionary
_class_cache = {}

@dataclass
class ExtractedArgument:
    """Represents a function argument."""
    name: str
    type_hint: Optional[str] = None
    default_value: Optional[str] = None
    is_required: bool = True


@dataclass
class ExtractionContext:
    """Context for extraction operations."""
    file_path: Optional[str] = None
    module_name: Optional[str] = None
    import_context: Optional[Dict[str, Set[str]]] = None
    metrics_enabled: bool = True
    include_source: bool = True
    max_line_length: int = 100
    include_private: bool = False
    include_metrics: bool = True
    resolve_external_types: bool = True


@dataclass
class CodeMetadata:
    """Represents metadata extracted from code analysis."""
    required_imports: Set[str]
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    dependencies: Set[str]

    @classmethod
    def create_empty(cls) -> 'CodeMetadata':
        """Create an empty metadata instance."""
        return cls(
            required_imports=set(),
            classes=[],
            functions=[],
            dependencies=set()
        )


@dataclass
class ExtractedElement:
    """Base class for extracted code elements."""
    name: str
    lineno: int
    source: Optional[str] = None
    docstring: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    decorators: List[str] = field(default_factory=list)
    complexity_warnings: List[str] = field(default_factory=list)


@dataclass
class ExtractedFunction(ExtractedElement):
    """Represents an extracted function."""
    return_type: Optional[str] = None
    is_method: bool = False
    is_async: bool = False
    is_generator: bool = False
    is_property: bool = False
    body_summary: str = ""
    args: List[ExtractedArgument] = field(default_factory=list)
    raises: List[str] = field(default_factory=list)
    ast_node: Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = None  # Store the AST node


@dataclass
class ExtractionResult:
    """Contains the complete extraction results."""
    classes: List['ExtractedClass'] = field(default_factory=list)
    functions: List[ExtractedFunction] = field(default_factory=list)
    variables: List[Dict[str, Any]] = field(default_factory=list)
    module_docstring: Optional[str] = None
    imports: Dict[str, Set[str]] = field(default_factory=dict)
    constants: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute value safely."""
        return getattr(self, key, default)


@dataclass
class ExtractedClass(ExtractedElement):
    """Represents an extracted class."""
    bases: List[str] = field(default_factory=list)
    methods: List[ExtractedFunction] = field(default_factory=list)
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    is_exception: bool = False
    instance_attributes: List[Dict[str, Any]] = field(default_factory=list)
    metaclass: Optional[str] = None
    ast_node: Optional[ast.ClassDef] = None  # Store the AST node


class CodeExtractor:
    """
    Extracts code elements and metadata from Python source code.

    Attributes:
        context (ExtractionContext): Context for extraction operations.
        errors (List[str]): List of errors encountered during extraction.
        metrics_calculator (Metrics): Metrics calculator instance.
        metadata (CodeMetadata): Metadata extracted from code analysis.
    """

    def __init__(self, context: Optional[ExtractionContext] = None) -> None:
        """Initialize the code extractor."""
        self.logger = logger  # Use the module-level logger
        self.context = context or ExtractionContext()
        self._module_ast: Optional[ast.Module] = None
        self._current_class: Optional[ast.ClassDef] = None
        self.errors: List[str] = []
        self.metrics_calculator = Metrics()  # Use your Metrics class
        self.metadata = CodeMetadata.create_empty()
        self.logger.debug(f"Processing in {__name__}")

    def _preprocess_code(self, source_code: str) -> str:
        """
        Preprocess source code to handle timestamps and other special cases.

        Args:
            source_code (str): The source code to preprocess.

        Returns:
            str: The preprocessed source code.
        """
        try:
            # Convert timestamps with leading zeros to string literals. Handles milliseconds/microseconds.
            pattern = r'\$\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2}(?:\.\d+)?\$'
            processed_code = re.sub(pattern, r'"\g<0>"', source_code)

            self.logger.debug("Preprocessed source code to handle timestamps.")
            return processed_code

        except Exception as e:
            self.logger.error(f"Error preprocessing code: {e}")
            return source_code

    def extract_code(self, source_code: str) -> Optional[ExtractionResult]:
        """
        Extract all code elements and metadata, continuing even if some parts fail.

        Args:
            source_code (str): The source code to extract.

        Returns:
            Optional[ExtractionResult]: The extraction result or None if critical extraction fails.
        """
        try:
            processed_source_code = self._preprocess_code(source_code)
            tree = ast.parse(processed_source_code)
            self._module_ast = tree
            self._add_parents(tree)

            # Initialize result with empty collections
            result = ExtractionResult(
                classes=[],
                functions=[],
                variables=[],
                module_docstring=ast.get_docstring(tree),
                imports={},
                constants=[],
                errors=[],
                metrics={},
                dependencies={}
            )

            # Try to analyze dependencies, but continue if it fails
            try:
                dependencies = self.metrics_calculator.analyze_dependencies(tree)
                result.dependencies = dependencies
                self.logger.debug(f"Module dependencies: {dependencies}")
            except Exception as e:
                error_msg = f"Dependency analysis failed: {str(e)}"
                self.logger.warning(error_msg)
                result.errors.append(error_msg)

            # Extract different code elements, continuing if individual extractions fail
            try:
                result.classes = self._extract_classes(tree)
            except Exception as e:
                error_msg = f"Class extraction failed: {str(e)}"
                self.logger.warning(error_msg)
                result.errors.append(error_msg)

            try:
                result.functions = self._extract_functions(tree)
            except Exception as e:
                error_msg = f"Function extraction failed: {str(e)}"
                self.logger.warning(error_msg)
                result.errors.append(error_msg)

            try:
                result.variables = self._extract_variables(tree)
            except Exception as e:
                error_msg = f"Variable extraction failed: {str(e)}"
                self.logger.warning(error_msg)
                result.errors.append(error_msg)

            try:
                result.constants = self._extract_constants(tree)
            except Exception as e:
                error_msg = f"Constant extraction failed: {str(e)}"
                self.logger.warning(error_msg)
                result.errors.append(error_msg)

            # Try to extract imports, but don't fail if it doesn't work
            try:
                result.imports = self._extract_imports(tree)
            except Exception as e:
                error_msg = f"Import extraction failed: {str(e)}"
                self.logger.warning(error_msg)
                result.errors.append(error_msg)
                result.imports = {'stdlib': set(), 'local': set(), 'third_party': set()}

            # Calculate metrics if enabled
            if self.context.metrics_enabled:
                try:
                    self._calculate_and_add_metrics(result, tree)
                except Exception as e:
                    error_msg = f"Metrics calculation failed: {str(e)}"
                    self.logger.warning(error_msg)
                    result.errors.append(error_msg)

            return result

        except SyntaxError as e:
            self.logger.error(f"Syntax error in source code: {str(e)}")
            return ExtractionResult(errors=[f"Syntax error: {str(e)}"])

        except Exception as e:
            self.logger.error(f"Extraction failed: {str(e)}")
            return ExtractionResult(errors=[f"Failed to extract code: {str(e)}"])

    def _calculate_and_add_metrics(self, result: ExtractionResult, tree: ast.AST) -> None:
        """Calculate and add metrics to the extraction result."""

        if not self.context.metrics_enabled:  # Early exit if metrics are disabled
            return

        try:
            for cls in result.classes:
                cls.metrics.update(self.metrics_calculator.calculate_halstead_metrics(cls.ast_node))
                cls.metrics.update(self._calculate_class_metrics(cls.ast_node))  # Reuse existing method
                cls.complexity_warnings.extend(self._get_complexity_warnings(cls.metrics))

                for method in cls.methods:
                    method.metrics.update(self.metrics_calculator.calculate_halstead_metrics(method.ast_node))
                    method.metrics.update(self._calculate_function_metrics(method.ast_node))
                    method.complexity_warnings.extend(self._get_complexity_warnings(method.metrics))

            # Module-level metrics
            result.metrics.update(self._calculate_module_metrics(tree))

            for func in result.functions:
                func.metrics.update(self.metrics_calculator.calculate_halstead_metrics(func.ast_node))
                func.metrics.update(self._calculate_function_metrics(func.ast_node))
                func.complexity_warnings.extend(self._get_complexity_warnings(func.metrics))

        except Exception as e:
            self.logger.error(f"Error calculating and adding metrics: {e}")
            self.errors.append(str(e))  # Add error to result.errors

    def analyze_dependencies(self, node: ast.AST, module_name: str = None) -> Dict[str, Set[str]]:
        """
        Analyzes module dependencies, including circular dependency detection.

        Args:
            node (ast.AST): The AST node to analyze.
            module_name (str, optional): The name of the module being analyzed.

        Returns:
            Dict[str, Set[str]]: A dictionary of module dependencies.
        """
        deps: Dict[str, Set[str]] = defaultdict(set)
        self.module_name = module_name

        try:
            for subnode in ast.walk(node):
                if isinstance(subnode, (ast.Import, ast.ImportFrom)):
                    try:
                        if isinstance(subnode, ast.Import):
                            for name in subnode.names:
                                # Simply add to third_party without verification
                                deps['third_party'].add(name.name)
                        elif isinstance(subnode, ast.ImportFrom) and subnode.module:
                            # Add the base module import
                            if subnode.names[0].name == '*':
                                self.logger.warning(f"Star import found: from {subnode.module} import *")
                                deps['third_party'].add(subnode.module)
                            else:
                                deps['third_party'].add(subnode.module)
                    except Exception as e:
                        self.logger.debug(f"Non-critical error processing import: {e}")
                        continue

            return dict(deps)

        except Exception as e:
            self.logger.warning(f"Dependency analysis failed gracefully: {str(e)}")
            return {'stdlib': set(), 'third_party': set(), 'local': set()}
        
    def _calculate_module_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """Calculate metrics for the entire module."""
        if not self.context.metrics_enabled:
            return {}

        try:
            # Estimate total lines by counting the lines in the source code
            if hasattr(tree, 'body') and tree.body:
                last_node = tree.body[-1]
                total_lines = last_node.end_lineno if hasattr(last_node, 'end_lineno') else last_node.lineno
            else:
                total_lines = 0

            complexity = self.metrics_calculator.calculate_complexity(tree)
            maintainability = self.metrics_calculator.calculate_maintainability_index(tree)
            halstead = self.metrics_calculator.calculate_halstead_metrics(tree)
            return {
                'total_lines': total_lines,
                'complexity': complexity,
                'maintainability': maintainability,
                'halstead': halstead
            }
        except Exception as e:
            self.logger.error(f"Error calculating module metrics: {e}")
            self.errors.append(str(e))
            return {}

    def _calculate_class_metrics(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Calculate metrics for a class."""
        if not self.context.metrics_enabled:
            return {}

        try:
            complexity = self.metrics_calculator.calculate_complexity(node)
            maintainability = self.metrics_calculator.calculate_maintainability_index(node)
            return {
                'method_count': len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]),
                'complexity': complexity,
                'maintainability': maintainability,
                'inheritance_depth': self._calculate_inheritance_depth(node)
            }
        except Exception as e:
            self.logger.error("Error calculating class metrics: %s", str(e))
            return {'error': str(e)}

    def _calculate_function_metrics(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        """Calculate metrics for a function."""
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.logger.error(f"Provided node is not a function definition: {type(node)}")
            return {}  # Return empty dictionary on error

        try:
            cyclomatic_complexity = self.metrics_calculator.calculate_cyclomatic_complexity(node)
            cognitive_complexity = self.metrics_calculator.calculate_cognitive_complexity(node)
            maintainability_index = self.metrics_calculator.calculate_maintainability_index(node)
            return {
                'cyclomatic_complexity': cyclomatic_complexity,
                'cognitive_complexity': cognitive_complexity,
                'maintainability_index': maintainability_index,
                'parameter_count': len(node.args.args),
                'return_complexity': self._calculate_return_complexity(node),
                'is_async': isinstance(node, ast.AsyncFunctionDef)
            }
        except Exception as e:
            self.logger.error(f"Error calculating metrics for function {getattr(node, 'name', 'unknown')}: {e}")
            return {}  # Return empty dictionary on error

    def _get_complexity_warnings(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate warnings based on complexity metrics.

        Args:
            metrics (Dict[str, Any]): The complexity metrics.

        Returns:
            List[str]: The list of complexity warnings.
        """
        warnings = []
        try:
            if metrics.get('cyclomatic_complexity', 0) > 10:
                warnings.append("High cyclomatic complexity")
            if metrics.get('cognitive_complexity', 0) > 15:
                warnings.append("High cognitive complexity")
            if metrics.get('maintainability_index', 100) < 20:
                warnings.append("Low maintainability index")
            if metrics.get('program_volume', 0) > 1000:
                warnings.append("High program volume (Halstead metric)")
            return warnings
        except Exception as e:
            self.logger.error(f"Error generating complexity warnings: {e}")
            return []

    def _add_parents(self, node: ast.AST) -> None:
        """
        Add parent references to AST nodes.

        Args:
            node (ast.AST): The AST node to add parent references to.
        """
        for child in ast.iter_child_nodes(node):
            setattr(child, 'parent', node)
            self._add_parents(child)

    def _extract_classes(self, tree: ast.AST) -> List[ExtractedClass]:
        """Extract all classes from the AST."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not self.context.include_private and node.name.startswith('_'):
                    continue
                try:
                    self._current_class = node
                    extracted_class = self._process_class(node)
                    extracted_class.ast_node = node  # Store the AST node!
                    classes.append(extracted_class)
                    self.logger.debug(f"Extracted class: {extracted_class.name}")
                except Exception as e:
                    error_msg = f"Failed to extract class {node.name}: {str(e)}"
                    self.logger.error(error_msg)
                    self.errors.append(error_msg)
                finally:
                    self._current_class = None
        return classes

    def _process_class(self, node: ast.ClassDef) -> ExtractedClass:
        """
        Process a class definition node.

        Args:
            node (ast.ClassDef): The class definition node.

        Returns:
            ExtractedClass: The processed class.
        """
        metrics = self._calculate_class_metrics(node)
        complexity_warnings = self._get_complexity_warnings(metrics)
        return ExtractedClass(
            name=node.name,
            docstring=ast.get_docstring(node),
            lineno=node.lineno,
            source=self._get_source_segment(node),
            metrics=metrics,
            dependencies=self._extract_dependencies(node),
            bases=self._extract_bases(node),
            methods=[
                self._process_function(n)
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ],
            attributes=self._extract_attributes(node),
            is_exception=self._is_exception_class(node),
            decorators=self._extract_decorators(node),
            instance_attributes=self._extract_instance_attributes(node),
            metaclass=self._extract_metaclass(node),
            complexity_warnings=complexity_warnings,
            ast_node=node  # Store the AST node.
        )

    def _get_source_segment(self, node: ast.AST) -> Optional[str]:
        """Get source code segment for a node."""
        if not self.context.include_source:
            return None
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                import astor
                return astor.to_source(node).strip()
        except Exception as e:
            self.logger.error(f"Error getting source segment: {e}")
            return None

    def _extract_bases(self, node: ast.ClassDef) -> List[str]:
        """
        Extract base classes.

        Args:
            node (ast.ClassDef): The class definition node.

        Returns:
            List[str]: The list of base classes.
        """
        bases = []
        for base in node.bases:
            try:
                base_name = self._get_name(base)
                bases.append(base_name)
            except Exception as e:
                self.logger.error(f"Error extracting base class: {e}")
                bases.append('unknown')
        return bases
    
    def _get_function_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[ExtractedArgument]:
        """Extract function arguments from a function definition node."""
        args = []
        for arg in node.args.args:
            arg_name = arg.arg
            type_hint = self._get_name(arg.annotation) if arg.annotation else None
            default_value = None
            is_required = True

            # Check if the argument has a default value
            if node.args.defaults:
                default_index = len(node.args.args) - len(node.args.defaults)
                if node.args.args.index(arg) >= default_index:
                    default_value = self._get_name(node.args.defaults[node.args.args.index(arg) - default_index])
                    is_required = False

            extracted_arg = ExtractedArgument(
                name=arg_name,
                type_hint=type_hint,
                default_value=default_value,
                is_required=is_required
            )
            args.append(extracted_arg)

        return args
    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> ExtractedFunction:
        """Process a function definition node."""
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(f"Expected FunctionDef or AsyncFunctionDef, got {type(node)}")

        try:
            metrics = self._calculate_function_metrics(node)
            docstring = ast.get_docstring(node)

            extracted_function = ExtractedFunction(
                name=node.name,
                lineno=node.lineno,
                source=self._get_source_segment(node),
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
                ast_node=node  # Store the AST Node
            )
            return extracted_function
        except Exception as e:
            self.logger.error(f"Failed to process function {node.name}: {e}")
            raise

    def _extract_functions(self, tree: ast.AST) -> List[ExtractedFunction]:
        """
        Extract top-level functions and async functions from the AST.

        Args:
            tree (ast.AST): The AST to extract functions from.

        Returns:
            List[ExtractedFunction]: The list of extracted functions.
        """
        functions = []
        try:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.logger.debug(
                        f"Found {'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}function: {node.name} "
                        f"(type: {type(node).__name__})"
                    )

                    parent = getattr(node, 'parent', None)
                    if isinstance(parent, ast.Module):
                        if not self.context.include_private and node.name.startswith('_'):
                            self.logger.debug(f"Skipping private function: {node.name}")
                            continue

                        try:
                            self.logger.debug(f"About to process function: {node.name}")

                            extracted_function = self._process_function(node)
                            functions.append(extracted_function)

                            self.logger.debug(
                                f"Successfully processed {'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}"
                                f"function: {node.name}"
                            )
                        except Exception as e:
                            self.logger.error(
                                f"Failed to process {'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}"
                                f"function {node.name}: {str(e)}"
                            )
                            self.errors.append(f"Failed to process function {node.name}: {str(e)}")

            return functions

        except Exception as e:
            self.logger.error(f"Error in _extract_functions: {str(e)}")
            return functions

    def _extract_variables(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract variables from the AST."""
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        annotation = None
                        value = None
                        if isinstance(node, ast.AnnAssign) and node.annotation:
                            annotation = self._get_name(node.annotation)
                        if node.value:
                            try:
                                value = self._get_name(node.value)
                            except Exception as e:
                                self.logger.error(f"Failed to get value for {var_name}: {e}")
                                value = "Unknown"

                        variables.append({
                            'name': var_name,
                            'type': annotation,
                            'value': value
                        })

        return variables

    def _get_name(self, node: Optional[ast.AST]) -> str:
        """Get string representation of a node."""
        if node is None:
            return "Any"

        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            try:
                value = self._get_name(node.value)
                slice_val = self._get_name(node.slice)
                return f"{value}[{slice_val}]"
            except Exception as e:
                self.logger.error(f"Error resolving subscript: {e}")
                return "Unknown"

        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                import astor
                return astor.to_source(node).strip()
        except Exception as e:
            self.logger.error(f"Error getting name from node {type(node).__name__}: {e}")
            return 'Unknown'

    def _get_return_type(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[str]:
        """
        Get the return type annotation for both regular and async functions.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function definition node.

        Returns:
            Optional[str]: The return type annotation if present, otherwise None.
        """
        if node.returns:
            try:
                return_type = self._get_name(node.returns)
                if isinstance(node, ast.AsyncFunctionDef) and not return_type.startswith('Coroutine'):
                    return_type = f'Coroutine[Any, Any, {return_type}]'
                return return_type
            except Exception as e:
                self.logger.error(f"Error getting return type for function {node.name}: {e}")
                return 'Any'
        return None

    def _get_body_summary(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        """
        Generate a summary of the function body for both regular and async functions.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function definition node.

        Returns:
            str: The summary of the function body.
        """
        try:
            if hasattr(ast, 'unparse'):
                body_lines = ast.unparse(node).split('\n')[1:]  # Skip the definition line
            else:
                import astor
                body_lines = astor.to_source(node).split('\n')[1:]

            if len(body_lines) > 5:
                return '\n'.join(body_lines[:5] + ['...'])
            return '\n'.join(body_lines)
        except Exception as e:
            self.logger.error(f"Error generating body summary: {str(e)}")
            return "Error generating body summary"

    def _extract_raises(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
        """
        Extract raised exceptions from function body.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function definition node.

        Returns:
            List[str]: The list of raised exceptions.
        """
        raises = set()
        try:
            for child in ast.walk(node):
                if isinstance(child, ast.Raise):
                    if child.exc:  # Check if exception is specified
                        exc_node = child.exc
                        try:
                            if isinstance(exc_node, ast.Call):
                                exception_name = self._get_name(exc_node.func)
                            elif isinstance(exc_node, (ast.Name, ast.Attribute)):
                                exception_name = self._get_name(exc_node)
                            else:
                                exception_name = "Exception"
                            raises.add(exception_name)
                        except Exception as e:
                            self.logger.debug(f"Could not process raise statement: {e}")
                            raises.add("UnknownException")
                    else:
                        self.logger.debug("Empty raise statement found.")  # Or add "raise" to the raises list if needed
            return list(raises)
        except Exception as e:
            self.logger.error(f"Error extracting raises: {e}")
            return []

    def _calculate_return_complexity(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
        """
        Calculate the complexity of return statements.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function definition node.

        Returns:
            int: The return complexity.
        """
        try:
            return_count = sum(1 for _ in ast.walk(node) if isinstance(_, ast.Return))
            return return_count
        except Exception as e:
            self.logger.error(f"Error calculating return complexity: {e}")
            return 0

    def _is_method(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """
        Check if a function is a method.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function definition node.

        Returns:
            bool: True if the function is a method, False otherwise.
        """
        return isinstance(getattr(node, 'parent', None), ast.ClassDef)

    def _is_generator(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """
        Check if a function is a generator.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function definition node.

        Returns:
            bool: True if the function is a generator, False otherwise.
        """
        return any(isinstance(child, (ast.Yield, ast.YieldFrom)) for child in ast.walk(node))

    def _is_property(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """
        Check if a function is a property.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function definition node.

        Returns:
            bool: True if the function is a property, False otherwise.
        """
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'property':
                return True
        return False

    def _extract_decorators(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> List[str]:
        """
        Extract decorators from a function or class.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]): The node to extract decorators from.

        Returns:
            List[str]: The list of extracted decorators.
        """
        decorators = []
        for decorator in node.decorator_list:
            try:
                decorator_name = self._get_name(decorator)
                decorators.append(decorator_name)
            except Exception as e:
                self.logger.error(f"Error extracting decorator: {e}")
                decorators.append('unknown')
        return decorators

    def _extract_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """
        Extract class attributes.

        Args:
            node (ast.ClassDef): The class definition node.

        Returns:
            List[Dict[str, Any]]: The list of extracted attributes.
        """
        attributes = []
        for child in node.body:
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        attr_name = target.id
                        annotation = None
                        value = None

                        if isinstance(child, ast.AnnAssign) and child.annotation:
                            annotation = self._get_name(child.annotation)

                        if child.value:
                            value = self._get_name(child.value)

                        attributes.append({
                            'name': attr_name,
                            'type': annotation,
                            'value': value
                        })
        return attributes

    def _extract_instance_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """
        Extract instance attributes from __init__ method.

        Args:
            node (ast.ClassDef): The class definition node.

        Returns:
            List[Dict[str, Any]]: The list of extracted instance attributes.
        """
        instance_attributes = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == '__init__':
                for stmt in child.body:
                    if isinstance(stmt, ast.Assign):
                        targets = stmt.targets
                        for target in targets:
                            if isinstance(target, ast.Attribute):
                                if isinstance(target.value, ast.Name) and target.value.id == 'self':
                                    attr_name = target.attr
                                    value = self._get_name(stmt.value)
                                    instance_attributes.append({'name': attr_name, 'value': value})
        return instance_attributes

    def _extract_metaclass(self, node: ast.ClassDef) -> Optional[str]:
        """
        Extract metaclass if specified.

        Args:
            node (ast.ClassDef): The class definition node.

        Returns:
            Optional[str]: The metaclass if specified, otherwise None.
        """
        for keyword in node.keywords:
            if keyword.arg == 'metaclass':
                return self._get_name(keyword.value)
        return None

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """
        Check if a class is an exception class.

        Args:
            node (ast.ClassDef): The class definition node.

        Returns:
            bool: True if the class is an exception class, False otherwise.
        """
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in {'Exception', 'BaseException'}:
                return True
            # Check for imported exceptions
            elif isinstance(base, ast.Attribute):
                if self._get_name(base) in {'Exception', 'BaseException'}:
                    return True
        return False

    def _extract_dependencies(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> Dict[str, Set[str]]:
        """
        Extract dependencies from a node.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]): The node to extract dependencies from.

        Returns:
            Dict[str, Set[str]]: The extracted dependencies.
        """
        try:
            dependencies = {
                'imports': self._extract_imports(node),
                'calls': self._extract_function_calls(node),
                'attributes': self._extract_attribute_access(node)
            }
            return dependencies
        except Exception as e:
            self.logger.error(f"Error extracting dependencies: {e}")
            return {'imports': set(), 'calls': set(), 'attributes': set()}

    def _extract_function_calls(self, node: ast.AST) -> Set[str]:
        """
        Extract function calls from a node.

        Args:
            node (ast.AST): The node to extract function calls from.

        Returns:
            Set[str]: The set of extracted function calls.
        """
        calls = set()
        try:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    try:
                        func_name = self._get_name(child.func)
                        calls.add(func_name)
                    except Exception as e:
                        self.logger.debug(f"Could not unparse function call: {e}")
            return calls
        except Exception as e:
            self.logger.error(f"Error extracting function calls: {e}")
            return set()

    def _extract_attribute_access(self, node: ast.AST) -> Set[str]:
        """
        Extract attribute accesses from a node.

        Args:
            node (ast.AST): The node to extract attribute accesses from.

        Returns:
            Set[str]: The set of extracted attribute accesses.
        """
        attributes = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                try:
                    attr_name = self._get_name(child)
                    attributes.add(attr_name)
                except Exception as e:
                    self.logger.error(f"Failed to unparse attribute access: {e}")
        return attributes

    def _extract_constants(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Extract module-level constants.

        Args:
            tree (ast.AST): The AST of the module.

        Returns:
            List[Dict[str, Any]]: The list of extracted constants.
        """
        constants = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        try:
                            value = self._get_name(node.value)
                            constants.append({
                                'name': target.id,
                                'value': value,
                                'type': type(ast.literal_eval(node.value)).__name__ if isinstance(node.value, ast.Constant) else None
                            })
                            self.logger.debug(f"Extracted constant: {target.id}")
                        except Exception as e:
                            self.logger.error(f"Error extracting constant {target.id}: {e}")
        return constants

    def _extract_imports(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Extract and categorize imports, handling star imports more carefully."""
        imports = {
            'stdlib': set(),
            'local': set(),
            'third_party': set()
        }

        for n in ast.walk(node):
            if isinstance(n, ast.Import):
                for name in n.names:
                    self._categorize_import(name.name, imports)
            elif isinstance(n, ast.ImportFrom):
                if n.names[0].name == '*':
                    self.logger.error(f"Star import encountered: from {n.module} import *, skipping.")
                    # Optionally: raise ExtractionError("Star imports are not supported.")

                elif n.module: 
                    self._categorize_import(n.module, imports)

        return imports

    def _categorize_import(self, module_name: str, deps: Dict[str, Set[str]]) -> None:
        """
        Categorizes an import as stdlib, third-party, or local.
        For external code being analyzed, don't try to verify imports exist.

        Args:
            module_name (str): The name of the module being imported.
            deps (Dict[str, Set[str]]): A dictionary to store dependencies.
        """
        try:
            # Simply categorize based on module name without trying to verify existence
            if module_name in sys.builtin_module_names:
                deps['stdlib'].add(module_name)
            elif module_name.startswith(('api', 'core', 'exceptions')):
                # Treat anything starting with these as local imports
                deps['local'].add(module_name)
            else:
                deps['third_party'].add(module_name)
        except Exception as e:
            self.logger.debug(f"Non-critical error categorizing import {module_name}: {e}")
            # Don't raise the error, just add to third-party
            deps['third_party'].add(module_name)

    def _get_import_map(self) -> Dict[str, str]:
        """Create a map of imported names to their modules."""
        import_map = {}
        if self._module_ast is None:
            return {}
        for node in ast.walk(self._module_ast):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_map[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module
                for alias in node.names:
                    imported_name = alias.asname or alias.name
                    if imported_name == '*':
                        self.logger.warning(f"Star import encountered: from {module_name} import *")
                        import_map[imported_name] = module_name
                    else:
                        import_map[imported_name] = f"{module_name}.{alias.name}" if module_name else alias.name
        return import_map

    def _resolve_base_class(self, base_name: str) -> Optional[ast.ClassDef]:
        """Resolve a base class from the current module or imports."""
        for node in ast.walk(self._module_ast):
            if isinstance(node, ast.ClassDef) and node.name == base_name:
                return node

        import_map = self._get_import_map()
        if base_name in import_map:
            module_name = import_map[base_name]
            return self._resolve_external_class(module_name, base_name)

        return None

    def _calculate_inheritance_depth(self, node: ast.ClassDef) -> int:
        """Calculate the inheritance depth of a class."""
        try:
            depth = 0
            bases = node.bases

            while bases:
                depth += 1
                new_bases = []
                for base in bases:
                    if isinstance(base, ast.Name):
                        base_class = self._resolve_base_class(base.id)
                        if base_class and base_class.bases:
                            new_bases.extend(base_class.bases)
                    elif isinstance(base, ast.Attribute):
                        module_part = self._get_name(base.value)
                        base_class = self._resolve_base_class(f"{module_part}.{base.attr}")
                        if base_class and base_class.bases:
                            new_bases.extend(base_class.bases)
                bases = new_bases

            return depth

        except Exception as e:
            self.logger.error(f"Error calculating inheritance depth: {e}")
            return 0

    def _resolve_external_class(self, module_name: str, class_name: str) -> Optional[ast.ClassDef]:
        """Resolves an external class, using a cache for performance."""
        cache_key = (module_name, class_name)
        if cache_key in _class_cache:
            return _class_cache[cache_key]

        try:
            module = importlib.import_module(module_name)

            # Check if class_name is simply the module itself (for situations where the module is the class).
            if module_name == class_name and isinstance(module, types.ModuleType):
                module_file = getattr(module, '__file__', None)
                if module_file and module_file.endswith(".py"):
                    with open(module_file, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read())
                        # In this case, return the entire module's AST as a pseudo ClassDef
                        _class_cache[cache_key] = tree 
                        return tree
                else:
                    _class_cache[cache_key] = None # cache the None result
                    return None 

            cls = getattr(module, class_name, None) # Normal class lookup
            if cls is None:
                _class_cache[cache_key] = None
                return None

            cls_module = getattr(cls, '__module__', None)
            if cls_module:
                module_path = cls.__module__
                module_obj = sys.modules.get(module_path)
                file_path = getattr(module_obj, '__file__', None)

                if file_path and file_path.endswith(".py"):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read())
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef) and node.name == class_name:
                                _class_cache[cache_key] = node
                                return node

        except (ModuleNotFoundError, ImportError) as e:
            self.logger.warning(f"Could not import module {module_name}: {e}")

        except Exception as e:
            self.logger.error(f"Unexpected error resolving external class: {e}")
       
        _class_cache[cache_key] = None # Cache negative results (not found)
        return None

# End of code_extraction.py module
