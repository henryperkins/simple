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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Union, Tuple, TypedDict, get_type_hints
from core.logger import LoggerSetup
from core.metrics import Metrics  # Ensure this imports the updated Metrics class
from exceptions import ExtractionError

# Define TypedDict classes
class ParameterDict(TypedDict):
    name: str
    type: str
    description: str
    optional: bool
    default_value: Optional[str]

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
class DocstringParameter(TypedDict):
    """Parameter documentation structure."""
    name: str
    type: str
    description: str
    optional: bool = False
    default_value: Optional[str] = None

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
    dependencies: Dict[str, Set[str]] = field(default_factory=dict) # Add this line

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

class CodeExtractor:
    """
    Extracts code elements and metadata from Python source code.

    Attributes:
        context (ExtractionContext): Context for extraction operations.
        logger (Logger): Logger instance for logging.
        _module_ast (Optional[ast.AST]): The abstract syntax tree of the module.
        _current_class (Optional[ast.ClassDef]): The current class being processed.
        errors (List[str]): List of errors encountered during extraction.
        metrics_calculator (Metrics): Metrics calculator instance.
        metadata (CodeMetadata): Metadata extracted from code analysis.
    """

    def __init__(self, context: Optional[ExtractionContext] = None) -> None:
        """
        Initialize the code extractor.

        Args:
            context (Optional[ExtractionContext]): Optional extraction context settings.
        """
        self.logger = LoggerSetup.get_logger(__name__)
        self.context = context or ExtractionContext()
        self._module_ast = None
        self._current_class = None
        self.errors = []
        self.metrics_calculator = Metrics()  # Use the updated Metrics class
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
            pattern = r'$(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2}(?:\.\d+)?)$'
            processed_code = re.sub(pattern, r'["\g<0>"]', source_code)

            self.logger.debug("Preprocessed source code to handle timestamps.")
            return processed_code

        except Exception as e:
            self.logger.error(f"Error preprocessing code: {e}")
            return source_code

    def extract_code(self, source_code: str) -> Optional[ExtractionResult]:
        """
        Extract all code elements and metadata, including detailed metrics and dependency analysis.

        Args:
            source_code (str): The source code to extract.

        Returns:
            Optional[ExtractionResult]: The extraction result or None if extraction fails.

        Raises:
            ExtractionError: If extraction fails.
        """
        try:
            processed_source_code = self._preprocess_code(source_code)
            tree = ast.parse(processed_source_code)  # Use preprocessed source
            self._module_ast = tree
            self._add_parents(tree)  # Needed for resolving inheritance, etc.

            # Analyze dependencies
            dependencies = self.metrics_calculator.analyze_dependencies(tree)
            self.logger.debug(f"Module dependencies: {dependencies}")

            # Generate dependency graph (optional, but useful)
            try:
                self.metrics_calculator.generate_dependency_graph(dependencies, "dependencies.png")  # Or another suitable path
            except Exception as e:
                self.logger.error(f"Failed to generate dependency graph: {e}")

            result = ExtractionResult(
                classes=self._extract_classes(tree),
                functions=self._extract_functions(tree),
                variables=self._extract_variables(tree),
                module_docstring=ast.get_docstring(tree),
                imports=self._extract_imports(tree),
                constants=self._extract_constants(tree),
                errors=self.errors, # Add errors to result
                dependencies=dependencies
            )

            if self.context.metrics_enabled:
                self._calculate_and_add_metrics(result, tree)  # Calculate metrics after extraction

            return result

        except SyntaxError as e:
            self.logger.error("Syntax error in source code: %s", str(e))
            return ExtractionResult(errors=[f"Syntax error: {str(e)}"])

        except ExtractionError as e: # Catch and re-raise custom extraction errors
            raise

        except Exception as e:
            self.logger.error("Extraction failed: %s", str(e))
            raise ExtractionError(f"Failed to extract code: {str(e)}") from e  # Chain the exception
            
    def _calculate_and_add_metrics(self, result: ExtractionResult, tree: ast.AST) -> None:
        """Calculate and add metrics to the extraction result."""
        for cls in result.classes:
            cls.metrics.update(self.metrics_calculator.calculate_halstead_metrics(cls.ast_node))

            cls.metrics.update({  # Calculating metrics directly here
                'method_count': len([n for n in cls.ast_node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]),
                'complexity': self.metrics_calculator.calculate_complexity(cls.ast_node),
                'maintainability': self.metrics_calculator.calculate_maintainability_index(cls.ast_node),
                'inheritance_depth': self._calculate_inheritance_depth(cls.ast_node)
            })
            cls.complexity_warnings.extend(self._get_complexity_warnings(cls.metrics))

            for method in cls.methods:
                method.metrics.update(self.metrics_calculator.calculate_halstead_metrics(method.ast_node)) # Calculate method Halstead metrics
                method.metrics.update(self._calculate_function_metrics(method.ast_node)) # Calculate other function metrics
                method.complexity_warnings.extend(self._get_complexity_warnings(method.metrics))

        # Module-level metrics
        result.metrics.update({
            'total_lines': len(ast.unparse(tree).splitlines()),
            'complexity': self.metrics_calculator.calculate_complexity(tree),
            'maintainability': self.metrics_calculator.calculate_maintainability_index(tree),
            'halstead': self.metrics_calculator.calculate_halstead_metrics(tree)
        })

        for func in result.functions:
            func.metrics.update(self.metrics_calculator.calculate_halstead_metrics(func.ast_node)) # Calculate function Halstead metrics
            func.metrics.update(self._calculate_function_metrics(func.ast_node))  # Call the actual function
            func.complexity_warnings.extend(self._get_complexity_warnings(func.metrics))
                
    def _add_detailed_metrics_and_warnings(self, result: ExtractionResult) -> None:
        """
        Add detailed metrics and code quality warnings to the extraction result.

        Args:
            result (ExtractionResult): The extraction result to enhance.
        """
        for cls in result.classes:
            cls.metrics.update(self.metrics_calculator.calculate_halstead_metrics(cls))
            cls.complexity_warnings.extend(self._get_complexity_warnings(cls.metrics))

        for func in result.functions:
            func.metrics.update(self.metrics_calculator.calculate_halstead_metrics(func))
            func.complexity_warnings.extend(self._get_complexity_warnings(func.metrics))

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

    def _analyze_tree(self, tree: ast.AST) -> None:
        """
        Analyze AST to extract required imports and other metadata.

        Args:
            tree (ast.AST): The AST to analyze.
        """
        try:
            for node in ast.walk(tree):
                # Handle function definitions (both sync and async)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_info = self._extract_function_info(node)
                    self.metadata.functions.append(func_info)
                    self.logger.debug("Found function: %s", node.name)

                # Handle classes
                elif isinstance(node, ast.ClassDef):
                    class_info = self._extract_class_info(node)
                    self.metadata.classes.append(class_info)
                    self.logger.debug("Found class: %s", node.name)

                # Handle imports
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._process_imports(node)

                # Handle potential datetime usage
                elif isinstance(node, ast.Name) and node.id in {'datetime', 'timedelta'}:
                    self.metadata.required_imports.add('datetime')

        except Exception as e:
            self.logger.exception("Error analyzing AST: %s", str(e))

    def _process_imports(self, node: ast.AST) -> None:
        """
        Process import nodes to track dependencies.

        Args:
            node (ast.AST): The import node to process.
        """
        try:
            if isinstance(node, ast.Import):
                for name in node.names:
                    self.metadata.dependencies.add(name.name)
                    self.logger.debug("Found import: %s", name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    if "*" not in [alias.name for alias in node.names]:
                        self.metadata.dependencies.add(node.module)
                        self.logger.debug("Found import from: %s", node.module)
                    else:
                        self.logger.warning("Star import detected from module %s", node.module)
        except Exception as e:
            self.logger.exception("Error processing imports: %s", str(e))

    def metadata_to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to a dictionary.

        Returns:
            Dict[str, Any]: The metadata dictionary.
        """
        return {
            'required_imports': list(self.metadata.required_imports),
            'classes': self.metadata.classes,
            'functions': self.metadata.functions,
            'dependencies': list(self.metadata.dependencies)
        }

    def _get_import_map(self) -> Dict[str, str]:
        """
        Create a map of imported names to their modules.

        Returns:
            Dict[str, str]: The import map.
        """
        import_map = {}
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
                        import_map[imported_name] = f"{module_name}.{imported_name}" if module_name else imported_name

        return import_map

    def _resolve_type_annotation(self, node: ast.AST) -> str:
        """Resolves type annotations to string representations, including custom classes."""
        try:
            if isinstance(node, ast.Name):
                resolved_class = None
                if self.context.file_path and self.context.resolve_external_types:
                    module_name = self._find_import_module(node.id)  # Reuse existing function
                    if module_name:
                        resolved_class = self._resolve_external_class(module_name.split('.')[0], node.id)

                if resolved_class:
                    if hasattr(resolved_class, '__module__') and hasattr(resolved_class, '__name__'):
                        return f"{resolved_class.__module__}.{resolved_class.__name__}"
                return node.id # Fallback to the simple name

            elif isinstance(node, ast.Attribute):
                value = self._resolve_type_annotation(node.value) # Recursive call
                return f"{value}.{node.attr}"

            elif isinstance(node, ast.Subscript):
                value = self._resolve_type_annotation(node.value) # Recursive call
                slice_value = self._resolve_type_annotation(node.slice) # Recursive call
                if value in ('List', 'Set', 'Dict', 'Tuple', 'Optional', 'Union'):
                    value = f"typing.{value}"
                return f"{value}[{slice_value}]"

            elif isinstance(node, ast.Constant):
                return str(node.value)

            elif isinstance(node, ast.Tuple):
                elts = [self._resolve_type_annotation(elt) for elt in node.elts] # Recursive call
                return f"Tuple[{', '.join(elts)}]"

            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                left = self._resolve_type_annotation(node.left) # Recursive call
                right = self._resolve_type_annotation(node.right) # Recursive call
                return f"typing.Union[{left}, {right}]"

            else:
                try:
                    return ast.unparse(node)
                except:
                    return "typing.Any"

        except Exception as e:
            self.logger.debug(f"Error resolving type annotation: {e}")
            return "typing.Any"

    def _find_import_module(self, name: str) -> Optional[str]:
        """Finds the module a name is imported from."""
        for imp in ast.walk(self._module_ast):
            if isinstance(imp, ast.Import):
                module = {alias.name: alias.name for alias in imp.names if alias.name == name or alias.asname == name}
                if module:
                    return module[name]  # Return if found in this import statement
            elif isinstance(imp, ast.ImportFrom) and imp.module:
                module = {alias.name: f"{imp.module}.{alias.name}" for alias in imp.names if alias.name == name or alias.asname == name}
                if module:
                    return module[name]  # Return if found in this importFrom statement
        return None

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
        """
        Extract variables from the AST.

        Args:
            tree (ast.AST): The AST to extract variables from.

        Returns:
            List[Dict[str, Any]]: The list of extracted variables.
        """
        variables = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.AnnAssign):
                continue

            target = node.target
            if not isinstance(target, ast.Name):
                continue

            var_name = target.id
            annotation = None
            value = None

            try:
                annotation = ast.unparse(node.annotation) if node.annotation else None
            except Exception as e:
                self.logger.error(f"Error unparsing annotation for {var_name}: {e}")
                continue

            try:
                value = ast.unparse(node.value) if node.value else None
            except Exception as e:
                self.logger.error(f"Error unparsing value for {var_name}: {e}")

            if not annotation:
                continue

            variable_data = {
                'name': var_name,
                'type': annotation,
                'value': value
            }

            # Check for TypedDict pattern in AST structure
            is_typeddict = (
                isinstance(node.annotation, ast.Subscript) and
                isinstance(node.annotation.value, ast.Name) and
                (node.annotation.value.id == "TypedDict" or
                 node.annotation.value.id.endswith("TypedDict"))
            )

            if is_typeddict:
                try:
                    # For TypedDict, we'll store it as a generic dictionary type
                    # since we can't reliably resolve the actual TypedDict fields
                    variable_data['type'] = "Dict[str, Any]"
                except Exception as e:
                    self.logger.error(f"Error processing TypedDict annotation for {var_name}: {e}")
                    variable_data['type'] = "Dict[str, Any]"  # Fallback type

            variables.append(variable_data)
            self.logger.debug(f"Extracted variable: {var_name}")

        return variables

    def _extract_function_info(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        """
        Extract information from a function definition node.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function definition node.

        Returns:
            Dict[str, Any]: The extracted function information.
        """
        return {
            'name': node.name,
            'args': self._get_function_args(node),
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'line_number': node.lineno
        }

    def _extract_class_info(self, node: ast.ClassDef) -> Dict[str, Any]:
        """
        Extract information from a class definition node.

        Args:
            node (ast.ClassDef): The class definition node.

        Returns:
            Dict[str, Any]: The extracted class information.
        """
        return {
            'name': node.name,
            'bases': [self._get_name(base) for base in node.bases],
            'methods': [m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))],
            'line_number': node.lineno
        }

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
            source=ast.unparse(node) if self.context.include_source else None,
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
            complexity_warnings=complexity_warnings
        )

    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> ExtractedFunction:
        """Process a function definition node."""
        if not (isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef)):
            raise ValueError(f"Expected FunctionDef or AsyncFunctionDef, got {type(node)}")
    
        try:
            metrics = self._calculate_function_metrics(node)
            docstring = ast.get_docstring(node)
    
            extracted_function = ExtractedFunction(
                name=node.name,
                lineno=node.lineno,
                source=ast.unparse(node) if self.context.include_source else None,
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
                raises=self._extract_raises(node)
            )
            extracted_function.ast_node = node  # Store the AST Node
            return extracted_function
        except Exception as e:
            self.logger.error(f"Failed to process function {node.name}: {e}")
            raise

    def _get_function_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[ExtractedArgument]:
        """
        Extract function arguments from both regular and async functions.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function definition node.

        Returns:
            List[ExtractedArgument]: The list of extracted arguments.
        """
        args = []
        try:
            # Process positional args
            for arg in node.args.args:
                arg_info = ExtractedArgument(
                    name=arg.arg,
                    type_hint=self._get_name(arg.annotation) if arg.annotation else None,
                    default_value=None,
                    is_required=True
                )
                args.append(arg_info)

            # Handle default values
            defaults = node.args.defaults
            if defaults:
                default_offset = len(args) - len(defaults)
                for i, default in enumerate(defaults):
                    arg_index = default_offset + i
                    args[arg_index].default_value = ast.unparse(default)
                    args[arg_index].is_required = False

            # Handle keyword-only args
            for arg in node.args.kwonlyargs:
                arg_info = ExtractedArgument(
                    name=arg.arg,
                    type_hint=self._get_name(arg.annotation) if arg.annotation else None,
                    default_value=None,
                    is_required=True
                )
                args.append(arg_info)

            # Handle keyword-only defaults
            kw_defaults = node.args.kw_defaults
            if kw_defaults:
                kw_offset = len(args) - len(kw_defaults)
                for i, default in enumerate(kw_defaults):
                    if default is not None:
                        arg_index = kw_offset + i
                        args[arg_index].default_value = ast.unparse(default)
                        args[arg_index].is_required = False

            return args
        except Exception as e:
            self.logger.error(f"Error extracting function arguments: {str(e)}")
            return []

    def _get_name(self, node: Optional[ast.AST]) -> str:
        """Get string representation of a name node, using _resolve_type_annotation."""
        if node is None:
            return 'Any'
        return self._resolve_type_annotation(node)

    def _calculate_class_metrics(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Calculate metrics for a class."""
        if not self.context.metrics_enabled:
            return {}
    
        try:
            complexity = self.metrics_calculator.calculate_complexity(node)
            return {
                'method_count': len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]),
                'complexity': complexity,
                'maintainability': self.metrics_calculator.calculate_maintainability_index(node),
                'inheritance_depth': self._calculate_inheritance_depth(node)
            }
        except Exception as e:
            self.logger.error("Error calculating class metrics: %s", str(e))
            return {'error': str(e)}

    def _calculate_module_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        if not self.context.metrics_enabled:
            return {}
    
        return {
            'total_lines': len(ast.unparse(tree).splitlines()),
            'complexity': self.metrics_calculator.calculate_complexity(tree),
            'maintainability': self.metrics_calculator.calculate_maintainability_index(tree),
            'halstead': self.metrics_calculator.calculate_halstead_metrics(tree)
        }

    def _calculate_function_metrics(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        """Calculate metrics for a function."""
        if not (isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef)):
            self.logger.error("Provided node is not a function definition: %s", ast.dump(node))
            return {}
    
        try:
            return {
                'cyclomatic_complexity': self.metrics_calculator.calculate_cyclomatic_complexity(node),
                'cognitive_complexity': self.metrics_calculator.calculate_cognitive_complexity(node),
                'maintainability_index': self.metrics_calculator.calculate_maintainability_index(node),
                'parameter_count': len(node.args.args),
                'return_complexity': self._calculate_return_complexity(node),
                'is_async': isinstance(node, ast.AsyncFunctionDef)
            }
        except Exception as e:
            self.logger.error(f"Error calculating metrics for function {node.name}: {e}")
            return {}

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
                        calls.add(ast.unparse(child.func))
                    except Exception as e:
                        self.logger.debug(f"Could not unparse function call: {e}")
            return calls
        except Exception as e:
            self.logger.error(f"Error extracting function calls: {e}")
            return set()

    def _is_generator(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """
        Check if a function is a generator.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function definition node.

        Returns:
            bool: True if the function is a generator, False otherwise.
        """
        try:
            for child in ast.walk(node):
                if isinstance(child, (ast.Yield, ast.YieldFrom)):
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error checking if function is generator: {e}")
            return False

    def _is_property(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """
        Check if a function is a property.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function definition node.

        Returns:
            bool: True if the function is a property, False otherwise.
        """
        try:
            return any(
                isinstance(decorator, ast.Name) and decorator.id == 'property'
                for decorator in node.decorator_list
            )
        except Exception as e:
            self.logger.error(f"Error checking if function is property: {e}")
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
        try:
            for decorator in node.decorator_list:
                try:
                    decorators.append(ast.unparse(decorator))
                except Exception as e:
                    self.logger.debug(f"Could not unparse decorator: {e}")
                    decorators.append("UnknownDecorator")
            return decorators
        except Exception as e:
            self.logger.error(f"Error extracting decorators: {e}")
            return []

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
                    attributes.add(ast.unparse(child))
                except Exception as e:
                    self.logger.error(f"Failed to unparse attribute access: {e}")
        return attributes

    def _extract_bases(self, node: ast.ClassDef) -> List[str]:
        """
        Extract base classes.

        Args:
            node (ast.ClassDef): The class definition node.

        Returns:
            List[str]: The list of base classes.
        """
        return [ast.unparse(base) for base in node.bases]

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
            if isinstance(child, ast.AnnAssign):
                attributes.append({
                    'name': ast.unparse(child.target),
                    'type': ast.unparse(child.annotation) if child.annotation else None,
                    'value': ast.unparse(child.value) if child.value else None
                })
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        attributes.append({
                            'name': target.id,
                            'type': None,
                            'value': ast.unparse(child.value)
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
        init_method = next((m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == '__init__'), None)
        if not init_method:
            return []

        attributes = []
        for child in ast.walk(init_method):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                        attributes.append({
                            'name': target.attr,
                            'type': None,
                            'value': ast.unparse(child.value)
                        })
        return attributes

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
                return ast.unparse(keyword.value)
        return None

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
                            value = ast.unparse(node.value)
                        except Exception:
                            value = 'Unknown'
                        constants.append({
                            'name': target.id,
                            'value': value,
                            'type': type(ast.literal_eval(node.value)).__name__ if isinstance(node.value, ast.Constant) else None
                        })
                        self.logger.debug(f"Extracted constant: {target.id}")
        return constants

    def _extract_imports(self, tree: ast.AST) -> Dict[str, Set[str]]:
        """
        Extract and categorize imports.

        Args:
            tree (ast.AST): The AST of the module.

        Returns:
            Dict[str, Set[str]]: The categorized imports.
        """
        imports = {
            'stdlib': set(),
            'local': set(),
            'third_party': set()
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    self._categorize_import(name.name, imports)
            elif isinstance(node, ast.ImportFrom) and node.module:
                self._categorize_import(node.module, imports)

        return imports

    def _categorize_import(self, module_name: str, imports: Dict[str, Set[str]]) -> None:
        """
        Categorize an import as stdlib, local, or third-party.

        Args:
            module_name (str): The name of the module.
            imports (Dict[str, Set[str]]): The categorized imports.
        """
        if module_name.startswith('.'):
            imports['local'].add(module_name)
        elif module_name in sys.stdlib_module_names:
            imports['stdlib'].add(module_name)
        else:
            imports['third_party'].add(module_name)
        self.logger.debug(f"Categorized import: {module_name}")

    def _is_method(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """
        Check if a function is a method.
    
        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function definition node.
    
        Returns:
            bool: True if the function is a method, False otherwise.
        """
        if self._current_class:
            return True
    
        for parent in ast.walk(node):  # Iterate over parents
            if isinstance(parent, ast.ClassDef) and node in parent.body:
                return True  # Return immediately if found
        return False

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """
        Check if a class is an exception class.

        Args:
            node (ast.ClassDef): The class definition node.

        Returns:
            bool: True if the class is an exception class, False otherwise.
        """
        return any(
            isinstance(base, ast.Name) and base.id in {'Exception', 'BaseException'}
            for base in node.bases
        )

    def _get_return_type(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[str]:
        """
        Get the return type annotation for both regular and async functions.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function definition node.

        Returns:
            Optional[str]: The return type annotation if present, otherwise None.
        """
        if node.returns:
            return_type = ast.unparse(node.returns)
            if isinstance(node, ast.AsyncFunctionDef) and not return_type.startswith('Coroutine'):
                return_type = f'Coroutine[Any, Any, {return_type}]'
            return return_type
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
            body_lines = ast.unparse(node).split('\n')[1:]  # Skip the definition line
            if len(body_lines) > 5:
                return '\n'.join(body_lines[:5] + ['...'])
            return '\n'.join(body_lines)
        except Exception as e:
            self.logger.error(f"Error generating body summary: {str(e)}")
            return "Error generating body summary"

    def _calculate_inheritance_depth(self, node: ast.ClassDef) -> int:
        """
        Calculate the inheritance depth of a class.

        Args:
            node (ast.ClassDef): The class definition node.

        Returns:
            int: The inheritance depth.
        """
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
                        try:
                            module_part = self._get_name(base.value)
                            base_class = self._resolve_base_class(f"{module_part}.{base.attr}")
                            if base_class and base_class.bases:
                                new_bases.extend(base_class.bases)
                        except Exception as e:
                            self.logger.debug(f"Could not resolve qualified base class: {e}")
                bases = new_bases

            return depth

        except Exception as e:
            self.logger.error(f"Error calculating inheritance depth: {e}")
            return 0

    def _resolve_base_class(self, base_name: str) -> Optional[ast.ClassDef]:
        """
        Resolve a base class from the current module or imports.

        Args:
            base_name (str): The name of the base class.

        Returns:
            Optional[ast.ClassDef]: The resolved base class if found, otherwise None.
        """
        for node in ast.walk(self._module_ast):
            if isinstance(node, ast.ClassDef) and node.name == base_name:
                return node

        import_map = self._get_import_map()
        if base_name in import_map:
            module_name = import_map[base_name]
            return self._resolve_external_class(module_name, base_name)

        return None

    def _resolve_external_class(self, module_name: str, class_name: str) -> Optional[type]:
        """
        Dynamically resolves a class from an external module.

        Args:
            module_name (str): The name of the module.
            class_name (str): The name of the class.

        Returns:
            Optional[type]: The resolved class, or None if not found.
        """
        try:
            # 1. Attempt direct import (in case it's somehow available)
            try:
                module = importlib.import_module(module_name)
                return getattr(module, class_name, None)
            except ImportError:
                pass  # Fallback to dynamic import

            # 2. Dynamic import from file path (if available)
            if self.context.file_path:
                module_path = module_name.replace('.', '/')
                file_path = Path(self.context.file_path).parent / f"{module_path}.py"
                if file_path.exists():
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module  # Add to sys.modules to prevent re-loading
                        try:
                            spec.loader.exec_module(module)
                            return getattr(module, class_name, None)
                        except Exception as e:
                            self.logger.error(f"Error executing module {module_name}: {e}")
                            self.errors.append(f"Error executing module {module_name}: {e}")  # Optional: Add to errors list
                            return None  # Explicitly return None after logging the error

            # 3. If all else fails, check current loaded modules
            if module_name in sys.modules:
                return getattr(sys.modules[module_name], class_name, None)

            return None

        except Exception as e:
            self.logger.error(f"Failed to resolve class {class_name} from {module_name}: {e}")
            return None

    def _add_parents(self, node: ast.AST) -> None:
        """
        Add parent references to AST nodes.

        Args:
            node (ast.AST): The AST node to add parent references to.
        """
        for child in ast.iter_child_nodes(node):
            child.parent = node
            self._add_parents(child)

    def _get_module_name(self, node: ast.Name, import_node: Union[ast.Import, ast.ImportFrom]) -> Optional[str]:
        """
        Get module name from import statement for a given node.

        Args:
            node (ast.Name): The AST node representing the type.
            import_node (Union[ast.Import, ast.ImportFrom]): The import statement node.

        Returns:
            Optional[str]: The module name if found, otherwise None.
        """
        if isinstance(import_node, ast.Import):
            for name in import_node.names:
                if name.name == node.id or name.asname == node.id:
                    return name.name
        elif isinstance(import_node, ast.ImportFrom) and import_node.module:
            for name in import_node.names:
                if name.name == node.id or name.asname == node.id:
                    return f"{import_node.module}.{name.name}"
        return None

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
                                exception_name = self._get_exception_name(exc_node.func)
                            elif isinstance(exc_node, (ast.Name, ast.Attribute)):
                                exception_name = self._get_exception_name(exc_node)
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

    def _get_exception_name(self, node: ast.AST) -> str:
        """
        Extract the name of the exception from the node.

        Args:
            node (ast.AST): The node representing the exception.

        Returns:
            str: The name of the exception.
        """
        try:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return f"{self._get_exception_name(node.value)}.{node.attr}"
            try:
                return ast.unparse(node)
            except (AttributeError, ValueError, TypeError):
                return "Exception"
        except Exception as e:
            self.logger.error(f"Error getting exception name: {e}")
            return "Exception"