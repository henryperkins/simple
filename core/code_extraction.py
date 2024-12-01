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
from core.metrics import Metrics
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
    def __init__(self, context: Optional[ExtractionContext] = None):
        """Initialize the code extractor.
        
        Args:
            context: Optional extraction context settings
        """
        self.logger = LoggerSetup.get_logger(__name__)
        self.context = context or ExtractionContext()
        self._module_ast = None
        self._current_class = None
        self.errors = []
        self.metrics_calculator = Metrics()
        self.metadata = CodeMetadata.create_empty()
        self.logger.debug(f"Processing in {__name__}")
        
    def _preprocess_code(self, source_code: str) -> str:
        """
        Preprocess source code to handle timestamps and other special cases.

        Args:
            source_code: The source code to preprocess.

        Returns:
            str: The preprocessed source code.
        """
        try:
            # Convert timestamps with leading zeros to string literals. Handles milliseconds/microseconds.
            pattern = r'\[(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2}(?:\.\d+)?)\]'
            processed_code = re.sub(pattern, r'["\g<0>"]', source_code)

            self.logger.debug("Preprocessed source code to handle timestamps.")
            return processed_code

        except Exception as e:
            self.logger.error(f"Error preprocessing code: {e}")
            return source_code
        
    def extract_code(self, source_code: str) -> Optional[ExtractionResult]:
        """Extract all code elements and metadata."""
        try:
            tree = ast.parse(source_code)
            self._module_ast = tree
            self._add_parents(tree)

            # First analyze tree to populate metadata
            self._analyze_tree(tree)

            result = ExtractionResult(
                classes=self._extract_classes(tree),
                functions=self._extract_functions(tree),
                variables=self._extract_variables(tree),
                module_docstring=ast.get_docstring(tree),
                imports=self._extract_imports(tree),
                constants=self._extract_constants(tree),
                metrics=self._calculate_module_metrics(tree)
            )
            
            return result

        except SyntaxError as e:
            self.logger.error("Syntax error in source code: %s", str(e))
            return ExtractionResult(errors=[f"Syntax error: {str(e)}"])
            
        except Exception as e:
            self.logger.error("Extraction failed: %s", str(e))
            raise ExtractionError(f"Failed to extract code: {str(e)}")

    def _analyze_tree(self, tree: ast.AST) -> None:
        """
        Analyze AST to extract required imports and other metadata.

        Args:
            tree: The AST to analyze
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
        """Process import nodes to track dependencies."""
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
        """Convert metadata to a dictionary."""
        return {
            'required_imports': list(self.metadata.required_imports),
            'classes': self.metadata.classes,
            'functions': self.metadata.functions,
            'dependencies': list(self.metadata.dependencies)
        }

    def _get_import_map(self) -> Dict[str, str]:
        """Create a map of imported names to their modules."""
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
        """
        Safely resolve type annotations to their string representation.

        Args:
            node: The AST node representing a type annotation

        Returns:
            str: String representation of the type
        """
        try:
            if isinstance(node, ast.Name):
                # Try to resolve the type if it's a custom class
                try:
                    resolved_class = None
                    if self.context.file_path:  # Only attempt resolution if we have a file path
                        module_name = None
                        
                        # Try to determine the module from imports
                        for imp in ast.walk(self._module_ast):
                            if isinstance(imp, ast.Import):
                                for name in imp.names:
                                    if name.name == node.id or name.asname == node.id:
                                        module_name = name.name
                                        break
                            elif isinstance(imp, ast.ImportFrom) and imp.module:
                                for name in imp.names:
                                    if name.name == node.id or name.asname == node.id:
                                        module_name = f"{imp.module}.{name.name}"
                                        break
                        
                        if module_name:
                            resolved_class = self._resolve_external_class(
                                module_name.split('.')[0], 
                                node.id
                            )
                    return node.id
                except Exception as e:
                    self.logger.debug(f"Could not resolve type {node.id}: {e}")
                    return node.id
            elif isinstance(node, ast.Attribute):
                value = self._resolve_type_annotation(node.value)
                return f"{value}.{node.attr}"
            elif isinstance(node, ast.Subscript):
                value = self._resolve_type_annotation(node.value)
                slice_value = self._resolve_type_annotation(node.slice)
                return f"{value}[{slice_value}]"
            return ast.unparse(node)
        except Exception as e:
            self.logger.debug(f"Error resolving type annotation: {e}")
            return "typing.Any"

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
        """Extract top-level functions and async functions from the AST."""
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
        """Extract information from a function definition node."""
        return {
            'name': node.name,
            'args': self._get_function_args(node),
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'line_number': node.lineno
        }

    def _extract_class_info(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Extract information from a class definition node."""
        return {
            'name': node.name,
            'bases': [self._get_name(base) for base in node.bases],
            'methods': [m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))],
            'line_number': node.lineno
        }

    def _process_class(self, node: ast.ClassDef) -> ExtractedClass:  
        """Process a class definition node."""  
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

    def _process_function(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> ExtractedFunction:
        """Process a function definition node."""
        if not (isinstance(node, ast.FunctionDef) or 
                isinstance(node, ast.AsyncFunctionDef)):
            raise ValueError(
                f"Expected FunctionDef or AsyncFunctionDef, got {type(node)}"
            )

        try:
            metrics = self._calculate_function_metrics(node)
            docstring = ast.get_docstring(node)
            
            return ExtractedFunction(
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
        except Exception as e:
            self.logger.error(f"Failed to process function {node.name}: {e}")
            raise

    def _get_function_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[ExtractedArgument]:
        """Extract function arguments from both regular and async functions."""
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
        """Get string representation of a name node."""
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
                'maintainability': (
                    self.metrics_calculator.calculate_maintainability_index(node)
                ),
                'inheritance_depth': self._calculate_inheritance_depth(node)
            }
        except Exception as e:
            self.logger.error("Error calculating class metrics: %s", str(e))
            return {'error': str(e)}

    def _calculate_module_metrics(self, tree: ast.AST) -> Dict[str, Any]:  
        """Calculate module-level metrics."""  
        if not self.context.metrics_enabled:  
            return {}  
  
        return {  
            'total_lines': len(ast.unparse(tree).splitlines()),  
            'complexity': self.metrics_calculator.calculate_complexity(tree),  
            'maintainability': self.metrics_calculator.calculate_maintainability_index(tree),  
            'halstead': self.metrics_calculator.calculate_halstead_metrics(tree)  
        }  

    def _calculate_function_metrics(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        """Calculate metrics for both regular and async functions."""
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

    def _get_complexity_warnings(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate warnings based on complexity metrics."""
        warnings = []
        try:
            if metrics.get('cyclomatic_complexity', 0) > 10:
                warnings.append("High cyclomatic complexity")
            if metrics.get('cognitive_complexity', 0) > 15:
                warnings.append("High cognitive complexity")
            if metrics.get('maintainability_index', 100) < 20:
                warnings.append("Low maintainability index")
            return warnings
        except Exception as e:
            self.logger.error(f"Error generating complexity warnings: {e}")
            return []

    def _extract_dependencies(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> Dict[str, Set[str]]:
        """Extract dependencies from a node."""
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
        """Extract function calls from a node."""
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
        """Check if a function is a generator."""
        try:
            for child in ast.walk(node):
                if isinstance(child, (ast.Yield, ast.YieldFrom)):
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error checking if function is generator: {e}")
            return False

    def _is_property(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if a function is a property."""
        try:
            return any(
                isinstance(decorator, ast.Name) and decorator.id == 'property'
                for decorator in node.decorator_list
            )
        except Exception as e:
            self.logger.error(f"Error checking if function is property: {e}")
            return False

    def _extract_decorators(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> List[str]:
        """Extract decorators from a function or class."""
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
        """Extract attribute accesses from a node."""  
        attributes = set()  
        for child in ast.walk(node):  
            if isinstance(child, ast.Attribute):  
                try:  
                    attributes.add(ast.unparse(child))  
                except Exception as e:  
                    self.logger.error(f"Failed to unparse attribute access: {e}")  
        return attributes  

    def _extract_bases(self, node: ast.ClassDef) -> List[str]:  
        """Extract base classes."""  
        return [ast.unparse(base) for base in node.bases]  

    def _extract_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:  
        """Extract class attributes."""  
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
        """Extract instance attributes from __init__ method."""
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
        """Extract metaclass if specified."""
        for keyword in node.keywords:
            if keyword.arg == 'metaclass':
                return ast.unparse(keyword.value)
        return None

    def _extract_constants(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract module-level constants."""
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
        """Extract and categorize imports."""
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
        """Categorize an import as stdlib, local, or third-party."""
        if module_name.startswith('.'):
            imports['local'].add(module_name)
        elif module_name in sys.stdlib_module_names:
            imports['stdlib'].add(module_name)
        else:
            imports['third_party'].add(module_name)
        self.logger.debug(f"Categorized import: {module_name}")

    def _is_method(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if a function is a method."""
        if self._current_class:
            return True

        for parent in ast.walk(node):
            if isinstance(parent, ast.ClassDef) and node in parent.body:
                return True
        return False

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is an exception class."""
        return any(
            isinstance(base, ast.Name) and base.id in {'Exception', 'BaseException'}
            for base in node.bases
        )

    def _get_return_type(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[str]:
        """Get the return type annotation for both regular and async functions."""
        if node.returns:
            return_type = ast.unparse(node.returns)
            if isinstance(node, ast.AsyncFunctionDef) and not return_type.startswith('Coroutine'):
                return_type = f'Coroutine[Any, Any, {return_type}]'
            return return_type
        return None

    def _get_body_summary(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        """Generate a summary of the function body for both regular and async functions."""
        try:
            body_lines = ast.unparse(node).split('\n')[1:]  # Skip the definition line
            if len(body_lines) > 5:
                return '\n'.join(body_lines[:5] + ['...'])
            return '\n'.join(body_lines)
        except Exception as e:
            self.logger.error(f"Error generating body summary: {str(e)}")
            return "Error generating body summary"

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
        """Resolve a base class from the current module or imports."""
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
            module_name: The name of the module.
            class_name: The name of the class.

        Returns:
            The resolved class, or None if not found.
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
                        spec.loader.exec_module(module)
                        return getattr(module, class_name, None)

            # 3. If all else fails, check current loaded modules
            if module_name in sys.modules:
                return getattr(sys.modules[module_name], class_name, None)

            return None

        except Exception as e:
            self.logger.error(f"Failed to resolve class {class_name} from {module_name}: {e}")
            return None

    def _resolve_type_annotation(self, node: ast.AST) -> str:
        """
        Safely resolve type annotations to their string representation.

        Args:
            node: The AST node representing a type annotation

        Returns:
            str: String representation of the type
        """
        try:
            if isinstance(node, ast.Name):
                # Try to resolve the type if it's a custom class
                try:
                    resolved_class = None
                    if self.context.file_path:  # Only attempt resolution if we have a file path
                        parent_node = getattr(node, 'parent', None)
                        module_name = None
                        
                        # Try to determine the module from imports
                        for imp in ast.walk(self._module_ast):
                            if isinstance(imp, ast.Import):
                                for name in imp.names:
                                    if name.name == node.id or name.asname == node.id:
                                        module_name = name.name
                                        break
                            elif isinstance(imp, ast.ImportFrom) and imp.module:
                                for name in imp.names:
                                    if name.name == node.id or name.asname == node.id:
                                        module_name = f"{imp.module}.{name.name}"
                                        break
                        
                        if module_name:
                            resolved_class = self._resolve_external_class(module_name.split('.')[0], node.id)
                
                    if resolved_class:
                        # Use the full qualified name if available
                        if hasattr(resolved_class, '__module__') and hasattr(resolved_class, '__name__'):
                            return f"{resolved_class.__module__}.{resolved_class.__name__}"
                except Exception as e:
                    self.logger.debug(f"Could not resolve type {node.id}: {e}")
                
                # Fall back to the simple name if resolution fails
                return node.id
                
            elif isinstance(node, ast.Attribute):
                value = self._resolve_type_annotation(node.value)
                return f"{value}.{node.attr}"
                
            elif isinstance(node, ast.Subscript):
                value = self._resolve_type_annotation(node.value)
                slice_value = self._resolve_type_annotation(node.slice)
                
                # Handle special cases for common container types
                if value in ('List', 'Set', 'Dict', 'Tuple', 'Optional', 'Union'):
                    value = f"typing.{value}"
                
                return f"{value}[{slice_value}]"
                
            elif isinstance(node, ast.Constant):
                return str(node.value)
                
            elif isinstance(node, ast.Tuple):
                elts = [self._resolve_type_annotation(elt) for elt in node.elts]
                return f"Tuple[{', '.join(elts)}]"
                
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                # Handle Union types written with | (Python 3.10+)
                left = self._resolve_type_annotation(node.left)
                right = self._resolve_type_annotation(node.right)
                return f"typing.Union[{left}, {right}]"
            
            else:
                # Try to use ast.unparse for other cases, fall back to Any
                try:
                    return ast.unparse(node)
                except:
                    return "typing.Any"
                    
        except Exception as e:
            self.logger.debug(f"Error resolving type annotation: {e}")
            return "typing.Any"

    def _add_parents(self, node: ast.AST) -> None:
        """Add parent references to AST nodes."""
        for child in ast.iter_child_nodes(node):
            child.parent = node
            self._add_parents(child)
    def _get_module_name(self, node: ast.Name, import_node: Union[ast.Import, ast.ImportFrom]) -> Optional[str]:
        """
        Get module name from import statement for a given node.
        
        Args:
            node: The AST node representing the type
            import_node: The import statement node
            
        Returns:
            Optional module name if found
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
        """Calculate the complexity of return statements."""
        try:
            return_count = sum(1 for _ in ast.walk(node) if isinstance(_, ast.Return))
            return return_count
        except Exception as e:
            self.logger.error(f"Error calculating return complexity: {e}")
            return 0

    def _extract_raises(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
        """Extract raised exceptions from function body."""
        raises = set()
        try:
            for child in ast.walk(node):
                if isinstance(child, ast.Raise) and child.exc:
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
            return list(raises)
        except Exception as e:
            self.logger.error(f"Error extracting raises: {e}")
            return []

    def _get_exception_name(self, node: ast.AST) -> str:
        """Extract the name of the exception from the node."""
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