"""Type definitions for code extraction.

This module provides data classes to represent various elements extracted from Python source code,
including functions, classes, and the overall extraction context.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Union
import ast
from core.metrics import Metrics
from pathlib import Path

@dataclass
class ExtractedArgument:
    """Represents a function argument.

    Attributes:
        name (str): The name of the argument.
        type_hint (Optional[str]): The type hint of the argument, if available.
        default_value (Optional[str]): The default value of the argument, if any.
        is_required (bool): Indicates whether the argument is required (i.e., has no default value).
    """
    name: str
    type_hint: Optional[str] = None
    default_value: Optional[str] = None
    is_required: bool = True

@dataclass
class ExtractionContext:
    """Context for code extraction operations.

    Attributes:
        metrics (Optional[Metrics]): An instance of Metrics for calculating code metrics.
        module_name (Optional[str]): The name of the module being analyzed.
        include_private (bool): Whether to include private members in the extraction.
        include_magic (bool): Whether to include magic methods in the extraction.
        include_nested (bool): Whether to include nested classes and functions.
        include_source (bool): Whether to include the source code in the extraction.
        metrics_enabled (bool): Whether metrics calculation is enabled.
        max_line_length (int): The maximum line length for code formatting.
        ignore_decorators (Set[str]): A set of decorators to ignore during extraction.
        base_path (Optional[Path]): The base path for resolving relative imports.
    """
    metrics: Optional[Metrics] = None
    module_name: Optional[str] = None
    include_private: bool = False
    include_magic: bool = False
    include_nested: bool = True
    include_source: bool = True
    metrics_enabled: bool = True
    max_line_length: int = 88
    ignore_decorators: Set[str] = field(default_factory=set)
    base_path: Optional[Path] = None

@dataclass
class ExtractedElement:
    """Base class for extracted code elements.

    Attributes:
        name (str): The name of the element.
        lineno (int): The line number where the element is defined.
        source (Optional[str]): The source code of the element, if included.
        docstring (Optional[str]): The docstring of the element, if available.
        metrics (Dict[str, Any]): A dictionary of metrics related to the element.
        dependencies (Dict[str, Set[str]]): A dictionary of dependencies related to the element.
        decorators (List[str]): A list of decorators applied to the element.
        complexity_warnings (List[str]): A list of warnings related to the element's complexity.
    """
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
    """Represents an extracted function.

    Attributes:
        return_type (Optional[str]): The return type annotation of the function, if available.
        is_method (bool): Indicates whether the function is a method.
        is_async (bool): Indicates whether the function is asynchronous.
        is_generator (bool): Indicates whether the function is a generator.
        is_property (bool): Indicates whether the function is a property.
        body_summary (str): A summary of the function body.
        args (List[ExtractedArgument]): A list of arguments for the function.
        raises (List[str]): A list of exceptions raised by the function.
        ast_node (Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]]): The AST node representing the function.
    """
    return_type: Optional[str] = None
    is_method: bool = False
    is_async: bool = False
    is_generator: bool = False
    is_property: bool = False
    body_summary: str = ""
    args: List[ExtractedArgument] = field(default_factory=list)
    raises: List[str] = field(default_factory=list)
    ast_node: Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = None

@dataclass
class ExtractedClass(ExtractedElement):
    """Represents an extracted class.

    Attributes:
        bases (List[str]): A list of base classes for the class.
        methods (List[ExtractedFunction]): A list of methods defined in the class.
        attributes (List[Dict[str, Any]]): A list of class-level attributes.
        is_exception (bool): Indicates whether the class is an exception class.
        instance_attributes (List[Dict[str, Any]]): A list of instance attributes.
        metaclass (Optional[str]): The metaclass of the class, if specified.
        ast_node (Optional[ast.ClassDef]): The AST node representing the class.
    """
    bases: List[str] = field(default_factory=list)
    methods: List[ExtractedFunction] = field(default_factory=list)
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    is_exception: bool = False
    instance_attributes: List[Dict[str, Any]] = field(default_factory=list)
    metaclass: Optional[str] = None
    ast_node: Optional[ast.ClassDef] = None

@dataclass
class ExtractionResult:
    """Contains the complete extraction results.

    Attributes:
        classes (List[ExtractedClass]): A list of extracted classes.
        functions (List[ExtractedFunction]): A list of extracted functions.
        variables (List[Dict[str, Any]]): A list of extracted variables.
        module_docstring (Optional[str]): The module-level docstring, if available.
        imports (Dict[str, Set[str]]): A dictionary of imports categorized by type.
        constants (List[Dict[str, Any]]): A list of extracted constants.
        errors (List[str]): A list of errors encountered during extraction.
        metrics (Dict[str, Any]): A dictionary of module-level metrics.
        dependencies (Dict[str, Set[str]]): A dictionary of module-level dependencies.
    """
    classes: List[ExtractedClass] = field(default_factory=list)
    functions: List[ExtractedFunction] = field(default_factory=list)
    variables: List[Dict[str, Any]] = field(default_factory=list)
    module_docstring: Optional[str] = None
    imports: Dict[str, Set[str]] = field(default_factory=dict)
    constants: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute value safely.

        Args:
            key (str): The attribute name to retrieve.
            default (Any): The default value to return if the attribute is not found.

        Returns:
            Any: The value of the attribute or the default value if not found.
        """
        return getattr(self, key, default)
