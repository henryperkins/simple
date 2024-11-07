"""
extract package.

This package provides utilities for extracting classes and functions from Python source code,
parsing their details, and analyzing them for documentation and metrics purposes.

Modules:
    - code: Functions to extract classes and functions from AST.
    - classes: ClassExtractor for extracting class details.
    - functions: FunctionExtractor for extracting function details.
    - utils: Utility functions for AST manipulation and annotation processing.
"""
from .base import BaseExtractor
from .code import extract_classes_and_functions_from_ast
from .functions import FunctionExtractor
from .classes import ClassExtractor
from .utils import add_parent_info, get_annotation

__all__ = [
    "BaseExtractor"
    "extract_classes_and_functions_from_ast",
    "FunctionExtractor",
    "ClassExtractor",
    "add_parent_info",
    "get_annotation"
]
