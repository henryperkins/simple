# core/extraction/__init__.py
"""
Code extraction subpackage for analyzing Python source code.

This subpackage provides functionality for:
- Extracting functions, classes and methods from source code
- Analyzing code complexity and dependencies
- Parsing docstrings and type hints
- Building inheritance hierarchies
- Generating code metadata

Main components:
- CodeExtractor: Main class for code extraction
- ClassExtractor: Extract class definitions and methods
- FunctionExtractor: Extract function and method definitions
- DependencyAnalyzer: Analyze code dependencies
- extract_dependencies_from_node: Extract dependencies from an AST node
"""

from core.extraction.code_extractor import CodeExtractor
from core.extraction.class_extractor import ClassExtractor
from core.extraction.function_extractor import FunctionExtractor
from core.extraction.dependency_analyzer import DependencyAnalyzer

__all__ = [
    "CodeExtractor",
    "ClassExtractor",
    "FunctionExtractor",
    "DependencyAnalyzer"
]
