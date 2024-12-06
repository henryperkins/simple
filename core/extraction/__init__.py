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
- ImportExtractor: Extract and analyze imports
"""

from core.extraction.code_extractor import CodeExtractor
from core.types import (
    ExtractedFunction,
    ExtractedClass,
    ExtractionResult,
    ExtractionContext
)

__all__ = [
    "CodeExtractor",
    "ExtractedFunction",
    "ExtractedClass", 
    "ExtractionResult",
    "ExtractionContext"
]