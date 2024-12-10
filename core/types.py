"""
Core type definitions for Python code analysis and documentation generation.
Handles all data structures needed for code extraction, analysis, metrics,
and documentation generation throughout the application.
"""

from core.types.base import (
    Injector,
    MetricData,
    DocstringData,
    DocumentationContext,
    ExtractionContext,
    ExtractedArgument,
    ExtractedElement,
    ExtractedFunction,
    ExtractedClass,
    ProcessingResult,
    DocumentationData
)

__all__ = [
    "Injector",
    "MetricData", 
    "DocstringData",
    "DocumentationContext",
    "ExtractionContext",
    "ExtractedArgument",
    "ExtractedElement",
    "ExtractedFunction",
    "ExtractedClass",
    "ProcessingResult",
    "DocumentationData"
]
