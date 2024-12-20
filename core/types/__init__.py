"""
Core type definitions for Python code analysis and documentation generation.
Handles all data structures needed for code extraction, analysis, metrics,
and documentation generation throughout the application.
"""

from core.types.docstring import DocstringData
from core.types.base import (
    DocumentationData,
    MetricData,
    ExtractedArgument,
    ExtractedElement,
    ExtractionContext,
    ExtractionResult,
    ExtractedFunction,
    ExtractedClass,
    ParsedResponse,
    TokenUsage,
    ProcessingResult,
    DocumentationContext,
)

__all__ = [
    "DocumentationData",
    "DocumentationContext",
    "DocstringData",
    "ExtractedArgument",
    "ExtractedElement",
    "ExtractionContext",
    "ExtractedFunction",
    "MetricData",
    "ExtractionResult",
    "ExtractedClass",
    "ParsedResponse",
    "TokenUsage",
    "ProcessingResult",
    "DocstringData",
]
