"""
Core type definitions for Python code analysis and documentation generation.
Handles all data structures needed for code extraction, analysis, metrics,
and documentation generation throughout the application.
"""

from core.types import (
    MetricData,
    BaseData,
    ParsedResponse,
    DocstringData,
    TokenUsage,
    ExtractedArgument,
    ExtractedElement,
    ExtractedFunction,
    ExtractedClass,
    ExtractionResult,
    ProcessingResult,
    DocumentationContext,
    ExtractionContext,
    DocumentationData
)

__all__ = [
    "MetricData",
    "BaseData",
    "ParsedResponse",
    "DocstringData",
    "TokenUsage",
    "ExtractedArgument",
    "ExtractedElement",
    "ExtractedFunction",
    "ExtractedClass",
    "ExtractionResult",
    "ProcessingResult",
    "DocumentationContext",
    "ExtractionContext",
    "DocumentationData"
]
