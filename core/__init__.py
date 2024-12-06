# core/__init__.py
"""
Core package for documentation generation and code analysis.

This package provides the core functionality for:
- Code extraction and analysis
- Documentation generation
- Response parsing
- Schema validation
- Metrics collection
- Caching
- Configuration management

Main components:
- DocstringProcessor: Process and validate docstrings
- ResponseParsingService: Parse and validate AI responses  
- CodeExtractor: Extract code elements from source
- Cache: Caching system for generated content
- Metrics: Performance and usage metrics
- DocumentationOrchestrator: Orchestrate documentation generation
"""

from core.docstring_processor import DocstringProcessor
from core.response_parsing import ResponseParsingService
from core.extraction.code_extractor import CodeExtractor
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.metrics import Metrics
from core.logger import LoggerSetup
from core.docs import DocumentationOrchestrator
from core.schema_loader import load_schema
from core.types import (
    DocstringData,
    ExtractedFunction,
    ExtractedClass,
    ExtractionResult,
    ParsedResponse,
    ProcessingResult,
)

__version__ = "0.1.0"

__all__ = [
    "DocstringProcessor",
    "ResponseParsingService", 
    "CodeExtractor",
    "Cache",
    "AzureOpenAIConfig",
    "Metrics",
    "LoggerSetup",
    "DocumentationOrchestrator",
    "load_schema",
    "DocstringData",
    "ExtractedFunction",
    "ExtractedClass", 
    "ExtractionResult",
    "ParsedResponse",
    "ProcessingResult",
]