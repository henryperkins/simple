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
- System monitoring

Main components:
- DocstringProcessor: Process and validate docstrings
- ResponseParsingService: Parse and validate AI responses  
- CodeExtractor: Extract code elements from source
- Cache: Caching system for generated content
- Metrics: Performance and usage metrics
- DocumentationOrchestrator: Orchestrate documentation generation
- AIService: Service for interacting with AI for documentation generation
- MetricsCollector: Collects and manages metrics
- SystemMonitor: Monitors system resources and performance
"""

from core.docstring_processor import DocstringProcessor
from core.response_parsing import ResponseParsingService
from core.extraction.code_extractor import CodeExtractor
from core.cache import Cache
from core.metrics import Metrics
from core.logger import LoggerSetup
from core.docs import DocumentationOrchestrator
from core.ai_service import AIService
from core.types import (
    DocstringData,
    ExtractedFunction,
    ExtractedClass,
    ExtractionResult,
    ParsedResponse,
    ProcessingResult,
)
from core.metrics_collector import MetricsCollector
from core.monitoring import SystemMonitor

__version__ = "0.1.0"

__all__ = [
    "DocstringProcessor",
    "ResponseParsingService",
    "CodeExtractor",
    "Cache",
    "Metrics",
    "LoggerSetup",
    "DocumentationOrchestrator",
    "AIService",
    "DocstringData",
    "ExtractedFunction",
    "ExtractedClass",
    "ExtractionResult",
    "ParsedResponse",
    "ProcessingResult",
    "MetricsCollector",
    "SystemMonitor",
]
