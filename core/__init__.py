"""
Core package for documentation generation and code analysis.

This package provides the core functionality for:
- Code extraction and analysis
- Documentation generation
- AI services integration
- Metrics and monitoring
- Error handling and logging
- Configuration management

Main components:
- DocumentationOrchestrator: Main orchestrator for documentation generation
- CodeExtractor: Extract and analyze source code
- AIService: Interface with AI models
- MetricsCollector: Track metrics and usage
- LoggerSetup: Centralized logging configuration
"""

from core.logger import LoggerSetup
from core.docstring_processor import DocstringProcessor
from core.extraction.code_extractor import CodeExtractor
from core.docs import DocumentationOrchestrator
from core.ai_service import AIService
from core.response_parsing import ResponseParsingService
from core.metrics_collector import MetricsCollector
from core.monitoring import SystemMonitor
from core.dependency_injection import setup_dependencies
from core.prompt_manager import PromptManager
from core.metrics import Metrics
from core.markdown_generator import MarkdownGenerator
from core.cache import Cache
from core.config import Config, AppConfig, AIConfig
from core.console import (
    display_code_snippet,
    print_status,
    print_error,
    print_success,
    print_warning,
    print_info,
    print_debug,
    display_metrics,
    create_progress,
)
from core.exceptions import (
    ProcessingError,
    ValidationError,
    ConnectionError,
    ConfigurationError,
    ExtractionError,
    IntegrationError,
    LiveError,
    DocumentationError,
    WorkflowError,
    TokenLimitError,
    DocumentationGenerationError,
    APICallError,
    DataValidationError,
    PromptGenerationError,
    TemplateLoadingError,
    DependencyAnalysisError,
    MaintainabilityError,
)
from core.types import (
    DocstringData,
    ExtractedFunction,
    ExtractedClass,
    ExtractionResult,
    TokenUsage,
    ProcessingResult,
    DocumentationContext,
    MetricData,
    ParsedResponse,
    ExtractedElement,
    ExtractedArgument,
)

# Initialize logging first
LoggerSetup.configure()
logger = LoggerSetup.get_logger(__name__)

__version__ = "0.1.0"

__all__ = [
    "DocumentationOrchestrator",
    "CodeExtractor",
    "AIService",
    "DocstringProcessor",
    "ResponseParsingService",
    "MetricsCollector",
    "SystemMonitor",
    "setup_dependencies",
    "PromptManager",
    "Metrics",
    "MarkdownGenerator",
    "Cache",
    "Config",
    "AppConfig",
    "AIConfig",
    "display_code_snippet",
    "print_status",
    "print_error",
    "print_success",
    "print_warning",
    "print_info",
    "print_debug",
    "display_metrics",
    "create_progress",
    "ProcessingError",
    "ValidationError",
    "ConnectionError",
    "ConfigurationError",
    "ExtractionError",
    "IntegrationError",
    "LiveError",
    "DocumentationError",
    "WorkflowError",
    "TokenLimitError",
    "DocumentationGenerationError",
    "APICallError",
    "DocstringData",
    "ExtractedFunction",
    "ExtractedClass",
    "ExtractionResult",
    "TokenUsage",
    "ProcessingResult",
    "DocumentationContext",
    "MetricData",
    "ParsedResponse",
    "ExtractedElement",
    "ExtractedArgument",
    "DataValidationError",
    "ProcessingError",
    "LoggerSetup",
]
