"""Core functionality for documentation generation."""

# Import from dedicated modules
from .cache import Cache
from .config import AzureOpenAIConfig
from .docs import DocumentationError, DocumentationContext
from .docstring_processor import DocstringData, DocstringProcessor
from .logger import LoggerSetup, log_debug, log_info, log_warning, log_error
from .markdown_generator import MarkdownConfig, MarkdownGenerator
from .metrics import Metrics
from .monitoring import SystemMonitor
from .response_parsing import ResponseParsingService
from .schema_loader import load_schema
from .types import *  # Types are an exception - OK to import * for type definitions

__version__ = "1.0.0"
__all__ = [
    # Core components
    'Cache', 'AzureOpenAIConfig', 'DocumentationError',
    'DocumentationContext', 'DocstringData', 'DocstringProcessor',
    'LoggerSetup', 'log_debug', 'log_info', 'log_warning', 'log_error',
    'MarkdownConfig', 'MarkdownGenerator', 'Metrics', 'SystemMonitor',
    'ResponseParsingService', 'load_schema',
]