# core/__init__.py
from .cache import Cache
from .code_extraction import (
    CodeExtractor, ExtractionContext, ExtractedArgument, ExtractedElement,
    ExtractedFunction, ExtractionResult, ExtractedClass
)
from .config import AzureOpenAIConfig
from .docs import DocumentationError, DocumentationContext, DocStringManager
from .docstring_processor import (
    DocstringData, DocstringProcessor
)
from .logger import LoggerSetup, log_debug, log_info, log_warning, log_error
from .markdown_generator import MarkdownConfig, MarkdownGenerator
from .metrics import MetricsError, Metrics
from .monitoring import SystemMonitor
from .metrics_collector import MetricsCollector
from .utils import (
    generate_hash, get_annotation, handle_exceptions, load_json_file,
    ensure_directory, validate_file_path, create_error_result, add_parent_info,
    get_file_stats, filter_files, get_all_files
)
from .types import ProcessingResult, AIHandler

__all__ = [
    'Cache', 'CodeExtractor', 'ExtractionContext', 'ExtractedArgument',
    'ExtractedElement', 'ExtractedFunction', 'ExtractionResult',
    'ExtractedClass', 'AzureOpenAIConfig', 'DocumentationError',
    'DocumentationContext', 'DocStringManager', 'DocstringData',
    'DocstringProcessor', 'LoggerSetup', 'log_debug', 'log_info',
    'log_warning', 'log_error', 'MarkdownConfig', 'MarkdownGenerator',
    'MetricsError', 'Metrics', 'SystemMonitor', 'MetricsCollector',
    'generate_hash', 'get_annotation', 'handle_exceptions', 'load_json_file',
    'ensure_directory', 'validate_file_path', 'create_error_result',
    'add_parent_info', 'get_file_stats', 'filter_files', 'get_all_files',
    'ProcessingResult', 'AIHandler'
]
