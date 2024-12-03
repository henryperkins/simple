# core/__init__.py
from .cache import Cache
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

# Import specific classes and functions from core/extraction
from .extraction.types import (
    ExtractedArgument, ExtractionContext, ExtractedElement, ExtractedFunction,
    ExtractedClass, ExtractionResult
)
from .extraction.utils import ASTUtils
from .extraction.code_extractor import CodeExtractor
from .extraction.function_extractor import FunctionExtractor
from .extraction.class_extractor import ClassExtractor
from .extraction.dependency_analyzer import DependencyAnalyzer

__all__ = [
    'Cache', 'AzureOpenAIConfig', 'DocumentationError',
    'DocumentationContext', 'DocStringManager', 'DocstringData',
    'DocstringProcessor', 'LoggerSetup', 'log_debug', 'log_info',
    'log_warning', 'log_error', 'MarkdownConfig', 'MarkdownGenerator',
    'MetricsError', 'Metrics', 'SystemMonitor', 'MetricsCollector',
    'generate_hash', 'get_annotation', 'handle_exceptions', 'load_json_file',
    'ensure_directory', 'validate_file_path', 'create_error_result',
    'add_parent_info', 'get_file_stats', 'filter_files', 'get_all_files',
    'ProcessingResult', 'AIHandler',
    # New modules
    'ExtractedArgument', 'ExtractionContext', 'ExtractedElement',
    'ExtractedFunction', 'ExtractedClass', 'ExtractionResult', 'ASTUtils',
    'CodeExtractor', 'FunctionExtractor', 'ClassExtractor',
    'DependencyAnalyzer'
]
