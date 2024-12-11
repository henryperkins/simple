from typing import Optional
from core.metrics import Metrics
from core.docstring_processor import DocstringProcessor
from core.types.base import Injector
from core.metrics_collector import MetricsCollector
from api.token_management import TokenManager
from core.response_parsing import ResponseParsingService
from core.logger import LoggerSetup
from core.extraction.dependency_analyzer import DependencyAnalyzer
from core.ai_service import AIService
from core.config import Config, AIConfig
from core.cache import Cache
import asyncio
from core.prompt_manager import PromptManager
from core.extraction.code_extractor import CodeExtractor
from core.markdown_generator import MarkdownGenerator

def setup_dependencies(correlation_id: Optional[str] = None):
    Injector.register('metrics_calculator', lambda: Metrics())
    Injector.register('docstring_processor', lambda: DocstringProcessor())
    Injector.register('config', lambda: Config())
    Injector.register('token_manager', lambda: TokenManager(model="gpt-4"))
    Injector.register('response_parser', lambda: ResponseParsingService(correlation_id=correlation_id))
    Injector.register('prompt_manager', lambda: PromptManager(correlation_id=correlation_id))
    Injector.register('code_extractor', lambda: CodeExtractor())
    Injector.register('markdown_generator', lambda: MarkdownGenerator())
    Injector.register('cache', lambda: Cache())
    Injector.register('semaphore', lambda: asyncio.Semaphore(5))
    Injector.register('config', lambda: Config())
    Injector.register('markdown_generator', lambda: MarkdownGenerator())
    Injector.register('ai_service', lambda: AIService(config=Injector.get('config').ai, correlation_id=correlation_id))
    Injector.register('logger', lambda: LoggerSetup.get_logger(__name__))
    Injector.register('docstring_parser', lambda: DocstringProcessor())
    Injector.register('dependency_analyzer', lambda: DependencyAnalyzer)
