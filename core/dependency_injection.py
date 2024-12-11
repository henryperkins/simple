from typing import Optional
from core.metrics import Metrics
from core.docstring_processor import DocstringProcessor
from core.types.base import Injector
from core.metrics_collector import MetricsCollector
from api.token_management import TokenManager
from core.response_parsing import ResponseParsingService
from core.ai_service import AIService
from core.config import AIConfig
from core.prompt_manager import PromptManager

def setup_dependencies(correlation_id: Optional[str] = None):
    Injector.register('metrics_calculator', lambda: Metrics(metrics_collector=MetricsCollector(correlation_id=correlation_id)))
    Injector.register('docstring_parser', lambda: DocstringProcessor())
    Injector.register('token_manager', lambda: TokenManager(model="gpt-4"))
    Injector.register('response_parser', lambda: ResponseParsingService(correlation_id=correlation_id))
    Injector.register('prompt_manager', lambda: PromptManager(correlation_id=correlation_id))
    Injector.register('ai_service', lambda: AIService(config=AIConfig(), correlation_id=correlation_id))
