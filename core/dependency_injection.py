"""Manages dependency injection for classes."""

from typing import Optional, Any, Dict
import logging
from core.config import Config


class Injector:
    """Manages dependency injection for classes."""

    _dependencies: Dict[str, Any] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, name: str, dependency: Any, force: bool = False) -> None:
        """Register a dependency with a name.

        Args:
            name: The name to register the dependency under.
            dependency: The dependency instance to register.
            force: Whether to overwrite an existing dependency.
        """
        if name in cls._dependencies and not force:
            raise ValueError(
                f"Dependency '{name}' already registered. Use force=True to overwrite."
            )

        cls._dependencies[name] = dependency
        logger = cls._get_logger()
        if logger:
            logger.info(f"Dependency '{name}' registered")

    @classmethod
    def get(cls, name: str) -> Any:
        """Retrieve a dependency by name."""
        if name not in cls._dependencies:
            error_message = (
                f"Dependency '{name}' not found. Available dependencies: "
                f"{list(cls._dependencies.keys())}"
            )
            raise KeyError(error_message)
        return cls._dependencies[name]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a dependency is registered."""
        return name in cls._dependencies

    @classmethod
    def clear(cls) -> None:
        """Clear all registered dependencies."""
        cls._dependencies.clear()
        cls._initialized = False
        logger = cls._get_logger()
        if logger:
            logger.info("All dependencies cleared")

    @classmethod
    def _get_logger(cls) -> Optional[logging.Logger]:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


def setup_dependencies(config: Config, correlation_id: Optional[str] = None):
    """Sets up the dependency injection framework."""
    if Injector._initialized:
        return

    # Clear existing dependencies first
    Injector.clear()

    # Register core dependencies
    Injector.register("config", config)
    Injector.register("correlation_id", correlation_id)

    # Import dependencies
    from core.logger import LoggerSetup
    from core.metrics_collector import MetricsCollector
    from core.metrics import Metrics
    from core.docstring_processor import DocstringProcessor
    from core.response_parsing import ResponseParsingService
    from core.prompt_manager import PromptManager
    from core.markdown_generator import MarkdownGenerator
    from core.cache import Cache
    from core.ai_service import AIService
    from api.token_management import TokenManager
    from core.extraction.function_extractor import FunctionExtractor
    from core.extraction.class_extractor import ClassExtractor
    from core.extraction.code_extractor import CodeExtractor
    from core.extraction.dependency_analyzer import DependencyAnalyzer
    from core.types.base import ExtractionContext
    import asyncio

    # Create instances instead of registering factory functions
    logger = LoggerSetup.get_logger(__name__)
    Injector.register("logger", logger)

    # Create and register MetricsCollector instance
    metrics_collector = MetricsCollector(correlation_id=correlation_id)
    Injector.register("metrics_collector", metrics_collector)

    # Create and register Metrics instance
    metrics = Metrics(metrics_collector=metrics_collector, correlation_id=correlation_id)
    Injector.register("metrics_calculator", metrics, force=True)  # Force overwrite if needed

    # Create and register TokenManager instance
    token_manager = TokenManager(
        model=config.ai.model,
        config=config.ai,
        correlation_id=correlation_id,
        metrics_collector=metrics_collector
    )
    Injector.register("token_manager", token_manager)

    # Create and register other instances
    docstring_processor = DocstringProcessor(metrics=metrics)
    Injector.register("docstring_processor", docstring_processor)

    response_parser = ResponseParsingService(correlation_id=correlation_id)
    Injector.register("response_parser", response_parser)

    prompt_manager = PromptManager(correlation_id=correlation_id)
    Injector.register("prompt_manager", prompt_manager)

    extraction_context = ExtractionContext()
    Injector.register("extraction_context", extraction_context)

    function_extractor = FunctionExtractor(context=extraction_context, correlation_id=correlation_id)
    Injector.register("function_extractor", function_extractor)

    class_extractor = ClassExtractor(context=extraction_context, correlation_id=correlation_id)
    Injector.register("class_extractor", class_extractor)

    dependency_analyzer = DependencyAnalyzer(context=extraction_context, correlation_id=correlation_id)
    Injector.register("dependency_analyzer", dependency_analyzer)

    code_extractor = CodeExtractor(context=extraction_context, correlation_id=correlation_id)
    Injector.register("code_extractor", code_extractor)

    markdown_generator = MarkdownGenerator()
    Injector.register("markdown_generator", markdown_generator)

    cache = Cache()
    Injector.register("cache", cache)

    semaphore = asyncio.Semaphore(5)
    Injector.register("semaphore", semaphore)

    ai_service = AIService(config=config.ai, correlation_id=correlation_id)
    Injector.register("ai_service", ai_service)

    from core.docs import DocumentationOrchestrator
    doc_orchestrator = DocumentationOrchestrator(
        ai_service=ai_service,
        code_extractor=code_extractor,
        markdown_generator=markdown_generator,
        prompt_manager=prompt_manager,
        docstring_processor=docstring_processor,
        response_parser=response_parser,
        correlation_id=correlation_id
    )
    Injector.register("doc_orchestrator", doc_orchestrator)

    Injector._initialized = True
