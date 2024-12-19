"""Manages dependency injection for classes."""

from typing import Any

from core.metrics_collector import MetricsCollector
from core.metrics import Metrics
from api.token_management import TokenManager
from core.docstring_processor import DocstringProcessor
from core.markdown_generator import MarkdownGenerator
from core.response_parsing import ResponseParsingService
from core.prompt_manager import PromptManager
from core.config import Config
from core.ai_service import AIService
from core.extraction.function_extractor import FunctionExtractor
from core.extraction.dependency_analyzer import DependencyAnalyzer
from core.extraction.code_extractor import CodeExtractor
from core.logger import LoggerSetup
from core.types.base import ExtractionContext
from core.docs import DocumentationOrchestrator


class Injector:
    """Manages dependency injection for classes."""

    _dependencies: dict[str, Any] = {}
    _initialized: bool = False
    _logger: Any = None

    @classmethod
    def _get_logger(cls) -> Any:
        """Get or initialize the logger."""
        if cls._logger is None:
            cls._logger = LoggerSetup.get_logger(__name__)
        return cls._logger

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
        cls._get_logger().info(f"Dependency '{name}' registered")

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
        cls._get_logger().info("All dependencies cleared")

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the injector is initialized."""
        return cls._initialized

    @classmethod
    def set_initialized(cls, value: bool) -> None:
        """Set the initialization status."""
        cls._initialized = value


async def setup_dependencies(config: Config, correlation_id: str | None = None) -> None:
    """
    Sets up the dependency injection framework by registering all components in the proper order.

    Args:
        config: Configuration object containing app and AI settings.
        correlation_id: Unique identifier for logging and correlation.
    """
    # Avoid reinitialization
    if Injector.is_initialized():
        return

    Injector.clear()
    logger = LoggerSetup.get_logger(__name__)
    logger.info("Starting dependency injection setup.")

    try:
        # 1. Register core configuration and correlation ID
        Injector.register("config", config)
        Injector.register("correlation_id", correlation_id)
        logger.debug("Registered 'config' and 'correlation_id'.")

        # 2. Register core utilities and services
        metrics_collector = MetricsCollector(correlation_id=correlation_id)
        Injector.register("metrics_collector", metrics_collector)
        logger.debug("Registered 'metrics_collector'.")

        metrics = Metrics(
            metrics_collector=metrics_collector, correlation_id=correlation_id
        )
        Injector.register("metrics_calculator", metrics, force=True)
        logger.debug("Registered 'metrics_calculator'.")

        token_manager = TokenManager(
            model=config.ai.model,
            config=config.ai,
            correlation_id=correlation_id,
            metrics_collector=metrics_collector,
        )
        Injector.register("token_manager", token_manager)
        logger.debug("Registered 'token_manager'.")

        # Register read_file_safe_async
        from utils import read_file_safe_async
        Injector.register("read_file_safe_async", read_file_safe_async)
        logger.debug("Registered 'read_file_safe_async'.")

        # Register logger
        logger_instance = LoggerSetup.get_logger("ClassExtractor")
        Injector.register("logger", logger_instance)
        logger.debug("Registered 'logger'.")

        # 3. Register processors and validators
        docstring_processor = DocstringProcessor(correlation_id=correlation_id)
        response_formatter = ResponseParsingService(correlation_id=correlation_id)
        Injector.register("docstring_processor", docstring_processor)
        Injector.register("response_formatter", response_formatter)

        logger.debug("Registered processors and validators.")

        markdown_generator = MarkdownGenerator(correlation_id=correlation_id)
        Injector.register("markdown_generator", markdown_generator)
        logger.debug("Registered 'markdown_generator'.")

        # Initialize response parser and prompt manager
        response_parser = ResponseParsingService(correlation_id=correlation_id)
        Injector.register("response_parser", response_parser)
        logger.debug("Registered 'response_parser'.")

        prompt_mgr = PromptManager(correlation_id=correlation_id)
        Injector.register("prompt_manager", prompt_mgr)
        logger.debug("Registered 'prompt_manager'.")

        # 4. Initialize AI service
        ai_service = AIService(config=config.ai, correlation_id=correlation_id)
        Injector.register("ai_service", ai_service)
        logger.debug("Registered 'ai_service'.")

        # 5. Initialize code extraction components
        extraction_context = ExtractionContext(
            module_name="default_module",
            base_path=config.project_root,
            include_private=False,
            include_nested=False,
            include_magic=True,
            docstring_processor=docstring_processor,
            metrics_collector=metrics_collector,
        )

        from core.extraction.class_extractor import ClassExtractor  # Local import
        function_extractor = FunctionExtractor(
            context=extraction_context, correlation_id=correlation_id
        )
        class_extractor = ClassExtractor(
            context=extraction_context, correlation_id=correlation_id
        )
        dependency_analyzer = DependencyAnalyzer(
            context=extraction_context, correlation_id=correlation_id
        )
        code_extractor = CodeExtractor(
            context=extraction_context, correlation_id=correlation_id
        )

        # Update extraction context
        extraction_context.function_extractor = function_extractor
        extraction_context.dependency_analyzer = dependency_analyzer

        # Register extraction components
        Injector.register("extraction_context", extraction_context)
        Injector.register("function_extractor", function_extractor)
        Injector.register("class_extractor", class_extractor)
        Injector.register("dependency_analyzer", dependency_analyzer)
        Injector.register("code_extractor", code_extractor)
        logger.debug("Registered code extraction components.")

        # 6. Register orchestrator
        doc_orchestrator = DocumentationOrchestrator(
            ai_service=ai_service,
            code_extractor=code_extractor,
            markdown_generator=markdown_generator,
            prompt_manager=prompt_mgr,
            docstring_processor=docstring_processor,
            response_parser=response_parser,
            correlation_id=correlation_id,
        )
        Injector.register("doc_orchestrator", doc_orchestrator)
        logger.debug("Registered 'doc_orchestrator'.")

        # Finalize initialization
        Injector.set_initialized(True)
        logger.info("Dependency injection setup complete.")
        logger.debug(f"Registered dependencies: {list(Injector._dependencies.keys())}")

    except Exception as e:
        logger.error(f"Error during dependency injection setup: {e}", exc_info=True)
        raise
    finally:
        logger.info("Dependency injection setup attempt completed.")
