"""
AI Interaction Handler Module

Manages interactions with Azure OpenAI API, handling token management,
caching, and response processing for documentation generation.
"""

from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import json
from dataclasses import dataclass

from core.logger import LoggerSetup, log_error, log_info, log_debug
from core.cache import Cache
from core.monitoring import MetricsCollector
from core.config import AzureOpenAIConfig
from core.docstring_processor import (
    DocstringProcessor,
    DocstringData,
    DocstringMetrics
)
from api.token_management import TokenManager
from api.api_client import AzureOpenAIClient
from exceptions import ValidationError, ProcessingError, CacheError

logger = LoggerSetup.get_logger(__name__)
config = AzureOpenAIConfig.from_env()

@dataclass
class ProcessingResult:
    """Result of AI processing operation."""
    content: str
    usage: Dict[str, Any]
    metrics: Optional[DocstringMetrics] = None
    cached: bool = False
    processing_time: float = 0.0

class AIInteractionHandler:
    """
    Handles AI interactions for documentation generation via Azure OpenAI API.

    Manages token limits, caching mechanisms, and metrics collection for robust processing.
    """

    def __init__(
        self,
        cache: Optional[Cache] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        token_manager: Optional[TokenManager] = None
    ):
        """Initialize the AI Interaction Handler."""
        try:
            self.cache = cache
            self.metrics = metrics_collector
            self.token_manager = token_manager or TokenManager(
                model=config.model_name,
                deployment_name=config.deployment_name
            )
            self.client = AzureOpenAIClient(
                token_manager=self.token_manager,
                metrics_collector=metrics_collector
            )
            self.docstring_processor = DocstringProcessor()
            self._initialize_tools()
            logger.info("AI Interaction Handler initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AI Interaction Handler: {str(e)}")
            raise

    def _initialize_tools(self) -> None:
        """Initialize the function tools for structured output."""
        self.docstring_tool = {
            "type": "function",
            "function": {
                "name": "generate_docstring",
                "description": "Generate a Python docstring with structured information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "A concise summary of what the code does"
                        },
                        "description": {
                            "type": "string",
                            "description": "A detailed description of the functionality"
                        },
                        "args": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": "string"},
                                    "description": {"type": "string"}
                                },
                                "required": ["name", "type", "description"]
                            },
                            "description": "List of arguments with their types and descriptions"
                        },
                        "returns": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["type", "description"]
                        },
                        "raises": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "description": {"type": "string"}
                                }
                            },
                            "description": "List of exceptions that may be raised"
                        }
                    },
                    "required": ["summary", "description", "args", "returns"]
                }
            }
        }

    def _create_documentation_prompt(
        self,
        source_code: str,
        metadata: Dict[str, Any],
        node: Optional[ast.AST] = None
    ) -> str:
        """
        Create a dynamic prompt based on the code and metadata.

        Args:
            source_code: Source code to document
            metadata: Code metadata
            node: Optional AST node for context

        Returns:
            str: Generated prompt
        """
        docstring_data = None
        if node:
            docstring_data = self.docstring_processor.process_node(node, source_code)

        prompt_parts = [
            "Generate comprehensive documentation for the following Python code.",
            "\nCode Analysis:",
            f"- Complexity Metrics:\n{self._format_metrics(metadata)}",
        ]

        if docstring_data and docstring_data.metrics:
            prompt_parts.append(
                f"- Docstring Metrics:\n{self._format_docstring_metrics(docstring_data.metrics)}"
            )

        if docstring_data and docstring_data.extraction_context:
            prompt_parts.append(
                f"\nCode Context:\n{self._format_context(docstring_data.extraction_context)}"
            )

        if metadata.get('docstring'):
            prompt_parts.append(f"\nExisting Documentation:\n{metadata['docstring']}")

        if metadata.get('class_info'):
            prompt_parts.append(
                f"\nClass Information:\n{self._format_class_info(metadata['class_info'])}"
            )

        prompt_parts.extend([
            "\nRequirements:",
            "- Follow Google Style Python docstring format",
            "- Include comprehensive parameter descriptions",
            "- Document return values and types",
            "- List possible exceptions",
            "- Add usage examples for complex functionality",
            f"\nSource Code:\n{source_code}",
            "\nProvide the documentation in a structured format using the specified function schema."
        ])

        return "\n".join(prompt_parts)

    def _format_metrics(self, metadata: Dict[str, Any]) -> str:
        """Format code metrics into a string."""
        lines = []
        for func in metadata.get('functions', []):
            metrics = func.get('metrics', {})
            lines.append(
                f"Function {func['name']}:\n"
                f"  - Cyclomatic Complexity: {metrics.get('cyclomatic_complexity')}\n"
                f"  - Cognitive Complexity: {metrics.get('cognitive_complexity')}\n"
                f"  - Maintainability Index: {metrics.get('maintainability_index')}"
            )
        return '\n'.join(lines)

    def _format_docstring_metrics(self, metrics: DocstringMetrics) -> str:
        """Format docstring metrics into a string."""
        return (
            f"  Completeness: {metrics.completeness_score:.1f}%\n"
            f"  Cognitive Complexity: {metrics.cognitive_complexity:.1f}\n"
            f"  Sections: {metrics.sections_count}\n"
            f"  Arguments Documented: {metrics.args_count}"
        )

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format extraction context into a string."""
        if context.get('type') == 'function':
            return (
                f"Function: {context['name']}\n"
                f"Arguments: {self._format_args(context['args'])}\n"
                f"Returns: {context['returns']}\n"
                f"Complexity: {context['complexity']}"
            )
        elif context.get('type') == 'class':
            return (
                f"Class: {context['name']}\n"
                f"Bases: {', '.join(context['bases'])}\n"
                f"Methods: {len(context['methods'])}"
            )
        return str(context)

    def _format_args(self, args: List[Dict[str, str]]) -> str:
        """Format function arguments into a string."""
        return ', '.join(
            f"{arg['name']}: {arg['type']}"
            + (f" = {arg['default']}" if arg.get('default') else "")
            for arg in args
        )

    def _format_class_info(self, class_info: Dict[str, Any]) -> str:
        """Format class information into a string."""
        info_parts = [
            f"Class Name: {class_info.get('name', 'Unknown')}",
            f"Base Classes: {', '.join(class_info.get('bases', []))}",
            f"Methods: {len(class_info.get('methods', []))}"
        ]
        return '\n'.join(info_parts)

    async def _generate_documentation(
        self,
        source_code: str,
        metadata: Dict[str, Any],
        node: Optional[ast.AST] = None
    ) -> Optional[ProcessingResult]:
        """Generate documentation using Azure OpenAI with function calling."""
        try:
            prompt = self._create_documentation_prompt(source_code, metadata, node)
            start_time = datetime.now()

            response, usage = await self.client.process_request(
                prompt=prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                tools=[self.docstring_tool],
                tool_choice={"type": "function", "function": {"name": "generate_docstring"}}
            )

            if not response:
                return None

            processing_time = (datetime.now() - start_time).total_seconds()

            if response.get("tool_calls"):
                tool_call = response["tool_calls"][0]
                if tool_call.function.name == "generate_docstring":
                    try:
                        docstring_data = DocstringData(**json.loads(tool_call.function.arguments))
                        
                        # Validate using docstring processor
                        is_valid, errors = self.docstring_processor.validate(
                            docstring_data,
                            docstring_data.extraction_context
                        )
                        if not is_valid:
                            raise ProcessingError(f"Invalid documentation: {errors}")

                        # Calculate metrics if node is provided
                        metrics = None
                        if node:
                            metrics = self.docstring_processor._calculate_metrics(docstring_data, node)

                        return ProcessingResult(
                            content=self.docstring_processor.format(docstring_data),
                            usage=usage or {},
                            metrics=metrics,
                            processing_time=processing_time
                        )
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse function call response: {e}")
                        raise ProcessingError("Invalid JSON in function response")

            raise ProcessingError("No valid function call response received")

        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}")
            raise ProcessingError(f"Failed to generate documentation: {str(e)}")

    async def process_code(
        self,
        source_code: str,
        node: Optional[ast.AST] = None,
        cache_key: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Process source code to generate and embed documentation.

        Args:
            source_code: Source code to process
            node: Optional AST node for context
            cache_key: Optional cache key

        Returns:
            Tuple[str, str]: Updated code and documentation
        """
        operation_start = datetime.now()

        try:
            if not source_code or not source_code.strip():
                raise ValidationError("Empty source code provided")

            # Check cache if enabled
            if self.cache and cache_key:
                try:
                    cached_result = await self._check_cache(cache_key)
                    if cached_result:
                        return cached_result
                except CacheError as e:
                    logger.warning(f"Cache error, proceeding without cache: {str(e)}")

            # Extract metadata
            metadata = self.docstring_processor.process_node(
                node or ast.parse(source_code),
                source_code
            ).extraction_context

            # Generate documentation
            result = await self._generate_documentation(source_code, metadata, node)
            if not result or not result.content:
                raise ProcessingError("Documentation generation failed")

            # Update code with documentation
            updated_code = await self._update_code(source_code, result.content)
            if not updated_code:
                raise ProcessingError("Code update failed")

            # Cache result if enabled
            if self.cache and cache_key:
                await self._cache_result(cache_key, updated_code, result.content)

            # Track metrics
            await self._track_metrics(operation_start, True, result.usage)

            return updated_code, result.content

        except Exception as e:
            await self._track_error("ProcessingError", e, operation_start)
            logger.error(f"Process code failed: {str(e)}")
            raise

    async def _update_code(self, source_code: str, documentation: str) -> Optional[str]:
        """Update source code with generated documentation."""
        try:
            return f'"""\n{documentation}\n"""\n\n{source_code}'
        except Exception as e:
            logger.error(f"Code update failed: {str(e)}")
            raise ProcessingError(f"Failed to update code: {str(e)}")

    async def _cache_result(
        self,
        cache_key: str,
        code: str,
        documentation: str
    ) -> None:
        """Cache the processing result."""
        try:
            await self.cache.save_docstring(
                cache_key,
                {
                    'code': code,
                    'docs': documentation
                }
            )
            logger.debug(f"Cached result for key: {cache_key}")
        except Exception as e:
            logger.error(f"Caching failed: {str(e)}")

    async def _track_metrics(
        self,
        start_time: datetime,
        success: bool,
        usage: Dict[str, Any]
    ) -> None:
        """Track operation metrics."""
        if self.metrics:
            duration = (datetime.now() - start_time).total_seconds()
            await self.metrics.track_operation(
                operation_type='documentation_generation',
                success=success,
                duration=duration,
                usage=usage
            )
        log_info(f"Operation metrics: success={success}, duration={duration}, usage={usage}")

    async def _track_error(
        self,
        error_type: str,
        error: Exception,
        start_time: datetime
    ) -> None:
        """Track and log error metrics."""
        if self.metrics:
            duration = (datetime.now() - start_time).total_seconds()
            await self.metrics.track_operation(
                operation_type='documentation_generation',
                success=False,
                duration=duration,
                error=f"{error_type}: {str(error)}"
            )

    async def _check_cache(
        self,
        cache_key: str
    ) -> Optional[Tuple[str, str]]:
        """Check if the result is cached."""
        try:
            cached_data = await self.cache.get_cached_docstring(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_data['code'], cached_data['docs']
            logger.debug(f"Cache miss for key: {cache_key}")
            return None
        except Exception as e:
            logger.error(f"Error checking cache: {str(e)}")
            raise CacheError(f"Cache check failed: {str(e)}")

    async def close(self) -> None:
        """Close the AI interaction handler and cleanup resources."""
        try:
            await self.client.close()
            logger.info("AI Interaction Handler closed successfully")
        except Exception as e:
            logger.error(f"Error closing AI handler: {str(e)}")

    async def __aenter__(self) -> 'AIInteractionHandler':
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any]
    ) -> None:
        """Async context manager exit."""
        await self.close()