# core/ai_service.py
from typing import Optional, Dict, Any, List, Tuple
from openai import AsyncAzureOpenAI
import json
import ast
import asyncio
import aiohttp
from pathlib import Path
from core.config import AIConfig
from core.response_parsing import ResponseParsingService
from core.cache import Cache
from core.metrics import Metrics
from core.types import DocumentationContext, ExtractionResult, ExtractionContext
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.extraction.code_extractor import CodeExtractor
from core.docstring_processor import DocstringProcessor
from exceptions import ConfigurationError, ProcessingError
from api.token_management import TokenCounter
import uuid

class AIService:
    """
    Handles all AI interactions and API management.
    
    This class provides a unified interface for AI operations including:
    - API communication
    - Token management
    - Response processing
    - Cache management
    - Resource cleanup
    """

    def __init__(
        self,
        config: AIConfig | None = None,
        cache: Cache | None = None,
        metrics: Metrics | None = None,
        response_parser: ResponseParsingService | None = None,
    ) -> None:
        """Initialize the AI service."""
        # Initialize the logger with a correlation ID
        base_logger = LoggerSetup.get_logger(__name__)
        correlation_id = str(uuid.uuid4())  # Generate a unique correlation ID
        self.logger = CorrelationLoggerAdapter(base_logger, correlation_id=correlation_id)
        
        try:
            self.config = config or AIConfig.from_env()
            if not hasattr(self.config, 'model'):
                raise ConfigurationError("model is not defined in AIConfig")
            
            self._client = AsyncAzureOpenAI(
                api_key=self.config.api_key,
                azure_endpoint=self.config.endpoint,
                api_version=self.config.api_version
            )
            self.cache = cache or None
            self.metrics = metrics or Metrics()
            self.response_parser = response_parser or ResponseParsingService()
            self.token_counter = TokenCounter(self.config.model)
            self.docstring_processor = DocstringProcessor()
            
            self.logger.info("AI service initialized successfully", extra={'correlation_id': self.logger.correlation_id})
        except Exception as e:
            self.logger.error("Failed to initialize AI service", exc_info=True, extra={'correlation_id': self.logger.correlation_id})
            raise ConfigurationError(f"AI service initialization failed: {e}") from e

    async def process_code(self, source_code: str) -> Tuple[str, str] | None:
        """Process source code through the AI service."""
        try:
            self.logger.info("Starting code processing", extra={'correlation_id': self.logger.correlation_id})
            
            tree = ast.parse(source_code)
            context = self._create_extraction_context(source_code, Path("module_path_placeholder"))
            
            extraction_result = await self._extract_code(context)
            if not extraction_result:
                return None

            prompt = self._create_dynamic_prompt(extraction_result)
            ai_response = await self._interact_with_ai(prompt)
            
            parsed_response = await self.response_parser.parse_response(
                ai_response, 
                expected_format="docstring"
            )

            if not parsed_response.validation_success:
                self.logger.error("Failed to validate AI response", extra={'correlation_id': self.logger.correlation_id})
                return None

            result = await self._integrate_ai_response(
                parsed_response.content,
                extraction_result
            )

            self.logger.info("Code processing completed successfully", extra={'correlation_id': self.logger.correlation_id})
            return result

        except Exception as e:
            self.logger.error(f"Error processing code: {e}", exc_info=True, extra={'correlation_id': self.logger.correlation_id})
            return None

    def _create_extraction_context(self, source_code: str, module_path: Path) -> ExtractionContext:
        """Create an extraction context for code processing."""
        return ExtractionContext(source_code=source_code)

    async def _extract_code(self, context: ExtractionContext) -> ExtractionResult | None:
        """Extract code elements for AI processing."""
        try:
            code_extractor = CodeExtractor(context)
            if context.source_code is None:
                raise ProcessingError("Source code is None")
            extraction_result = await code_extractor.extract_code(context.source_code)
            if not extraction_result:
                raise ProcessingError("Code extraction failed")
            return extraction_result
        except Exception as e:
            self.logger.error(f"Error extracting code: {e}", exc_info=True, extra={'correlation_id': self.logger.correlation_id})
            raise

    async def _interact_with_ai(
        self, 
        prompt: str, 
        retries: int = 3,
        delay: int = 5
    ) -> str:
        """Interact with the AI model."""
        if self.cache:
            cache_key = f"ai_response:{hash(prompt)}"
            cached_response = await self.cache.get_cached_docstring(cache_key)
            if cached_response:
                return cached_response["content"]

        for attempt in range(retries):
            try:
                token_count = self.token_counter.estimate_tokens(prompt)
                request_params = {
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": "You are a Python documentation expert."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": min(
                        self.config.max_tokens - token_count,
                        1000
                    ),
                    "temperature": 0.7
                }

                response = await self._client.chat.completions.create(
                    model=str(request_params["model"]),
                    messages=[{"role": msg["role"], "content": msg["content"]} for msg in request_params.get("messages", [])],
                    max_tokens=int(request_params["max_tokens"]),
                    temperature=float(request_params["temperature"]) if isinstance(request_params["temperature"], (int, float)) else 0.7
                )
                response_content = response.choices[0].message.content

                if self.cache:
                    await self.cache.save_docstring(cache_key, {"content": response_content})

                if response_content is not None:
                    return response_content
                else:
                    raise ProcessingError("AI response content is None")

            except Exception as e:
                self.logger.error(f"AI interaction error on attempt {attempt + 1}: {e}", exc_info=True, extra={'correlation_id': self.logger.correlation_id})
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                else:
                    raise

    async def generate_docstring(
        self,
        func_name: str,
        is_class: bool,
        params: list[dict[str, Any]] | None = None,
        return_type: str = "Any",
        complexity_score: int = 0,
        existing_docstring: str = "",
        decorators: list[str] | None = None,
        exceptions: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Generate a docstring for a function or class."""
        params = params or []
        decorators = decorators or []
        exceptions = exceptions or []

        try:
            prompt = self._create_docstring_prompt({
                "name": func_name,
                "params": params,
                "returns": {"type": return_type},
                "complexity": complexity_score,
                "existing_docstring": existing_docstring,
                "decorators": decorators,
                "raises": exceptions,
                "is_class": is_class,
            })

            response = await self._interact_with_ai(prompt)
            parsed_response = await self.response_parser.parse_response(
                response,
                expected_format="docstring"
            )

            if not parsed_response.validation_success:
                raise ProcessingError("AI response validation failed")

            return parsed_response.content

        except Exception as e:
            self.logger.error(f"Error generating docstring: {e}", exc_info=True, extra={'correlation_id': self.logger.correlation_id})
            raise

    def _create_docstring_prompt(self, details: dict[str, Any]) -> str:
        """Create a prompt for generating docstrings."""
        return json.dumps(details, indent=2)

    def _create_dynamic_prompt(self, extraction_result: ExtractionResult) -> str:
        """Create a dynamic prompt for AI interaction."""
        try:
            module_docstring = extraction_result.module_docstring
            classes = extraction_result.classes
            functions = extraction_result.functions

            prompt = (
                "Generate enhanced documentation following this JSON schema:\n\n"
                "{\n"
                '  "summary": "Brief overview of the module",\n'
                '  "description": "Detailed explanation of functionality",\n'
                '  "args": [{"name": "param_name", "type": "param_type", "description": "param_desc"}],\n'
                '  "returns": {"type": "return_type", "description": "return_description"},\n'
                '  "raises": [{"exception": "error_type", "description": "error_description"}],\n'
                '  "complexity": integer\n'
                "}\n\n"
                f"Current Documentation:\n{json.dumps(module_docstring, indent=2)}\n\n"
                "Available Classes:\n"
                + "\n".join(f"- {cls.name}" for cls in classes)
                + "\n\nAvailable Functions:\n"
                + "\n".join(f"- {func.name}" for func in functions)
                + "\n\nProvide the enhanced documentation as a JSON object matching the schema."
            )
            return prompt
        except Exception as e:
            self.logger.error(f"Error creating prompt: {e}", exc_info=True, extra={'correlation_id': self.logger.correlation_id})
            raise

    async def _integrate_ai_response(
        self, 
        ai_response: dict[str, Any],
        extraction_result: ExtractionResult
    ) -> tuple[str, str]:
        """Integrate AI response into the source code."""
        try:
            ai_response = self._ensure_required_fields(ai_response)
            processed_response = [{
                "name": "__module__",
                "docstring": ai_response,
                "type": "Module"
            }]

            integration_result = self.docstring_processor.process_batch(
                processed_response,
                extraction_result.source_code
            )
            
            if not integration_result:
                raise ProcessingError("Docstring integration failed")

            code = integration_result.get("code", "")
            documentation = self._generate_markdown_documentation(
                ai_response,
                extraction_result
            )
            
            return code, documentation

        except Exception as e:
            self.logger.error(f"Error integrating AI response: {e}", exc_info=True, extra={'correlation_id': self.logger.correlation_id})
            raise

    def _ensure_required_fields(self, ai_response: dict[str, Any]) -> dict[str, Any]:
        """Ensure required fields are present in the AI response."""
        return ai_response

    def _generate_markdown_documentation(
        self, 
        ai_response: dict[str, Any],
        extraction_result: ExtractionResult
    ) -> str:
        """Generate markdown documentation from AI response."""
        return "Generated Markdown Documentation"

    async def test_connection(self) -> None:
        """Test the connection to the AI service."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.config.endpoint,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                    headers={"api-key": self.config.api_key}
                ) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        raise ConnectionError(f"Connection failed: {response_text}")
            self.logger.info("Connection test successful", extra={'correlation_id': self.logger.correlation_id})
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}", exc_info=True, extra={'correlation_id': self.logger.correlation_id})
            raise

    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self._client:
                await self._client.close()
            if self.cache:
                await self.cache.close()
            self.logger.info("AI service cleanup completed", extra={'correlation_id': self.logger.correlation_id})
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True, extra={'correlation_id': self.logger.correlation_id})

    async def __aenter__(self) -> "AIService":
        """Async context manager entry."""
        await self.test_connection()
        return self

    async def __aexit__(self, exc_type: BaseException, exc_val: BaseException, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()