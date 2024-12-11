"""
AI service module for interacting with the AI model.
"""

from typing import Dict, Any, Optional
import asyncio
import aiohttp
from urllib.parse import urljoin
from datetime import datetime

from core.config import AIConfig
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.types.base import Injector
from core.types.base import DocumentationContext, ProcessingResult
from core.docstring_processor import DocstringProcessor
from core.prompt_manager import PromptManager
from core.response_parsing import ResponseParsingService
from api.token_management import TokenManager
from core.console import (
    print_info,
    print_error,
    print_warning,
    display_metrics,
    display_metrics
)

class AIService:
    """Service for interacting with the AI model."""

    def __init__(self, config: Optional[AIConfig] = None, correlation_id: Optional[str] = None) -> None:
        """
        Initialize AI service.

        Args:
            config: AI service configuration.
            correlation_id: Optional correlation ID for tracking related operations.
        """
        self.config = config or Injector.get('config')().ai
        self.correlation_id = correlation_id
        self.logger = CorrelationLoggerAdapter(LoggerSetup.get_logger(__name__))
        self.prompt_manager: PromptManager = Injector.get('prompt_manager')
        self.response_parser: ResponseParsingService = Injector.get('response_parser')
        try:
            self.docstring_processor = Injector.get('docstring_processor')
        except KeyError:
            self.logger.warning("Docstring processor not registered, using default")
            self.docstring_processor = DocstringProcessor()
            Injector.register('docstring_processor', self.docstring_processor)
        self.token_manager: TokenManager = Injector.get('token_manager')
        self.semaphore = Injector.get('semaphore')
        self._client = None

        print_info("Initializing AI service")

    async def generate_documentation(self, context: DocumentationContext) -> ProcessingResult:
        """
        Generates documentation using the AI model.

        Args:
            context: Documentation context containing source code and metadata.

        Returns:
            ProcessingResult containing enhanced documentation or error information.
        """

        try:
            prompt = self.prompt_manager.create_documentation_prompt(
                module_name=context.metadata.get("module_name", ""),
                file_path=context.metadata.get("file_path", ""),
                source_code=context.source_code,
                classes=context.classes,
                functions=context.functions
            )

            async with self.semaphore:
                response = await self._make_api_call_with_retry(prompt)

            parsed_response = await self.response_parser.parse_response(
                response,
                expected_format="docstring"
            )

            if parsed_response.errors:
                print_error(f"Error parsing AI response: {parsed_response.errors} with correlation ID: {self.correlation_id}")
                self.logger.error(f"Error parsing AI response: {parsed_response.errors}")
                return ProcessingResult(
                    content={"error": "Failed to parse AI response"},
                    usage={},
                    metrics={},
                    is_cached=False,
                    processing_time=0.0,
                    validation_status=False,
                    validation_errors=parsed_response.errors,
                    schema_errors=[]
                )

            # Further processing and validation of the parsed response
            docstring_data = self.docstring_processor.parse(parsed_response.content)
            is_valid, validation_errors = self.docstring_processor.validate(docstring_data)

            if not is_valid:
                print_warning(f"Docstring validation failed: {validation_errors} with correlation ID: {self.correlation_id}")
                self.logger.warning(f"Docstring validation failed: {validation_errors}")

                # Attempt to fix common docstring issues
                fixed_content = self.docstring_processor.fix_common_docstring_issues(parsed_response.content)
                docstring_data = self.docstring_processor.parse(fixed_content)
                is_valid, validation_errors = self.docstring_processor.validate(docstring_data)
                if is_valid:
                    parsed_response.content = fixed_content
                else:
                    print_error(f"Failed to fix docstring issues: {validation_errors} with correlation ID: {self.correlation_id}")
                    self.logger.error(f"Failed to fix docstring issues: {validation_errors}")
                    return ProcessingResult(
                        content={"error": "Failed to fix docstring issues"},
                        usage={},
                        metrics={},
                        is_cached=False,
                        processing_time=0.0,
                        validation_status=False,
                        validation_errors=validation_errors,
                        schema_errors=[]
                    )

            # Return the validated and processed docstring
            return ProcessingResult(
                content=docstring_data.to_dict(),
                usage=response.get("usage", {}),
                metrics={
                    "processing_time": parsed_response.parsing_time,
                    "response_size": len(str(response)),
                    "validation_success": is_valid
                },
                is_cached=False,
                processing_time=parsed_response.parsing_time,
                validation_status=is_valid,
                validation_errors=validation_errors,
                schema_errors=[]
            )

        except Exception as e:
            self.logger.error(f"Error generating documentation: {e}", exc_info=True)
            print_error(f"Error: {e} during generate_documentation in ai_service with correlation ID: {self.correlation_id}")
            return ProcessingResult(
                content={"error": str(e)},
                usage={},
                metrics={},
                is_cached=False,
                processing_time=0.0,
                validation_status=False,
                validation_errors=["An unexpected error occurred"],
                schema_errors=[]
            )

    async def _make_api_call_with_retry(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Makes an API call to the AI model with retry logic.

        Args:
            prompt: The prompt to send to the AI model.
            max_retries: Maximum number of retries for the API call.

        Returns:
            The raw response from the AI model.
        """
        headers = {
            "api-key": self.config.api_key,
            "Content-Type": "application/json"
        }

        if self.correlation_id:
            headers["x-correlation-id"] = self.correlation_id

        request_params = await self.token_manager.validate_and_prepare_request(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )

        # Add function calling parameters
        request_params["functions"] = [self.prompt_manager.get_function_schema()]
        request_params["function_call"] = {"name": "generate_docstring"}

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    endpoint = self.config.endpoint.rstrip('/') + '/'
                    path = f"openai/deployments/{self.config.deployment}/chat/completions"
                    url = urljoin(endpoint, path) + "?api-version=2024-02-15-preview"

                    async with session.post(
                        url,
                        headers=headers,
                        json=request_params,
                        timeout=self.config.timeout
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            self.logger.error(f"API call failed with status {response.status}: {error_text}")
                            if attempt == max_retries - 1:
                                raise Exception(f"API call failed after {max_retries} retries: {error_text}")
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                self.logger.error(f"Error during API call attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise

    async def close(self) -> None:
        """Closes the aiohttp client session."""
        print_info(f"Closing AI service with correlation ID: {self.correlation_id}")
        if self._client:
            await self._client.close()

    async def __aenter__(self) -> "AIService":
        """Enter method for async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit method for async context manager."""
        await self.close()
