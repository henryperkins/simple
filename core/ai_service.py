import asyncio
import json
import time
from typing import Any, Dict, Optional, TypeVar

from aiohttp.client import ClientSession
from aiohttp.client import ClientTimeout
from openai import AzureOpenAI

from api.token_management import TokenManager
from core.config import AIConfig
from core.docstring_processor import DocstringProcessor
from core.exceptions import (
    APICallError,
    ConfigurationError,
    DocumentationError,
    InvalidRequestError,
    ResponseParsingError,
)
from core.logger import CorrelationLoggerAdapter, LoggerSetup
from core.metrics_collector import MetricsCollector
from core.prompt_manager import PromptManager
from core.types.base import DocumentationContext, ProcessingResult
from utils import log_and_raise_error

T = TypeVar("T")  # For generic type hints


class AIService:
    """
    Manages interactions with the Azure OpenAI API.

    This service handles API calls, response formatting, and error handling
    for the Azure OpenAI API. It uses asynchronous operations, retry logic,
    and structured data handling to ensure reliable and efficient communication
    with the AI model.
    """

    def __init__(
        self,
        config: Optional[AIConfig] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the AI Service with Azure OpenAI configurations.

        :param config: Optional AIConfig object with Azure OpenAI details.
        :param correlation_id: Optional correlation ID for logging context.
        """
        # Delayed import to avoid circular dependency
        from core.dependency_injection import Injector

        self.config: AIConfig = config or Injector.get("config").ai
        self.correlation_id = correlation_id
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(
                f"{__name__}.{self.__class__.__name__}",
                correlation_id=self.correlation_id,
            ),
            extra={"correlation_id": self.correlation_id},
        )

        # Initialize the Azure OpenAI client
        try:
            self.client = AzureOpenAI(
                azure_endpoint=self.config.azure_api_base,
                api_key=self.config.api_key,
                api_version=self.config.azure_api_version,
                default_headers=(
                    {"x-correlation-id": self.correlation_id}
                    if self.correlation_id
                    else None
                ),
            )
            self.logger.info(
                "AI Service initialized",
                extra={
                    "model": self.config.model,
                    "deployment": self.config.azure_deployment_name,
                    "endpoint": self.config.azure_api_base,
                },
            )
        except Exception as e:
            log_and_raise_error(
                self.logger,
                e,
                ConfigurationError,
                "Failed to initialize AI Service",
                self.correlation_id,
            )

        self.prompt_manager = PromptManager(correlation_id=correlation_id)
        self.response_parser = Injector.get("response_parser")  # type: ignore
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)
        self.token_manager = TokenManager(
            model=self.config.model,
            deployment_name=self.config.azure_deployment_name,
            correlation_id=correlation_id,
        )

        # Initialize docstring processor
        try:
            self.docstring_processor = Injector.get("docstring_processor")
        except KeyError:
            self.logger.warning(
                "Docstring processor not registered, using default",
                extra={
                    "status": "warning",
                    "type": "fallback_processor",
                },
            )
            self.docstring_processor = DocstringProcessor()
            Injector.register("docstring_processor", self.docstring_processor)

        self.semaphore = asyncio.Semaphore(self.config.api_call_semaphore_limit)
        self._client: Optional[ClientSession] = None

    async def close(self) -> None:
        """Close the aiohttp client session."""
        if self._client:
            await self._client.close()
            self.logger.info("AI Service client session closed")

    async def start(self) -> None:
        """Start the aiohttp client session if not already started."""
        if self._client is None:
            self._client = ClientSession()
            self.logger.info("AI Service client session initialized")

    async def _make_api_call_with_retry(
        self,
        prompt: str,
        function_schema: Optional[Dict[str, Any]],
        max_retries: int = 3,
        log_extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an API call with retry logic following Azure best practices."""
        if not self.config or not self.config.api_key:
            raise ConfigurationError("API configuration is missing or invalid")

        headers = {
            "api-key": self.config.api_key,
            "Content-Type": "application/json",
        }
        if self.correlation_id:
            headers["x-correlation-id"] = self.correlation_id

        # Initialize request_params at the start
        request_params = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        # Validate Azure configuration
        if not self.config.azure_deployment_name or not self.config.azure_api_base:
            raise ConfigurationError(
                "Azure deployment name or API base URL is missing"
            )

        # Construct Azure OpenAI API URL
        url = (
            f"{self.config.azure_api_base.rstrip('/')}/openai/deployments/"
            f"{self.config.azure_deployment_name}/chat/completions"
            f"?api-version={self.config.azure_api_version}"
        )

        # Token validation and optimization
        try:
            prompt_tokens = self.token_manager.estimate_tokens(prompt)
            max_allowed_tokens = int(self.config.max_tokens * 0.75)

            is_valid, metrics, message = self.token_manager.validate_request(
                prompt, max_allowed_tokens
            )

            if not is_valid:
                optimized_prompt, usage = self.token_manager.optimize_prompt(
                    prompt,
                    max_tokens=max_allowed_tokens,
                    preserve_sections=[
                        "summary",
                        "description",
                        "parameters",
                        "returns",
                    ],
                )
                prompt = optimized_prompt
                prompt_tokens = usage.prompt_tokens

                # Update request parameters
                request_params["messages"][0]["content"] = prompt
                request_params["max_tokens"] = max(
                    100, self.config.max_tokens - prompt_tokens
                )
        except ValueError as e:
            raise InvalidRequestError("Failed to validate tokens") from e

        # Add function schema if provided
        if function_schema:
            request_params["functions"] = [function_schema]
            request_params["function_call"] = {
                "name": function_schema.get("name", "auto")
            }
        else:
            request_params["function_call"] = "none"

        # Initialize retry counter
        retry_count = 0
        last_error = None

        # Retry loop
        while retry_count <= max_retries:
            try:
                if not self._client:
                    await self.start()

                response_json = await self._make_single_api_call(
                    url, headers, request_params, retry_count, log_extra
                )
                if response_json:
                    return response_json

            except asyncio.TimeoutError as e:
                last_error = e
                self.logger.warning(
                    f"Request timeout (attempt {retry_count + 1}/{max_retries + 1})"
                )
            except Exception as e:
                last_error = e
                self.logger.error(
                    f"API call failed (attempt {retry_count + 1}/{max_retries + 1}): {str(e)}"
                )

            # Increment retry counter
            retry_count += 1

            if retry_count <= max_retries:
                # Calculate backoff time: 2^retry_count seconds
                backoff = 2 ** retry_count
                self.logger.info(f"Retrying in {backoff} seconds...")
                await asyncio.sleep(backoff)

        # If we get here, all retries failed
        error_msg = f"API call failed after {max_retries} retries"
        if last_error:
            error_msg += f": {str(last_error)}"
        raise APICallError(error_msg)

    async def _make_single_api_call(
        self,
        url: str,
        headers: Dict[str, str],
        request_params: Dict[str, Any],
        attempt: int,
        log_extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Makes a single API call and handles response."""
        self.logger.info(
            "Making API call",
            extra={
                "url": url,
                "attempt": attempt + 1,
                "correlation_id": self.correlation_id,
            },
        )

        # Validate credentials before making call
        if not self.config.api_key:
            raise APICallError(
                "Invalid API key. Please set AZURE_OPENAI_KEY in your environment variables."
            )

        if not self.config.azure_deployment_name:
            raise APICallError(
                "Invalid deployment name. Please set AZURE_DEPLOYMENT_NAME in your environment variables."
            )

        try:
            async with self._client.post(
                url,
                headers=headers,
                json=request_params,
                timeout=ClientTimeout(total=self.config.timeout),
            ) as response:
                if response.status == 200:
                    try:
                        response_json = await response.json()
                        self.logger.info(
                            "API call succeeded",
                            extra={
                                "status_code": response.status,
                                "correlation_id": self.correlation_id,
                            },
                        )
                        return response_json
                    except json.JSONDecodeError as e:
                        log_and_raise_error(
                            self.logger,
                            e,
                            ResponseParsingError,
                            f"Invalid JSON received from AI response (attempt {attempt + 1})",
                            self.correlation_id,
                        )
                        return None

                # Handle error responses
                error_text = await response.text()
                self.logger.error(
                    f"API call failed with status {response.status}: {error_text}",
                    extra={
                        "status_code": response.status,
                        "correlation_id": self.correlation_id,
                    },
                )

                if response.status == 401:
                    # Authentication error
                    raise APICallError(
                        "Authentication failed - invalid credentials. Please check your API key and endpoint."
                    )
                elif response.status == 403:
                    # Forbidden error
                    raise APICallError(
                        "Access forbidden. You may not have permission to access this resource."
                    )
                elif response.status == 429:
                    # Too many requests
                    raise APICallError(
                        "Rate limit exceeded. Please wait before making more requests."
                    )
                elif response.status >= 500:
                    # Server error
                    self.logger.warning(
                        f"Server error (status {response.status}). Retrying..."
                    )
                else:
                    # Other errors
                    raise APICallError(
                        f"API call failed with status {response.status}: {error_text}"
                    )

        except asyncio.CancelledError:
            self.logger.warning(
                "API call was cancelled",
                extra={"attempt": attempt + 1, "correlation_id": self.correlation_id},
            )
            return None

        except Exception as e:
            log_and_raise_error(
                self.logger,
                e,
                APICallError,
                "Unexpected error during API call",
                self.correlation_id,
            )

        return None

    async def generate_documentation(
        self, context: DocumentationContext, schema: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Generate documentation for the provided source code context."""
        start_time = time.time()
        log_extra = {
            "correlation_id": self.correlation_id,
            "module": context.metadata.get("module_name", ""),
            "file": context.metadata.get("file_path", ""),
        }

        try:
            # Create documentation prompt first
            prompt_result = await self.prompt_manager.create_documentation_prompt(
                context=context
            )

            if not prompt_result:
                raise DocumentationError("Failed to create documentation prompt")

            # Extract prompt content with proper validation
            if isinstance(prompt_result.content, dict):
                prompt = prompt_result.content.get("prompt", "")
            else:
                prompt = str(prompt_result.content)

            if not prompt:
                raise DocumentationError("Generated prompt is empty")

            # First try to optimize the prompt before validation
            optimized_prompt, usage = self.token_manager.optimize_prompt(
                prompt,
                max_tokens=int(self.config.max_tokens * 0.75),  # Use 75% of limit
                preserve_sections=["summary", "description", "parameters", "returns"],
            )
            prompt = optimized_prompt
            prompt_tokens = usage.prompt_tokens

            # Validate token count
            is_valid, metrics, message = self.token_manager.validate_request(
                prompt, max_completion_tokens=self.config.max_completion_tokens
            )

            if not is_valid:
                self.logger.warning(f"Token validation failed: {message}")
                # Try more aggressive optimization with lower token limit
                optimized_prompt, usage = self.token_manager.optimize_prompt(
                    prompt,
                    max_tokens=int(
                        self.config.max_tokens * 0.5
                    ),  # Use 50% of limit for more aggressive optimization
                    preserve_sections=[
                        "summary",
                        "description",
                    ],  # Keep only essential sections
                )
                prompt = optimized_prompt
                prompt_tokens = usage.prompt_tokens

                # Final validation
                is_valid, metrics, message = self.token_manager.validate_request(
                    prompt, self.config.max_completion_tokens
                )
                if not is_valid:
                    self.logger.warning(
                        "Unable to reduce token usage sufficiently; returning partial documentation."
                    )
                    # Instead of raising DocumentationError, return a minimal response
                    return ProcessingResult(
                        content="Partial documentation due to token limit.",
                        usage={},
                        metrics={},
                        validation_status=False,
                        validation_errors=[
                            "Token limit exceeded; partial documentation returned."
                        ],
                        schema_errors=[],
                    )

            self.logger.info(
                "Rendered prompt",
                extra={
                    **log_extra,
                    "prompt_length": len(prompt),
                    "prompt_tokens": prompt_tokens,
                },
            )

            # Add function calling instructions if schema is provided
            if schema:
                function_schema = self.prompt_manager.get_function_schema(schema)
            else:
                function_schema = None

            # Make API call with retry logic
            response = await self._make_api_call_with_retry(
                prompt,
                function_schema,
                max_retries=self.config.api_call_max_retries,
                log_extra=log_extra,
            )

            # Parse response, track metrics, etc.
            summary = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")[:100]
            )

            self.logger.info(f"AI Response Summary: {summary}", extra=log_extra)
            parsed_response = await self.response_parser.parse_response(
                response, expected_format="docstring", validate_schema=True
            )

            processing_time = time.time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=parsed_response.validation_success,
                duration=processing_time,
                metadata={
                    "module": context.metadata.get("module_name", ""),
                    "file": str(context.module_path),
                    "tokens": response.get("usage", {}),
                    "validation_success": parsed_response.validation_success,
                    "errors": parsed_response.errors
                    if not parsed_response.validation_success
                    else None,
                },
            )

            return ProcessingResult(
                content=parsed_response.content,
                usage=response.get("usage", {}),
                metrics={"processing_time": processing_time},
                validation_status=parsed_response.validation_success,
                validation_errors=parsed_response.errors or [],
                schema_errors=[],
            )

        except Exception as e:
            processing_time = time.time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=False,
                duration=processing_time,
                metadata={"error": str(e)},
            )
            log_and_raise_error(
                self.logger,
                e,
                DocumentationError,
                "Documentation generation failed",
                self.correlation_id,
            )
            raise