import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar

from aiohttp.client import ClientSession  # type: ignore
from aiohttp.client import ClientTimeout  # type: ignore
from aiohttp import ClientError
from openai import AzureOpenAI

from api.token_management import TokenManager
from core.config import AIConfig
from core.docstring_processor import DocstringProcessor
from core.exceptions import (
    APICallError,
    DocumentationError,
    InvalidRequestError,
    ResponseParsingError,
    ConfigurationError
)
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.metrics_collector import MetricsCollector
from core.prompt_manager import PromptManager
from core.types.base import ProcessingResult, DocumentationContext
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
        self, config: Optional[AIConfig] = None, correlation_id: Optional[str] = None
    ) -> None:
        """
        Initialize the AI Service with Azure OpenAI configurations.

        :param config: Optional AIConfig object with Azure OpenAI details.
        :param correlation_id: Optional correlation ID for logging context.
        """
        # Delayed import to avoid circular dependency
        from core.dependency_injection import Injector

        self.config: AIConfig = config or Injector.get("config")().ai
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
                azure_endpoint=self.config.endpoint,
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
                    "deployment": self.config.deployment,
                    "endpoint": self.config.endpoint,
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

        self.semaphore = asyncio.Semaphore(10)  # Default concurrency limit
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
        function_schema: Optional[dict[str, Any]],
        max_retries: int = 3,
        log_extra: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Make an API call with retry logic following Azure best practices."""
        headers = {"api-key": self.config.api_key, "Content-Type": "application/json"}
        if self.correlation_id:
            headers["x-correlation-id"] = self.correlation_id

        try:
            # Estimate and validate tokens with aggressive optimization
            prompt_tokens = self.token_manager.estimate_tokens(prompt)
            max_allowed_tokens = int(self.config.max_tokens * 0.75)  # Use 75% of limit for safety
            
            is_valid, metrics, message = self.token_manager.validate_request(
                prompt, max_allowed_tokens
            )
            
            if not is_valid:
                self.logger.warning(
                    f"Initial request validation failed: {message}. Attempting optimization..."
                )
                # Try aggressive optimization
                optimized_prompt, usage = self.token_manager.optimize_prompt(
                    prompt,
                    max_tokens=max_allowed_tokens,
                    preserve_sections=["summary", "description", "parameters", "returns"]
                )
                prompt = optimized_prompt
                prompt_tokens = usage.prompt_tokens
                
                # Validate again after optimization
                is_valid, metrics, message = self.token_manager.validate_request(
                    prompt, max_allowed_tokens
                )
                if not is_valid:
                    self.logger.warning("Still above token limit after optimization. Force truncating.")
                    prompt = prompt[: int(len(prompt) * 0.6)]
                    prompt_tokens = self.token_manager.estimate_tokens(prompt)
                    # Return truncated prompt instead of raising error
                    self.logger.warning(f"Using truncated prompt (tokens={prompt_tokens}).")
                    request_params["messages"][0]["content"] = prompt

            # Prepare request parameters with optimized prompt
            request_params = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.config.max_tokens - prompt_tokens,
                "temperature": self.config.temperature,
            }

        except ValueError as e:
            log_and_raise_error(
                self.logger,
                e,
                InvalidRequestError,
                "Invalid request parameters",
                self.correlation_id,
            )

        if function_schema:
            request_params["tools"] = [
                {"type": "function", "function": function_schema}
            ]
            if not request_params.get("tool_choice"):
                request_params["tool_choice"] = "auto"

        # Construct the URL using AZURE_API_BASE and AZURE_DEPLOYMENT_NAME
        url = (
            f"{self.config.azure_api_base.rstrip('/')}/openai/deployments/"
            f"{self.config.azure_deployment_name}/chat/completions"
            f"?api-version={self.config.azure_api_version}"
        )
        self.logger.debug(f"Constructed API URL: {url}")

        for attempt in range(max_retries):
            try:
                if self._client is None:
                    await self.start()

                if self._client:
                    response_json = await self._make_single_api_call(
                        url, headers, request_params, attempt, log_extra
                    )
                    if response_json:
                        # Track token usage
                        prompt_tokens = response_json.get('usage', {}).get('prompt_tokens', 0)
                        completion_tokens = response_json.get('usage', {}).get('completion_tokens', 0)
                        self.token_manager.track_request(prompt_tokens, completion_tokens)
                        self.logger.debug(f"Tracked token usage: prompt={prompt_tokens}, completion={completion_tokens}")

                        # Save raw response to a debug file
                        debug_file = (
                            Path("logs") / f"api_response_{self.correlation_id}.json"
                        )
                        debug_file.write_text(
                            json.dumps(response_json, indent=2), encoding="utf-8"
                        )
                        self.logger.debug(f"Raw API response saved to {debug_file}")
                        return response_json

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Request timeout (attempt {attempt + 1})", extra=log_extra
                )
                await asyncio.sleep(2**attempt)
            except APICallError as e:
                if attempt == max_retries - 1:
                    log_and_raise_error(
                        self.logger,
                        e,
                        APICallError,
                        f"API call failed after {max_retries} retries",
                        self.correlation_id,
                    )
                await asyncio.sleep(2**attempt)
            except Exception as e:
                log_and_raise_error(
                    self.logger,
                    e,
                    APICallError,
                    "Unexpected error during API call",
                    self.correlation_id,
                )
                await asyncio.sleep(2**attempt)

        raise APICallError(f"API call failed after {max_retries} retries")

    async def _make_single_api_call(
        self,
        url: str,
        headers: dict[str, str],
        request_params: dict[str, Any],
        attempt: int,
        log_extra: Optional[dict[str, str]] = None,
    ) -> Optional[dict[str, Any]]:
        """Makes a single API call and handles response."""
        self.logger.info(
            "Making API call",
            extra={
                "url": url,
                "attempt": attempt + 1,
                "correlation_id": self.correlation_id,
            },
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
                        # Save raw response to a debug file
                        debug_file = (
                            Path("logs") / f"api_response_{self.correlation_id}.json"
                        )
                        debug_file.write_text(
                            json.dumps(response_json, indent=2), encoding="utf-8"
                        )
                        self.logger.debug(f"Raw API response saved to {debug_file}")
                        return response_json
                    except json.JSONDecodeError as e:
                        log_and_raise_error(
                            self.logger,
                            e,
                            ResponseParsingError,
                            f"Invalid JSON received from AI response (attempt {attempt + 1}), retrying",
                            self.correlation_id,
                            raw_response=await response.text(),
                        )
                        return None

                error_text = await response.text()
                log_extra = {
                    "status_code": response.status,
                    "error_text": error_text[:200],
                    "correlation_id": self.correlation_id,
                    "azure_api_base": self.config.azure_api_base,
                    "azure_deployment_name": self.config.azure_deployment_name,
                }
                self.logger.error(
                    "API call failed",
                    extra=log_extra,
                )

                if response.status == 429:  # Rate limit
                    retry_after = int(
                        response.headers.get("Retry-After", 2 ** (attempt + 2))
                    )
                    self.logger.warning(
                        f"Rate limit hit. Retrying after {retry_after} seconds.",
                        extra={
                            "attempt": attempt + 1,
                            "correlation_id": self.correlation_id,
                        },
                    )
                    await asyncio.sleep(retry_after)
                    return None
                elif response.status == 503:  # Service unavailable
                    if "DeploymentNotFound" in error_text:
                        self.logger.critical(
                            "Azure OpenAI deployment not found. Please verify the following:\n"
                            "1. The deployment name in the configuration matches an existing deployment in your Azure OpenAI resource.\n"
                            "2. The deployment is active and fully provisioned.\n"
                            "3. The API key and endpoint are correct.\n"
                            "4. If the deployment was recently created, wait a few minutes and try again.\n"
"5. Ensure the deployment name is set correctly in the AZURE_DEPLOYMENT_NAME environment variable.",
                            extra={
                                "azure_api_base": self.config.azure_api_base,
                                "azure_deployment_name": self.config.azure_deployment_name,
                                "correlation_id": self.correlation_id,
                            },
                        )
                        await asyncio.sleep(10)  # Wait longer for deployment issues
                        return None
                    self.logger.warning(
                        f"Service unavailable. Retrying after {2**attempt} seconds.",
                        extra={
                            "attempt": attempt + 1,
                            "correlation_id": self.correlation_id,
                        },
                    )
                    await asyncio.sleep(2**attempt)
                    return None
                else:
                    log_and_raise_error(
                        self.logger,
                        Exception(
                            f"API call failed with status code: {response.status}"
                        ),
                        APICallError,
                        "API call failed",
                        self.correlation_id,
                        status_code=response.status,
                        error_text=error_text,
                        azure_api_base=self.config.azure_api_base,
                        azure_deployment_name=self.config.azure_deployment_name,
                    )
        except ClientError as e:
            log_and_raise_error(
                self.logger,
                e,
                APICallError,
                "Client error during API call",
                self.correlation_id,
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
                preserve_sections=["summary", "description", "parameters", "returns"]
            )
            prompt = optimized_prompt
            prompt_tokens = usage.prompt_tokens

            # Validate token count
            is_valid, metrics, message = self.token_manager.validate_request(
                prompt, 
                max_completion_tokens=self.config.max_completion_tokens
            )
            
            if not is_valid:
                self.logger.warning(f"Token validation failed: {message}")
                # Try more aggressive optimization with lower token limit
                optimized_prompt, usage = self.token_manager.optimize_prompt(
                    prompt,
                    max_tokens=int(self.config.max_tokens * 0.5),  # Use 50% of limit for more aggressive optimization
                    preserve_sections=["summary", "description"]  # Keep only essential sections
                )
                prompt = optimized_prompt
                prompt_tokens = usage.prompt_tokens
                
                # Final validation
                is_valid, metrics, message = self.token_manager.validate_request(
                    prompt, self.config.max_completion_tokens
                )
                if not is_valid:
                    self.logger.warning("Unable to reduce token usage sufficiently; returning partial documentation.")
                    # Instead of raising DocumentationError, return a minimal response
                    return ProcessingResult(
                        content="Partial documentation due to token limit.",
                        usage={},
                        metrics={},
                        validation_status=False,
                        validation_errors=["Token limit exceeded; partial documentation returned."],
                        schema_errors=[],
                    )

            self.logger.info(
                "Rendered prompt", 
                extra={**log_extra, "prompt_length": len(prompt), "prompt_tokens": prompt_tokens}
            )

            # Add function calling instructions if schema is provided
            if schema:
                prompt = self.prompt_manager.get_prompt_with_schema(prompt, schema)
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
                    "errors": parsed_response.errors if not parsed_response.validation_success else None,
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
