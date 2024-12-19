"""AI service module for handling Azure OpenAI API interactions."""

import asyncio
import json
import time
from typing import Any, Dict, Optional, TypeVar

from aiohttp.client import ClientSession  # type: ignore
from aiohttp.client import ClientTimeout  # type: ignore
from openai import AzureOpenAI  # type: ignore

from api.token_management import TokenManager
from core.config import AIConfig
from core.docstring_processor import DocstringProcessor
from core.exceptions import APICallError, DocumentationError
from core.logger import LoggerSetup
from core.metrics_collector import MetricsCollector
from core.prompt_manager import PromptManager
from core.types.base import ProcessingResult, DocumentationContext

T = TypeVar('T')  # For generic type hints


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
        correlation_id: Optional[str] = None
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
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}",
            correlation_id=self.correlation_id,
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
            self.logger.error(f"Failed to initialize AI Service: {e}", exc_info=True)
            raise

        self.prompt_manager = PromptManager(correlation_id=correlation_id)
        self.response_parser = Injector.get("response_parser")  # type: ignore
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)

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

        self.token_manager = TokenManager(
            model=self.config.model,
            config=self.config,
            correlation_id=correlation_id,
        )
        self.semaphore = asyncio.Semaphore(10)  # Default concurrency limit
        self._client: Optional[ClientSession] = None

    async def start(self) -> None:
        """Start the aiohttp client session if not already started."""
        if self._client is None:
            self._client = ClientSession()
            self.logger.info("AI Service client session initialized")

    def _add_source_code_to_content(
        self, content: Dict[str, Any], source_code: str
    ) -> Dict[str, Any]:
        """Add source code metadata to content fields."""
        content["source_code"] = source_code
        content.setdefault("code_metadata", {})["source_code"] = source_code
        return content

    def _add_source_code_to_function_call(
        self, function_call: Dict[str, Any], source_code: str
    ) -> Dict[str, Any]:
        """Add source code to the function call arguments if possible."""
        if "arguments" in function_call:
            try:
                args = json.loads(function_call["arguments"])
                if isinstance(args, dict):
                    args["source_code"] = source_code
                    function_call["arguments"] = json.dumps(args)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing function call arguments: {e}")
        return function_call

    def _add_source_code_to_message(
        self, message: Dict[str, Any], source_code: str
    ) -> Dict[str, Any]:
        """Add source code to message content or function/tool calls."""
        if "content" in message and message["content"] is not None:
            try:
                content = json.loads(message["content"])
                if isinstance(content, dict):
                    content = self._add_source_code_to_content(content, source_code)
                    message["content"] = json.dumps(content)
            except (json.JSONDecodeError, AttributeError) as e:
                self.logger.error(f"Error parsing response content: {e}")
                fallback_content = {
                    "summary": "Error parsing content",
                    "description": str(message.get("content", "")),
                    "args": [],
                    "returns": {
                        "type": "Any",
                        "description": "No return value description available"
                    },
                    "raises": [],
                    "complexity": 1,
                    "source_code": source_code
                }
                message["content"] = json.dumps(fallback_content)

        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                if "function" in tool_call:
                    tool_call["function"] = self._add_source_code_to_function_call(
                        tool_call["function"], source_code
                    )
        elif "function_call" in message:
            message["function_call"] = self._add_source_code_to_function_call(
                message["function_call"], source_code
            )
        return message

    def _add_source_code_to_response(
        self, response: Dict[str, Any], source_code: str
    ) -> Dict[str, Any]:
        """Add source code metadata to the entire response structure."""
        response["source_code"] = source_code
        if "choices" in response:
            for choice in response["choices"]:
                if "message" in choice:
                    choice["message"] = self._add_source_code_to_message(
                        choice["message"], source_code
                    )
        return response

    def _format_fallback_response(
        self, response: Dict[str, Any], log_extra: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Format a fallback response for invalid structures."""
        self.logger.warning(
            "Response format is invalid, creating fallback.",
            extra={"response": response},
        )

        fallback_content = {
            "summary": "Invalid response format",
            "description": "The response did not match the expected structure.",
            "args": [],
            "returns": {
                "type": "Any",
                "description": "No return value description provided."
            },
            "raises": [],
            "complexity": 1
        }

        fallback_response = {
            "choices": [
                {"message": {"content": json.dumps(fallback_content)}}
            ],
            "usage": {}
        }

        self.logger.debug(
            f"Formatted fallback response: {fallback_response}", extra=log_extra
        )
        return fallback_response

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

        request_params = await self.token_manager.validate_and_prepare_request(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            truncation_strategy=str(self.config.truncation_strategy) if self.config.truncation_strategy else None,
            tool_choice=str(self.config.tool_choice) if self.config.tool_choice else None,
            parallel_tool_calls=self.config.parallel_tool_calls,
            response_format=str(self.config.response_format) if self.config.response_format else None,
            stream_options=self.config.stream_options,
        )

        if function_schema:
            request_params["tools"] = [
                {"type": "function", "function": function_schema}
            ]
            if not request_params.get("tool_choice"):
                request_params["tool_choice"] = "auto"

        # Construct the URL using AZURE_API_BASE and AZURE_DEPLOYMENT_NAME
        url = (
            f"{self.config.azure_api_base.rstrip('/')}/openai/deployments/{self.config.azure_deployment_name}"
            f"/chat/completions?api-version={self.config.azure_api_version}"
        )
        self.logger.debug(f"Constructed API URL: {url}")

        for attempt in range(max_retries):
            try:
                if self._client is None:
                    await self.start()

                if self._client:
                    self.logger.info(
                        "Making API call",
                        extra={"url": url, "attempt": attempt + 1, "correlation_id": self.correlation_id},
                    )

                    async with self._client.post(
                        url,
                        headers=headers,
                        json=request_params,
                        timeout=ClientTimeout(total=self.config.timeout),
                    ) as response:
                        if response.status == 200:
                            response_json = await response.json()
                            self.logger.info(
                                "API call succeeded",
                                extra={"status_code": response.status, "correlation_id": self.correlation_id},
                            )
                            # Check if the response is valid JSON
                            try:
                                json.dumps(response_json)
                                return response_json
                            except (TypeError, ValueError) as e:
                                self.logger.warning(
                                    f"Invalid JSON received from AI response (attempt {attempt + 1}), retrying: {e}",
                                    extra={
                                        **(log_extra or {}),
                                        "raw_response": response_json,
                                    }
                                )
                                await asyncio.sleep(2**attempt)
                                continue
                        error_text = await response.text()
                        self.logger.error(
                            "API call failed",
                            extra={
                                "status_code": response.status,
                                "error_text": error_text[:200],  # Limit error text length
                                "correlation_id": self.correlation_id,
                                "azure_api_base": self.config.azure_api_base,
                                "azure_deployment_name": self.config.azure_deployment_name,
                            },
                        )

                        # Handle specific error cases
                        if response.status == 429:  # Rate limit
                            retry_after = int(
                                response.headers.get("Retry-After", 2**(attempt + 2))
                            )
                            self.logger.warning(
                                f"Rate limit hit. Retrying after {retry_after} seconds.",
                                extra={"attempt": attempt + 1, "correlation_id": self.correlation_id},
                            )
                            await asyncio.sleep(retry_after)
                            continue
                        elif response.status == 503:  # Service unavailable
                            if "DeploymentNotFound" in error_text:
                                self.logger.critical(
                                    "Azure OpenAI deployment not found. Please verify the following:\n"
                                    "1. The deployment name in the configuration matches an existing deployment in your Azure OpenAI resource.\n"
                                    "2. The deployment is active and fully provisioned.\n"
                                    "3. The API key and endpoint are correct.\n"
                                    "4. If the deployment was recently created, wait a few minutes and try again.",
                                    extra={
                                        "azure_api_base": self.config.azure_api_base,
                                        "azure_deployment_name": self.config.azure_deployment_name,
                                        "correlation_id": self.correlation_id,
                                    },
                                )
                                retry_after = 10  # Wait longer for deployment issues
                                await asyncio.sleep(retry_after)
                                continue
                            self.logger.warning(
                                f"Service unavailable. Retrying after {2**attempt} seconds.",
                                extra={"attempt": attempt + 1, "correlation_id": self.correlation_id},
                            )
                            await asyncio.sleep(2**attempt)
                            continue
                        else:
                            raise APICallError(f"API call failed: {error_text}")

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Request timeout (attempt {attempt + 1})", extra=log_extra
                )
                await asyncio.sleep(2**attempt)
            except Exception as e:
                self.logger.error(
                    f"Error during API call: {e}", exc_info=True, extra=log_extra
                )
                if attempt == max_retries - 1:
                    raise APICallError(
                        f"API call failed after {max_retries} retries: {e}"
                    )
                await asyncio.sleep(2**attempt)

    async def generate_documentation(
        self, context: DocumentationContext, schema: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Generate documentation for the provided source code context.

        :param context: A DocumentationContext object containing source code and metadata.
        :param schema: Optional function schema to influence the AI's response format.
        :return: A ProcessingResult with parsed and validated documentation content.
        """
        start_time = time.time()
        log_extra = {
            "correlation_id": self.correlation_id,
            "module": context.metadata.get("module_name", ""),
            "file": context.metadata.get("file_path", ""),
        }

        try:
            # Validate input
            if not context.source_code or not context.source_code.strip():
                raise DocumentationError("Source code is missing or empty")

            # Create documentation prompt
            prompt_result = await self.prompt_manager.create_documentation_prompt(context=context)

            # Add function calling instructions if schema is provided
            if schema:
                base_prompt = prompt_result.content["prompt"] if isinstance(prompt_result.content, dict) else str(prompt_result.content)
                prompt = self.prompt_manager.get_prompt_with_schema(base_prompt, schema)
                function_schema = self.prompt_manager.get_function_schema(schema)
            else:
                prompt = prompt_result.content["prompt"] if isinstance(prompt_result.content, dict) else str(prompt_result.content)
                function_schema = None

            # Make API call with retry logic
            response = await self._make_api_call_with_retry(
                prompt,
                function_schema,
                max_retries=self.config.api_call_max_retries,
                log_extra=log_extra,
            )

            # Log the raw response before validation
            self.logger.debug(f"Raw AI response: {response}", extra=log_extra)

            # Parse and validate response
            parsed_response = await self.response_parser.parse_response(
                response, expected_format="docstring", validate_schema=True
            )

            # Track metrics based on validation success
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
                    "errors": parsed_response.errors if not parsed_response.validation_success else None
                },
            )

            # Return ProcessingResult with validation status and any errors
            return ProcessingResult(
                content=parsed_response.content,  # Use the content even if validation failed
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
            self.logger.error(
                f"Documentation generation failed: {e}", exc_info=True, extra=log_extra
            )
            raise

    async def close(self) -> None:
        """Close the aiohttp client session."""
        if self._client:
            await self._client.close()
            self._client = None
            self.logger.info("AI Service client session closed")

    async def __aenter__(self) -> "AIService":
        """Enter the async context manager by starting the client session."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Exit the async context manager by closing the client session."""
        await self.close()
