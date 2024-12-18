"""AI service module for handling Azure OpenAI API interactions."""

from typing import Any, Optional
import asyncio
import json
import time

import aiohttp

from api.token_management import TokenManager
from core.config import AIConfig
from core.docstring_processor import DocstringProcessor
from core.exceptions import APICallError, DataValidationError, DocumentationError
from core.logger import LoggerSetup
from core.metrics_collector import MetricsCollector
from core.prompt_manager import PromptManager
from core.types.base import ProcessingResult, DocumentationContext
from openai import AzureOpenAI


class AIService:
    """
    Manages interactions with the Azure OpenAI API.
    
    This service handles API calls, response formatting, and error handling
    for the Azure OpenAI API. It uses a combination of asynchronous operations,
    retry logic, and structured data handling to ensure reliable and efficient
    communication with the AI model.
    """

    def __init__(
        self, config: Optional[AIConfig] = None, correlation_id: Optional[str] = None
    ) -> None:
        """Initialize the AI Service with Azure OpenAI configurations."""
        # Delayed import to avoid circular dependency
        from core.dependency_injection import Injector

        self.config = config or Injector.get("config")().ai
        self.correlation_id = correlation_id
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}",
            correlation_id=self.correlation_id,
        )

        # Initialize the Azure OpenAI client following best practices
        try:
            self.client = AzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                api_version=self.config.azure_api_version,
                default_headers={
                    "x-correlation-id": self.correlation_id
                } if self.correlation_id else None
            )
            self.logger.info(
                "AI Service initialized",
                extra={
                    "model": self.config.model,
                    "deployment": self.config.deployment
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize AI Service: {e}", exc_info=True)
            raise

        self.prompt_manager = PromptManager(correlation_id=correlation_id)
        self.response_parser = Injector.get("response_parser")
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)

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
        self.semaphore = asyncio.Semaphore(10)  # Default semaphore value
        self._client: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        if self._client is None:
            self._client = aiohttp.ClientSession()
            self.logger.info("AI Service client session initialized")

    def _add_source_code_to_content(self, content: dict[str, Any], source_code: str) -> dict[str, Any]:
        """Add source code to the content part of the response."""
        if isinstance(content, dict):
            content["source_code"] = source_code
            content.setdefault("code_metadata", {})["source_code"] = source_code
        return content

    def _add_source_code_to_function_call(self, function_call: dict[str, Any], source_code: str) -> dict[str, Any]:
        """Add source code to the function call arguments."""
        if isinstance(function_call, dict) and "arguments" in function_call:
            try:
                args = json.loads(function_call["arguments"])
                if isinstance(args, dict):
                    args["source_code"] = source_code
                    function_call["arguments"] = json.dumps(args)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing function call arguments: {e}")
        return function_call

    def _add_source_code_to_message(self, message: dict[str, Any], source_code: str) -> dict[str, Any]:
        """Add source code to the message content or function call."""
        if "content" in message and message["content"] is not None:
            try:
                content = json.loads(message["content"])
                content = self._add_source_code_to_content(content, source_code)
                message["content"] = json.dumps(content)
            except (json.JSONDecodeError, AttributeError) as e:
                self.logger.error(
                    f"Error parsing response content: {e}",
                    extra={
                        "correlation_id": self.correlation_id,
                        "response_content": message.get("content", ""),
                    },
                )
                message["content"] = json.dumps({
                    "summary": "Error parsing content",
                    "description": str(message["content"]),
                    "source_code": source_code
                })
        if "function_call" in message:
            message["function_call"] = self._add_source_code_to_function_call(message["function_call"], source_code)
        return message

    def _add_source_code_to_response(self, response: dict[str, Any], source_code: str) -> dict[str, Any]:
        """Add source code to the response."""
        if isinstance(response, dict):
            response["source_code"] = source_code
            if "choices" in response:
                for choice in response["choices"]:
                    if "message" in choice:
                        choice["message"] = self._add_source_code_to_message(choice["message"], source_code)
        return response

    def _format_summary_description_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Format response with summary or description."""
        return {
            "choices": [{"message": {"content": json.dumps(response)}}],
            "usage": response.get("usage", {}),
        }

    def _format_function_call_response(self, response: dict[str, Any], log_extra: Optional[dict[str, str]] = None) -> dict[str, Any]:
        """Format response with function call."""
        formatted_response = {
            "choices": [{"message": {"function_call": response["function_call"]}}],
            "usage": response.get("usage", {}),
        }
        # Validate the function call arguments
        args = json.loads(response["function_call"].get("arguments", "{}"))
        is_valid, errors = self.response_parser._validate_content(args, "function")
        if not is_valid:
            self.logger.error(f"Function call arguments validation failed: {errors}")
            raise DataValidationError(f"Invalid function call arguments: {errors}")
        self.logger.debug(f"Formatted function call response to: {formatted_response}", extra=log_extra)
        return formatted_response

    def _format_tool_calls_response(self, response: dict[str, Any], log_extra: Optional[dict[str, str]] = None) -> dict[str, Any]:
        """Format response with tool calls."""
        formatted_response = {
            "choices": [{"message": {"tool_calls": response["tool_calls"]}}],
            "usage": response.get("usage", {}),
        }
        self.logger.debug(f"Formatted tool calls response to: {formatted_response}", extra=log_extra)
        return formatted_response

    def _format_fallback_response(self, response: dict[str, Any], log_extra: Optional[dict[str, str]] = None) -> dict[str, Any]:
        """Format a fallback response when the response format is invalid."""
        self.logger.warning("Response format is invalid, creating fallback.", extra={"response": response})
        fallback_response = {
            "choices": [{"message": {"content": json.dumps({"summary": "Invalid response format", "description": "The response did not match the expected structure."})}}],
            "usage": {},
        }

        # Ensure 'returns' field exists with a default if missing
        for choice in fallback_response.get("choices", []):
            if "message" in choice and "content" in choice["message"]:
                try:
                    content = json.loads(choice["message"]["content"])
                    if isinstance(content, dict) and "returns" not in content:
                        content["returns"] = {"type": "Any", "description": "No return description provided"}
                        choice["message"]["content"] = json.dumps(content)
                except (json.JSONDecodeError, TypeError) as e:
                    self.logger.error(f"Error formatting fallback response: {e}")

        self.logger.debug(f"Formatted generic response to: {fallback_response}", extra=log_extra)
        return fallback_response

    def _format_response(self, response: dict[str, Any], log_extra: Optional[dict[str, str]] = None) -> dict[str, Any]:
        """Format the response for further processing."""
        try:
            # Case 1: Already in choices format
            if "choices" in response and isinstance(response["choices"], list):
                return response

            # Case 2: Direct summary/description format
            if "summary" in response or "description" in response:
                formatted = {
                    "choices": [{
                        "message": {
                            "content": json.dumps(response)
                        }
                    }],
                    "usage": response.get("usage", {})
                }
                self.logger.debug(f"Formatted direct response to: {formatted}", extra=log_extra)
                return formatted

            # Case 3: Function call format  
            if "function_call" in response:
                formatted = {
                    "choices": [{
                        "message": {
                            "function_call": response["function_call"]
                        }
                    }],
                    "usage": response.get("usage", {})
                }
                self.logger.debug(f"Formatted function call to: {formatted}", extra=log_extra)
                return formatted

            # Case 4: Fallback format
            self.logger.warning("Invalid response format, creating fallback", extra=log_extra)
            return {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "summary": "Invalid response format",
                            "description": str(response)
                        })
                    }
                }],
                "usage": {}
            }
        except Exception as e:
            self.logger.error(f"Error formatting response: {e}", exc_info=True)
            return self._format_fallback_response(response, log_extra)

    async def _make_api_call_with_retry(
        self,
        prompt: str,
        function_schema: Optional[dict[str, Any]],
        max_retries: int = 3,
        log_extra: Optional[dict[str, str]] = None
    ) -> dict[str, Any]:
        """Make an API call with retry logic following Azure best practices."""
        headers = {
            "api-key": self.config.api_key,
            "Content-Type": "application/json"
        }
        if self.correlation_id:
            headers["x-correlation-id"] = self.correlation_id

        request_params = await self.token_manager.validate_and_prepare_request(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        if function_schema:
            request_params["tools"] = [{"type": "function", "function": function_schema}]
            request_params["tool_choice"] = "auto"

        # Construct the URL using AZURE_API_BASE and AZURE_DEPLOYMENT_NAME
        url = (
            f"{self.config.azure_api_base.rstrip('/')}/openai/deployments/{self.config.azure_deployment_name}"
            f"/chat/completions?api-version={self.config.azure_api_version}"
        )

        # Log the input sent to the AI
        with open("ai_input.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"{json.dumps(request_params, indent=2)}\n")

        for attempt in range(max_retries):
            try:
                if self._client is None:
                    await self.start()

                if self._client:
                    async with self._client.post(
                        url,
                        headers=headers,
                        json=request_params,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                    ) as response:
                        if response.status == 200:
                            raw_response = await response.text()
                            with open("raw_ai_responses.log", "a", encoding="utf-8") as log_file:
                                log_file.write(f"{raw_response}\n")
                            return json.loads(raw_response)

                        error_text = await response.text()
                        self.logger.warning(
                            f"API call failed (attempt {attempt + 1}): {error_text}",
                            extra=log_extra
                        )

                        # Handle specific error cases
                        if response.status == 429:  # Rate limit
                            retry_after = response.headers.get('Retry-After')
                            if retry_after:
                                self.logger.warning(
                                    f"Rate limit hit. Retrying after {retry_after} seconds.",
                                    extra={"attempt": attempt + 1, "retry_after": retry_after},
                                )
                                await asyncio.sleep(int(retry_after))
                            else:
                                self.logger.warning(
                                    f"Rate limit hit. Retrying with exponential backoff.",
                                    extra={"attempt": attempt + 1},
                                )
                                await asyncio.sleep(2 ** attempt)
                            continue
                        elif response.status == 503:  # Service unavailable
                            await asyncio.sleep(2 ** attempt)
                            continue
                        else:
                            raise APICallError(f"API call failed: {error_text}")

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Request timeout (attempt {attempt + 1})",
                    extra=log_extra
                )
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                self.logger.error(
                    f"Error during API call: {e}",
                    exc_info=True,
                    extra={
                        **(log_extra or {}),
                        "attempt": attempt + 1,
                        "url": url,
                        "headers": headers,
                        "request_params": request_params,
                    },
                )
                if attempt == max_retries - 1:
                    raise APICallError(f"API call failed after {max_retries} retries: {e}")
                await asyncio.sleep(2 ** attempt)

        raise APICallError(f"API call failed after {max_retries} retries")

    async def generate_documentation(
        self, context: DocumentationContext, schema: Optional[dict[str, Any]] = None
    ) -> ProcessingResult:
        """Generate documentation for the provided source code context."""
        start_time = time.time()
        log_extra = {
            "correlation_id": self.correlation_id,
            "module": context.metadata.get("module_name", ""),
            "file": context.metadata.get("file_path", "")
        }

        try:
            # Validate input
            if not context.source_code or not context.source_code.strip():
                raise DocumentationError("Source code is missing or empty")

            # Create documentation prompt
            prompt_result = await self.prompt_manager.create_documentation_prompt(
                context=context
            )

            # Add function calling instructions if schema provided
            if schema:
                prompt = self.prompt_manager.get_prompt_with_schema(
                    prompt_result.content["prompt"] if isinstance(prompt_result.content, dict) else str(prompt_result.content),
                    schema
                )
                function_schema = self.prompt_manager.get_function_schema(schema)
            else:
                prompt = prompt_result.content["prompt"] if isinstance(prompt_result.content, dict) else str(prompt_result.content)
                function_schema = None

            # Make API call with proper error handling and retries
            try:
                response = await self._make_api_call_with_retry(
                    prompt, 
                    function_schema,
                    max_retries=self.config.api_call_max_retries,
                    log_extra=log_extra
                )

                # Parse and validate response
                parsed_response = await self.response_parser.parse_response(
                    response, 
                    expected_format="docstring",
                    validate_schema=True
                )

                if not parsed_response.validation_success:
                    raise DataValidationError(
                        f"Response validation failed: {parsed_response.errors}"
                    )

                # Track metrics
                processing_time = time.time() - start_time
                await self.metrics_collector.track_operation(
                    operation_type="documentation_generation",
                    success=True,
                    duration=processing_time,
                    metadata={
                        "module": context.metadata.get("module_name", ""),
                        "file": str(context.module_path),
                        "tokens": response.get("usage", {}),
                    },
                )

                return ProcessingResult(
                    content=parsed_response.content,
                    usage=response.get("usage", {}),
                    metrics={"processing_time": processing_time},
                    validation_status=True,
                    validation_errors=[],
                    schema_errors=[],
                )

            except Exception as api_error:
                self.logger.error(
                    f"API call failed: {api_error}",
                    exc_info=True,
                    extra=log_extra
                )
                raise APICallError(f"Failed to generate documentation: {api_error}")

        except Exception as e:
            processing_time = time.time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=False,
                duration=processing_time,
                metadata={"error": str(e)},
            )
            self.logger.error(
                f"Documentation generation failed: {e}",
                exc_info=True,
                extra=log_extra
            )
            raise

    async def close(self) -> None:
        """Closes the aiohttp client session."""
        if self._client:
            await self._client.close()
            self._client = None
            self.logger.info("AI Service client session closed")

    async def __aenter__(self) -> "AIService":
        """Enter method for async context manager."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Exit method for async context manager."""
        await self.close()
