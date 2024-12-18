from typing import Any, Optional
import time
import asyncio
from urllib.parse import urljoin
import json
import aiohttp

from core.config import AIConfig
from core.docstring_processor import DocstringProcessor
from core.logger import LoggerSetup
from core.prompt_manager import PromptManager
from api.token_management import TokenManager
from core.exceptions import APICallError, DataValidationError, DocumentationError
from core.types.base import (
    ProcessingResult,
    DocumentationContext,
    ExtractedClass,
    ExtractedFunction,
)
from core.metrics_collector import MetricsCollector

class AIService:
    """Service for interacting with the AI model to generate documentation.

    This class manages the entire process of generating documentation from source code,
    including preparing prompts, making API calls, and handling responses. It is designed
    to be used asynchronously and supports retry logic for robust API interactions.

    Attributes:
        config (AIConfig): Configuration for the AI model and service parameters.
        correlation_id (str): An optional ID for correlating logs and operations.
    """

    def __init__(
        self, config: Optional[AIConfig] = None, correlation_id: Optional[str] = None
    ) -> None:
        """
        Initialize the AIService with optional configuration and correlation ID.

        Args:
            config (AIConfig, optional): Configuration for the AI model. If not provided,
                it will be retrieved from the dependency injector.
            correlation_id (str, optional): An ID used for correlating logs and operations
                across different parts of the system.
        """
        # Delayed import to avoid circular dependency
        from core.dependency_injection import Injector

        self.config = config or Injector.get("config")().ai
        self.correlation_id = correlation_id
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}", correlation_id=self.correlation_id
        )
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

        self.logger.info(
            "AI Service initialized",
            extra={
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "timeout": f"{self.config.timeout}s",
            },
        )

    async def start(self) -> None:
        """Start the AI service by initializing the client session.

        This method should be called before making any API calls to ensure the client
        session is properly set up.
        """
        if self._client is None:
            self._client = aiohttp.ClientSession()
            self.logger.info("AI Service client session initialized")

    def _add_source_code_to_response(self, response: dict[str, Any], source_code: str) -> dict[str, Any]:
        """Recursively adds source code to the response content and function call arguments.
        
        Args:
            response (dict[str, Any]): The API response to modify.
            source_code (str): The source code to add.
        
        Returns:
            dict[str, Any]: The modified response with source code added.
        """
        if isinstance(response, dict):
            response["source_code"] = source_code  # Add source code at the top level
            if "choices" in response:
                for choice in response["choices"]:
                    if "message" in choice:
                        message = choice["message"]
                        if "content" in message and message["content"] is not None:
                            try:
                                content = json.loads(message["content"])
                                if isinstance(content, dict):
                                    content["source_code"] = source_code
                                    content.setdefault("code_metadata", {})["source_code"] = source_code
                                message["content"] = json.dumps(content)
                            except (json.JSONDecodeError, AttributeError) as e:
                                self.logger.error(f"Error parsing response content: {e}")
                                message["content"] = json.dumps({
                                    "summary": "Error parsing content",
                                    "description": str(message["content"]),
                                    "source_code": source_code
                                })
                        if "function_call" in message:
                            try:
                                args = json.loads(message["function_call"].get("arguments", "{}"))
                                if isinstance(args, dict):
                                    args["source_code"] = source_code
                                message["function_call"]["arguments"] = json.dumps(args)
                            except json.JSONDecodeError as e:
                                self.logger.error(f"Error parsing function call arguments: {e}")
        return response
    
    def _format_response(self, response: dict[str, Any], log_extra: Optional[dict[str, str]] = None) -> dict[str, Any]:
        """Format the API response into a consistent structure.

        Args:
            response (dict[str, Any]): The raw API response to format.
            log_extra (dict[str, str], optional): Additional logging information.

        Returns:
            dict[str, Any]: The formatted response.
        """
        if "choices" in response and isinstance(response["choices"], list):
            return response

        if "summary" in response or "description" in response:
            return {
                "choices": [{"message": {"content": json.dumps(response)}}],
                "usage": response.get("usage", {}),
            }

        if "function_call" in response:
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

        if "tool_calls" in response:
            formatted_response = {
                "choices": [{"message": {"tool_calls": response["tool_calls"]}}],
                "usage": response.get("usage", {}),
            }
            self.logger.debug(f"Formatted tool calls response to: {formatted_response}", extra=log_extra)
            return formatted_response

        self.logger.warning("Response format is invalid, creating fallback.", extra={"response": response})
        return {
            "choices": [{"message": {"content": json.dumps({"summary": "Invalid response format", "description": "The response did not match the expected structure."})}}],
            "usage": {},
        }
        
        if "summary" in response or "description" in response:
            return {
                "choices": [{"message": {"content": json.dumps(response)}}],
                "usage": response.get("usage", {}),
            }
        
        if "function_call" in response:
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

        if "tool_calls" in response:
            formatted_response = {
                "choices": [{"message": {"tool_calls": response["tool_calls"]}}],
                "usage": response.get("usage", {}),
            }
            self.logger.debug(f"Formatted tool calls response to: {formatted_response}", extra=log_extra)
            return formatted_response
        
        formatted_response = {
            "choices": [{"message": {"content": json.dumps(response)}}],
            "usage": {},
        }
        
        # Ensure 'returns' field exists with a default if missing
        for choice in formatted_response.get("choices", []):
            if "message" in choice and "content" in choice["message"]:
                try:
                    content = json.loads(choice["message"]["content"])
                    if isinstance(content, dict) and "returns" not in content:
                        content["returns"] = {"type": "Any", "description": "No return description provided"}
                        choice["message"]["content"] = json.dumps(content)
                except (json.JSONDecodeError, TypeError) as e:
                    self.logger.error(f"Error adding default returns: {e}", extra=log_extra)
        
        self.logger.debug(f"Formatted generic response to: {formatted_response}", extra=log_extra)
        return formatted_response

    async def _make_api_call_with_retry(
        self, prompt: str, function_schema: Optional[dict[str, Any]], max_retries: int = 3, log_extra: Optional[dict[str, str]] = None
    ) -> dict[str, Any]:
        """Makes an API call to the AI model with retry logic.

        This method attempts to make an API call to the AI model, retrying up to a specified
        number of times in case of failures. It uses exponential backoff between retries.

        Args:
            prompt (str): The prompt to send to the AI model.
            function_schema (Optional[dict[str, Any]]): The schema for the function call.
            max_retries (int): The maximum number of retry attempts. Default is 3.
            log_extra (dict[str, str], optional): Additional logging information.

        Returns:
            dict[str, Any]: The response from the AI model.

        Raises:
            APICallError: If the API call fails after the maximum number of retries.
        """
        headers = {"api-key": self.config.api_key, "Content-Type": "application/json"}

        if self.correlation_id:
            headers["x-correlation-id"] = self.correlation_id

        request_params = await self.token_manager.validate_and_prepare_request(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        if request_params["max_tokens"] < 100:
            self.logger.warning(
                "Token availability is low. Consider reducing prompt size.", extra=log_extra
            )

        if function_schema:
            request_params["functions"] = [function_schema]
            request_params["function_call"] = "auto"  # Automatically select the function
            request_params["function_call"] = "auto"  # Automatically select the function

        self.logger.info(
            "Making API call with retry",
            extra={
                "prompt_tokens": request_params.get("max_tokens", 0),
                "temperature": request_params.get("temperature", 0),
                "retries": max_retries,
                **(log_extra or {}),
            },
        )

        for attempt in range(max_retries):
            try:
                endpoint = self.config.endpoint.rstrip("/") + "/"
                path = f"openai/deployments/{self.config.deployment}/chat/completions"
                url = urljoin(endpoint, path) + "?api-version=2024-10-01-preview"

                if self._client is None:
                    await self.start()

                if self._client is None:
                    raise APICallError("Failed to initialize client session")

                self.logger.info(f"Making API call (attempt {attempt + 1}/{max_retries}) to url: '{url}'", extra=log_extra)
                self.logger.debug(
                    f"Generated prompt: {request_params.get('prompt', 'No prompt found')}",
                    extra=log_extra,
                )
                async with self._client.post(
                    url,
                    headers=headers,
                    json=request_params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    self.logger.debug(
                        f"Raw response text: {await response.text()}",
                        extra={"status_code": response.status, "url": url},
                    )
                    url,
                    headers=headers,
                    json=request_params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    if response.status == 200:
                        try:
                            api_response = await response.json()
                        except aiohttp.ContentTypeError as e:
                            error_text = await response.text()
                            self.logger.error(
                                f"Invalid response content type or empty response: {error_text}",
                                extra={"status_code": response.status, "url": url},
                            )
                            raise APICallError(
                                f"Invalid response content type or empty response: {error_text}"
                            ) from e
                        self.logger.debug(f"Raw API response: {api_response}", extra=log_extra)
                        if not api_response or not isinstance(api_response, dict):
                            error_text = await response.text()
                            self.logger.error(
                                f"Empty or invalid JSON response: {error_text}",
                                extra={"status_code": response.status, "url": url},
                            )
                            return {
                                "choices": [{"message": {"content": "Empty or invalid JSON response"}}]
                            }  # Return a fallback response

                        # Validate response format
                        if not isinstance(api_response, dict) or "choices" not in api_response:
                            self.logger.error(f"Invalid response format: {api_response}", extra=log_extra)
                            return {
                                "choices": [{"message": {"content": "Invalid response format"}}]
                            }  # Return a fallback response

                        return api_response  # Return successful response immediately
                        self.logger.debug(f"Raw API response: {api_response}", extra=log_extra)
                        if not isinstance(api_response, dict):
                            self.logger.error(f"Invalid response format: {api_response}", extra=log_extra)
                            return {
                                "choices": [{"message": {"content": "Invalid response format"}}]
                            }  # Return a fallback response
                        if not isinstance(api_response, dict):
                            self.logger.error(f"Invalid response format: {api_response}", extra=log_extra)

                        # Validate response format
                        if not isinstance(api_response, dict) or "choices" not in api_response:
                            self.logger.error(f"Invalid response format: {api_response}", extra=log_extra)
                            return {
                                "choices": [{"message": {"content": "Invalid response format"}}]
                            }  # Return a fallback response

                        return api_response  # Return successful response immediately

                    error_text = await response.text()
                    if attempt == max_retries - 1:
                        raise APICallError(
                            f"API call failed after {max_retries} retries: {error_text} - url: {url}"
                        )
                    self.logger.warning(f"API call failed (attempt {attempt + 1}): {error_text} - url: {url}", extra=log_extra)
                    await asyncio.sleep(2**attempt)  # Exponential backoff

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == max_retries - 1:
                    raise APICallError(
                        f"API call failed after {max_retries} retries due to client error: {e} - url: {url}"
                    ) from e
                self.logger.warning(f"Client error (attempt {attempt + 1}): {e} - url: {url}", extra=log_extra)
                await asyncio.sleep(2**attempt)

        raise APICallError(f"API call failed after {max_retries} retries.")

    async def generate_documentation(
        self, context: DocumentationContext, schema: Optional[dict[str, Any]] = None
    ) -> ProcessingResult:
        """Generate documentation for the provided source code context."""
        start_time = time.time()
        module_name = (
            context.metadata.get("module_name", "") if context.metadata else ""
        )
        file_path = (
            context.metadata.get("file_path", "") if context.metadata else ""
        )
        log_extra = {"log_module": module_name, "file": file_path}
        try:
            self.logger.debug(
                f"Source code length: {len(context.source_code)}", extra=log_extra
            )
            self.logger.debug(
                f"First 50 characters of source code: {context.source_code[:50]}...",
                extra=log_extra,
            )

            if not context.source_code or not context.source_code.strip():
                self.logger.error(
                    "Source code is missing or empty",
                    extra={
                        "status": "error",
                        "type": "missing_source",
                        "log_module": module_name,
                        "file": file_path,
                    },
                )
                raise DocumentationError("Source code is missing or empty")

            self.logger.info(
                "Generating documentation",
                extra={
                    "log_module": module_name,
                    "file": file_path,
                    "code_length": len(context.source_code),
                    "classes": len(context.classes),
                    "functions": len(context.functions),
                },
            )

            # Convert classes and functions to proper types
            classes = [
                cls_data if isinstance(cls_data, ExtractedClass) else ExtractedClass(**cls_data)
                for cls_data in context.classes or []
            ]
            functions = [
                func_data if isinstance(func_data, ExtractedFunction) else ExtractedFunction(**func_data)
                for func_data in context.functions or []
            ]

            # Create documentation prompt
            self.logger.info("Generating documentation prompt.", extra=log_extra)
            self.logger.debug(
                f"Source code before creating prompt: {context.source_code[:50]}...",
                extra=log_extra
            )
            prompt = await self.prompt_manager.create_documentation_prompt(
                module_name=module_name,
                file_path=file_path,
                source_code=context.source_code,
                classes=classes,
                functions=functions,
            )

            # Add function calling instructions to the prompt if schema is provided
            if schema:
                prompt = self.prompt_manager.get_prompt_with_schema(prompt, schema)

            # Get the function schema
            function_schema = self.prompt_manager.get_function_schema(schema) if schema else None

            # Validate and prepare request
            request_params = await self.token_manager.validate_and_prepare_request(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            if request_params["max_tokens"] < 100:
                self.logger.warning(
                    "Token availability is low. Consider reducing prompt size.", extra=log_extra
                )

            self.logger.info("Making API call to generate documentation.", extra=log_extra)
            self.logger.debug(f"Generated prompt: {prompt}", extra=log_extra)
            async with self.semaphore:
                response = await self._make_api_call_with_retry(
                    str(prompt), function_schema, log_extra=log_extra
                )
            
            # Add source code to response
            response = self._add_source_code_to_response(response, context.source_code)
            
            # Format the response
            formatted_response = self._format_response(response, log_extra)

            # Parse response into DocstringData
            self.logger.info("Parsing and validating response", extra=log_extra)
            self.logger.debug(
                f"Source code before parsing response: {context.source_code[:50]}...",
                extra=log_extra
            )
            parsed_response = await self.response_parser.parse_response(
                formatted_response, expected_format="docstring", validate_schema=True
            )

            if not parsed_response.validation_success:
                self.logger.error(                    "Response validation failed",
                    extra={
                        "status": "error",
                        "type": "validation",
                        "errors": parsed_response.errors,
                        "log_module": module_name,
                        "file": file_path,
                    },
                )
                raise DataValidationError(
                    f"Response validation failed: {parsed_response.errors}"
                )

            # Track metrics
            processing_time = time.time() - start_time
            self.logger.info(
                f"Documentation generation completed in {processing_time:.2f} seconds.",
                extra=log_extra,
            )
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=True,
                duration=processing_time,
                metadata={
                    "log_module": module_name,
                    "file": file_path,
                    "code_length": len(context.source_code),
                    "classes": len(classes),
                    "functions": len(functions),
                },
                usage=formatted_response.get("usage", {}),
            )

            api_metrics: dict[str, Any] = {
                "Processing Time": f"{processing_time:.2f}s",
                "Prompt Tokens": formatted_response.get("usage", {}).get("prompt_tokens", 0),
                "Completion Tokens": formatted_response.get("usage", {}).get(
                    "completion_tokens", 0
                ),
                "Total Tokens": formatted_response.get("usage", {}).get("total_tokens", 0),
            }
            self.logger.info(f"API metrics: {api_metrics}", extra=log_extra)

            return ProcessingResult(
                content=parsed_response.content,
                usage=formatted_response.get("usage", {}),
                metrics={
                    "processing_time": processing_time,
                    "validation_success": True,
                },
                validation_status=True,
                validation_errors=[],
                schema_errors=[],
            )

        except DataValidationError as e:
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=False,
                duration=time.time() - start_time,
                metadata={"error_type": "validation_error", "error_message": str(e)},
            )
            self.logger.error(f"Data validation failed: {e}", exc_info=True, extra=log_extra)
            raise
        except DocumentationError as e:
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=False,
                duration=time.time() - start_time,
                metadata={"error_type": "documentation_error", "error_message": str(e)},
            )
            self.logger.error(f"Documentation error: {e}", exc_info=True, extra=log_extra)
            raise
        except Exception as e:
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=False,
                duration=time.time() - start_time,
                metadata={"error_type": "unexpected_error", "error_message": str(e)},
            )
            self.logger.error(f"Unexpected error: {e}", exc_info=True, extra=log_extra)
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
