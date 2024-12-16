"""
Service for interacting with the AI model to generate documentation.
"""
from typing import Any
import time
import asyncio
from urllib.parse import urljoin
import json
import aiohttp

from core.config import AIConfig
from core.docstring_processor import DocstringProcessor
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.prompt_manager import PromptManager
from core.dependency_injection import Injector
from api.token_management import TokenManager
from core.exceptions import APICallError, DataValidationError, DocumentationError
from core.types.base import (
    ProcessingResult,
    DocumentationContext,
    ExtractedClass,
    ExtractedFunction,
)
from core.types.docstring import DocstringData
from core.metrics_collector import MetricsCollector
from core.console import (
    print_info,
    print_phase_header,
    display_processing_phase,
    display_metrics,
    display_api_metrics,
    create_status_table,
)


class AIService:
    """Service for interacting with the AI model to generate documentation."""

    def __init__(
        self, config: AIConfig | None = None, correlation_id: str | None = None
    ) -> None:
        """
        Initialize AI service.

        Args:
            config: AI service configuration.
            correlation_id: Optional correlation ID for tracking related operations.
        """
        self.config = config or Injector.get("config")().ai
        self.correlation_id = correlation_id
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__),
            extra={"correlation_id": self.correlation_id}
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
                    "correlation_id": self.correlation_id,
                    "sanitized_info": {"status": "warning", "type": "fallback_processor"}
                }
            )
            self.docstring_processor = DocstringProcessor()
            Injector.register("docstring_processor", self.docstring_processor)

        self.token_manager = TokenManager(
            model=self.config.model,
            config=self.config,
            correlation_id=correlation_id,
        )
        self.semaphore = asyncio.Semaphore(10)  # Default semaphore value
        self._client: aiohttp.ClientSession | None = None

        print_phase_header("AI Service Initialization")
        create_status_table("Configuration", {
            "Model": self.config.model,
            "Max Tokens": self.config.max_tokens,
            "Temperature": self.config.temperature,
            "Timeout": f"{self.config.timeout}s"
        })

    async def start(self) -> None:
        """Start the AI service by initializing the client session."""
        if self._client is None:
            self._client = aiohttp.ClientSession()
            print_info("AI Service client session initialized")

    async def generate_documentation(
        self, context: DocumentationContext, schema: dict[str, Any] | None = None
    ) -> ProcessingResult:
        start_time = time.time()
        print_phase_header("Documentation Generation")

        try:
            self.logger.info(f"Source code length: {len(context.source_code)}")
            self.logger.info(f"First 50 characters of source code: {context.source_code[:50]}...")

            if not context.source_code or not context.source_code.strip():
                self.logger.error(
                    "Source code is missing or empty",
                    extra={
                        "correlation_id": self.correlation_id,
                        "sanitized_info": {
                            "status": "error",
                            "type": "missing_source",
                            "module": context.metadata.get("module_name", "unknown"),
                            "file": context.metadata.get("file_path", "unknown")
                        }
                    }
                )
                raise DocumentationError("Source code is missing or empty")

            module_name = (
                context.metadata.get("module_name", "") if context.metadata else ""
            )
            file_path = (
                context.metadata.get("file_path", "") if context.metadata else ""
            )

            display_processing_phase("Context Information", {
                "Module": module_name or "Unknown",
                "File": file_path or "Unknown",
                "Code Length": len(context.source_code),
                "Classes": len(context.classes),
                "Functions": len(context.functions)
            })

            # Convert classes and functions to proper types
            classes = []
            if context.classes:
                for cls_data in context.classes:
                    if isinstance(cls_data, ExtractedClass):
                        classes.append(cls_data)
                    else:
                        classes.append(ExtractedClass(**cls_data))

            functions = []
            if context.functions:
                for func_data in context.functions:
                    if isinstance(func_data, ExtractedFunction):
                        functions.append(func_data)
                    else:
                        functions.append(ExtractedFunction(**func_data))

            # Create documentation prompt
            self.logger.info("Generating documentation prompt.")
            self.logger.debug(f"Source code before creating prompt: {context.source_code[:50]}...")
            prompt = await self.prompt_manager.create_documentation_prompt(
                module_name=module_name,
                file_path=file_path,
                source_code=context.source_code,
                classes=classes,
                functions=functions,
            )

            # Add function calling instructions to the prompt
            if schema:
                prompt = self.prompt_manager.get_prompt_with_schema(prompt, schema)

            # Get the function schema
            function_schema = self.prompt_manager.get_function_schema(schema)

            # Validate and prepare request
            request_params = await self.token_manager.validate_and_prepare_request(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            if request_params["max_tokens"] < 100:
                print_info("Warning: Token availability is low. Consider reducing prompt size.")

            print_info("Making API call to generate documentation")
            async with self.semaphore:
                response = await self._make_api_call_with_retry(
                    str(prompt),
                    function_schema
                )

            # Add source code to content
            if isinstance(response, dict):
                response["source_code"] = context.source_code
                response.setdefault("code_metadata", {})["source_code"] = context.source_code

            # Parse response into DocstringData
            print_info("Parsing and validating response")
            self.logger.debug(f"Source code before parsing response: {context.source_code[:50]}...")
            parsed_response = await self.response_parser.parse_response(
                response,
                expected_format="docstring",
                validate_schema=True
            )

            if not parsed_response.validation_success:
                self.logger.error(
                    "Response validation failed",
                    extra={
                        "correlation_id": self.correlation_id,
                        "sanitized_info": {
                            "status": "error",
                            "type": "validation",
                            "errors": parsed_response.errors
                        }
                    }
                )
                raise DataValidationError(
                    f"Response validation failed: {parsed_response.errors}"
                )

            # Create validated DocstringData instance
            content_copy = parsed_response.content.copy()
            content_copy.pop('source_code', None)  # Remove source_code if present
            self.logger.debug(f"Source code after parsing response: {context.source_code[:50]}...")
            docstring_data = DocstringData(
                summary=str(content_copy.get("summary", "")),
                description=str(content_copy.get("description", "")),
                args=content_copy.get("args", []),
                returns=content_copy.get("returns", {"type": "Any", "description": ""}),
                raises=content_copy.get("raises", []),
                complexity=int(content_copy.get("complexity", 1))
            )
            is_valid, validation_errors = docstring_data.validate()

            if not is_valid:
                self.logger.error(
                    "Docstring validation failed",
                    extra={
                        "correlation_id": self.correlation_id,
                        "sanitized_info": {
                            "status": "error",
                            "type": "docstring_validation",
                            "errors": validation_errors
                        }
                    }
                )
                raise DataValidationError(f"Docstring validation failed: {validation_errors}")

            # Track metrics
            processing_time = time.time() - start_time
            self.logger.info(f"Documentation generation completed in {processing_time:.2f} seconds.")
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=True,
                duration=processing_time,
                metadata={
                    "module": module_name,
                    "file": file_path,
                    "code_length": len(context.source_code),
                    "classes": len(context.classes),
                    "functions": len(context.functions)
                },
                usage=response.get("usage", {})
            )

            # Display metrics
            api_metrics: dict[str, Any] = {
                "Processing Time": f"{processing_time:.2f}s",
                "Prompt Tokens": response.get("usage", {}).get("prompt_tokens", 0),
                "Completion Tokens": response.get("usage", {}).get("completion_tokens", 0),
                "Total Tokens": response.get("usage", {}).get("total_tokens", 0)
            }
            display_api_metrics(api_metrics)

            return ProcessingResult(
                content=docstring_data.to_dict(),
                usage=response.get("usage", {}),
                metrics={
                    "processing_time": processing_time,
                    "validation_success": True
                },
                validation_status=True,
                validation_errors=[],
                schema_errors=[]
            )

        except DataValidationError as e:
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=False,
                duration=time.time() - start_time,
                metadata={"error_type": "validation_error", "error_message": str(e)}
            )
            raise
        except DocumentationError as e:
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=False,
                duration=time.time() - start_time,
                metadata={"error_type": "documentation_error", "error_message": str(e)}
            )
            raise
        except Exception as e:
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=False,
                duration=time.time() - start_time,
                metadata={"error_type": "generation_error", "error_message": str(e)}
            )
            raise APICallError(str(e)) from e

    def _format_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Format the response to ensure it has the expected structure.

        Args:
            response: Raw response from the API

        Returns:
            Properly formatted response with choices array
        """
        if "choices" in response and isinstance(response["choices"], list):
            return response

        if "summary" in response or "description" in response:
            return {
                "choices": [{"message": {"content": json.dumps(response)}}],
                "usage": response.get("usage", {}),
            }

        if "function_call" in response:
            return {
                "choices": [{"message": {"function_call": response["function_call"]}}],
                "usage": response.get("usage", {}),
            }

        if "tool_calls" in response:
            return {
                "choices": [{"message": {"tool_calls": response["tool_calls"]}}],
                "usage": response.get("usage", {}),
            }

        return {
            "choices": [{"message": {"content": json.dumps(response)}}],
            "usage": {},
        }

    async def _make_api_call_with_retry(
        self, prompt: str, function_schema: dict[str, Any], max_retries: int = 3
    ) -> dict[str, Any]:
        """
        Makes an API call to the AI model with retry logic.

        Args:
            prompt: The prompt to send to the AI model.
            function_schema: The function schema for function calling.
            max_retries: Maximum number of retries for the API call.

        Returns:
            dict[str, Any]: The raw response from the AI model.

        Raises:
            APICallError: If all retries fail.
        """
        headers = {"api-key": self.config.api_key, "Content-Type": "application/json"}

        if self.correlation_id:
            headers["x-correlation-id"] = self.correlation_id

        request_params = await self.token_manager.validate_and_prepare_request(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        request_params["functions"] = [function_schema]
        request_params["function_call"] = {"name": "generate_docstring"}

        request_metrics: dict[str, Any] = {
            "Prompt Tokens": request_params.get("max_tokens", 0),
            "Temperature": request_params.get("temperature", 0),
            "Retries": max_retries
        }
        display_metrics(request_metrics, title="API Request Parameters")

        for attempt in range(max_retries):
            try:
                endpoint = self.config.endpoint.rstrip("/") + "/"
                path = f"openai/deployments/{self.config.deployment}/chat/completions"
                url = urljoin(endpoint, path) + "?api-version=2024-10-21"

                if self._client is None:
                    await self.start()

                if self._client is None:
                    raise APICallError("Failed to initialize client session")

                print_info(f"Making API call (attempt {attempt + 1}/{max_retries})")
                async with self._client.post(
                    url,
                    headers=headers,
                    json=request_params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    
                    error_text = await response.text()
                    if attempt == max_retries - 1:
                        raise APICallError(
                            f"API call failed after {max_retries} retries: {error_text}"
                        )
                    print_info(f"API call failed (attempt {attempt + 1}): {error_text}")
                    await asyncio.sleep(2**attempt)  # Exponential backoff

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == max_retries - 1:
                    raise APICallError(
                        f"API call failed after {max_retries} retries due to client error: {e}"
                    ) from e
                print_info(f"Client error (attempt {attempt + 1}): {e}")
                await asyncio.sleep(2**attempt)

        raise APICallError(f"API call failed after {max_retries} retries.")

    async def close(self) -> None:
        """Closes the aiohttp client session."""
        if self._client:
            await self._client.close()
            self._client = None
            print_info("AI Service client session closed")

    async def __aenter__(self) -> "AIService":
        """Enter method for async context manager."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit method for async context manager."""
        await self.close()
