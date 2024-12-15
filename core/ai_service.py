"""
Service for interacting with the AI model to generate documentation.
"""
from typing import Any, Dict, Optional
import asyncio
from urllib.parse import urljoin
import json
import aiohttp
import time

from core.config import AIConfig
from core.docstring_processor import DocstringProcessor
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.prompt_manager import PromptManager
from core.dependency_injection import Injector
from api.token_management import TokenManager
from core.exceptions import APICallError, DataValidationError
from core.types.base import ProcessingResult, DocumentationContext, DocstringData, DocstringSchema


class AIService:
    """
    Service for interacting with the AI model to generate documentation.
    """

    def __init__(
        self, config: Optional[AIConfig] = None, correlation_id: Optional[str] = None
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
        try:
            self.docstring_processor = Injector.get("docstring_processor")
        except KeyError:
            self.logger.warning(
                "Docstring processor not registered, using default",
                extra={"correlation_id": self.correlation_id},
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
            "AIService instance created",
            extra={"correlation_id": self.correlation_id},
        )

    async def start(self) -> None:
        """
        Start the AI service by initializing the client session.
        """
        if self._client is None:
            self._client = aiohttp.ClientSession()

    async def generate_documentation(
        self, context: "DocumentationContext", schema: Optional[Dict[str, Any]] = None
    ) -> "ProcessingResult":
        """
        Generates documentation using the AI model.

        Args:
            context: Documentation context containing source code and metadata.
            schema: Optional schema for structured output.

        Returns:
            ProcessingResult containing enhanced documentation or error information.

        Raises:
            DocumentationGenerationError: If there's an issue generating documentation.
        """
        self.logger.info(
            "generate_documentation called",
            extra={"correlation_id": self.correlation_id},
        )
        try:
            module_name = (
                context.metadata.get("module_name", "") if context.metadata else ""
            )
            file_path = (
                context.metadata.get("file_path", "") if context.metadata else ""
            )

            # Create documentation prompt
            prompt = await self.prompt_manager.create_documentation_prompt(
                module_name=module_name,
                file_path=file_path,
                source_code=context.source_code,
                classes=context.classes,
                functions=context.functions,
            )

            # Add function calling instructions to the prompt
            if schema:
                prompt = self.prompt_manager.get_prompt_with_schema(prompt, schema)

            # Get the function schema
            function_schema = self.prompt_manager.get_function_schema(schema)

            async with self.semaphore:
                response = await self._make_api_call_with_retry(prompt, function_schema)

            # Parse response into DocstringData
            parsed_response = await self.response_parser.parse_response(
                response, 
                expected_format="docstring",
                validate_schema=True
            )

            if not parsed_response.validation_success:
                self.logger.error(
                    f"Response validation failed: {parsed_response.errors}",
                    extra={"correlation_id": self.correlation_id}
                )
                raise DataValidationError(
                    f"Response validation failed: {parsed_response.errors}"
                )

            # Create validated DocstringData instance
            docstring_data = DocstringData(**parsed_response.content)
            is_valid, validation_errors = docstring_data.validate()

            if not is_valid:
                self.logger.error(
                    f"Docstring validation failed: {validation_errors}",
                    extra={"correlation_id": self.correlation_id}
                )
                raise DataValidationError(f"Docstring validation failed: {validation_errors}")

            return ProcessingResult(
                content=docstring_data.to_dict(),
                usage=response.get("usage", {}),
                metrics={
                    "processing_time": parsed_response.parsing_time,
                    "validation_success": True
                },
                validation_status=True,
                validation_errors=[],
                schema_errors=[]
            )

        except DataValidationError as e:
            self.logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            self.logger.error(
                f"Error generating documentation: {e}",
                exc_info=True,
                extra={"correlation_id": self.correlation_id},
            )
            raise APICallError(str(e)) from e

    def _format_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
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
        self, prompt: str, function_schema: Dict[str, Any], max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Makes an API call to the AI model with retry logic.

        Args:
            prompt: The prompt to send to the AI model.
            function_schema: The function schema for function calling.
            max_retries: Maximum number of retries for the API call.

        Returns:
            Dict[str, Any]: The raw response from the AI model.

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

        for attempt in range(max_retries):
            try:
                endpoint = self.config.endpoint.rstrip("/") + "/"
                path = f"openai/deployments/{self.config.deployment}/chat/completions"
                url = urljoin(endpoint, path) + "?api-version=2024-10-21"

                if self._client is None:
                    await self.start()

                async with self._client.post(
                    url,
                    headers=headers,
                    json=request_params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        if attempt == max_retries - 1:
                            raise APICallError(
                                f"API call failed after {max_retries} retries: {error_text}"
                            )
                        await asyncio.sleep(2**attempt)  # Exponential backoff

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == max_retries - 1:
                    raise APICallError(
                        f"API call failed after {max_retries} retries due to client error: {e}"
                    ) from e

        raise APICallError(f"API call failed after {max_retries} retries.")

    async def close(self) -> None:
        """Closes the aiohttp client session."""
        if self._client:
            await self._client.close()
            self._client = None

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
