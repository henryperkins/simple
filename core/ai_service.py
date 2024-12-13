from typing import Dict, Any, Optional
import asyncio
from urllib.parse import urljoin

import aiohttp
import json

from core.config import AIConfig
from core.console import print_info, print_error, print_warning
from core.docstring_processor import DocstringProcessor
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.prompt_manager import PromptManager
from core.dependency_injection import Injector
from api.token_management import TokenManager
from core.exceptions import DocumentationGenerationError, APICallError
from core.types.base import ProcessingResult, DocumentationContext


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
        self._client = None

        print_info(
            f"Initializing AI service with correlation ID: {self.correlation_id}"
        )
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
        self, context: "DocumentationContext"
    ) -> "ProcessingResult":
        """
        Generates documentation using the AI model.

        Args:
            context: Documentation context containing source code and metadata.

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

            async with self.semaphore:
                response = await self._make_api_call_with_retry(prompt)

            parsed_response = await self.response_parser.parse_response(
                response, expected_format="docstring"
            )

            # Process and validate the parsed response
            docstring_data = self.docstring_processor.parse(parsed_response.content)
            #is_valid, validation_errors = self.docstring_processor.validate( # REMOVED
            #    docstring_data
            #)
            is_valid = parsed_response.validation_success
            validation_errors = parsed_response.errors
            self.logger.info(f"Docstring validation status: {is_valid}")

            if not is_valid:
                print_warning(
                    f"Docstring validation failed: {', '.join(validation_errors)} with correlation ID: {self.correlation_id}"
                )
                self.logger.warning(
                    f"Docstring validation failed: {', '.join(validation_errors)}",
                    extra={"correlation_id": self.correlation_id},
                )

                return ProcessingResult(
                    content={"error": "Docstring validation failed"},
                    usage={},
                    metrics={
                        "processing_time": parsed_response.parsing_time,
                        "response_size": len(str(response)),
                        "validation_success": is_valid,
                    },
                    is_cached=False,
                    processing_time=parsed_response.parsing_time,
                    validation_status=False,
                    validation_errors=validation_errors,
                    schema_errors=parsed_response.errors,
                )

            # Return the validated and processed docstring
            return ProcessingResult(
                content=docstring_data.to_dict(),
                usage=response.get("usage", {}),
                metrics={
                    "processing_time": parsed_response.parsing_time,
                    "response_size": len(str(response)),
                    "validation_success": is_valid,
                },
                is_cached=False,
                processing_time=parsed_response.parsing_time,
                validation_status=is_valid,
                validation_errors=validation_errors,
                schema_errors=parsed_response.errors,
            )

        except Exception as e:
            self.logger.error(
                f"Error generating documentation: {e}",
                exc_info=True,
                extra={"correlation_id": self.correlation_id},
            )
            print_error(
                f"Error: {e} during generate_documentation in ai_service "
                f"with correlation ID: {self.correlation_id}"
            )
            raise DocumentationGenerationError(str(e)) from e

    async def _make_api_call_with_retry(
        self, prompt: str, max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Makes an API call to the AI model with retry logic.

        Args:
            prompt: The prompt to send to the AI model.
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

        # Add function calling parameters
        function_schema = self.prompt_manager.get_function_schema()
        request_params["functions"] = [function_schema]
        request_params["function_call"] = {"name": "generate_docstring"}

        for attempt in range(max_retries):
            try:
                endpoint = self.config.endpoint.rstrip("/") + "/"
                path = f"openai/deployments/{self.config.deployment}/chat/completions"
                url = urljoin(endpoint, path) + "?api-version=2024-10-21"

                if self._client is None:
                    await self.start()

                self.logger.debug(
                    f"API call attempt {attempt + 1}, URL: {url}",
                    extra={
                        "correlation_id": self.correlation_id,
                        "request_headers": headers,
                        "request_body": json.dumps(request_params),
                    },
                )

                async with self._client.post(
                    url,
                    headers=headers,
                    json=request_params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        self.logger.debug(
                            f"API call successful, response: {json.dumps(response_json)}",
                            extra={"correlation_id": self.correlation_id},
                        )
                        return response_json
                    else:
                        error_text = await response.text()
                        self.logger.error(
                            f"API call failed with status {response.status}: "
                            f"{error_text}",
                            extra={"correlation_id": self.correlation_id, "response_text": error_text, "status_code": response.status},
                        )
                        if attempt == max_retries - 1:
                            raise APICallError(
                                f"API call failed after {max_retries} retries: "
                                f"{error_text}"
                            ) from None
                        await asyncio.sleep(2**attempt)  # Exponential backoff

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                self.logger.error(
                    f"Error during API call attempt {attempt + 1}: {e}",
                    extra={"correlation_id": self.correlation_id},
                )
                if attempt == max_retries - 1:
                    raise APICallError(
                        f"API call failed after {max_retries} retries due to client error: {e}"
                    ) from e

        raise APICallError(f"API call failed after {max_retries} retries.")

    async def close(self) -> None:
        """Closes the aiohttp client session."""
        print_info(f"Closing AI service with correlation ID: {self.correlation_id}")
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
        exc_tb: Optional[Any],
    ) -> None:
        """Exit method for async context manager."""
        await self.close()
