# ai_service.py

import asyncio
import json
import time
from typing import Any, Dict, Optional, TypeVar

import aiohttp

from core.config import AIConfig
from core.exceptions import (
    APICallError,
    DocumentationError,
    InvalidRequestError,
)
from core.logger import CorrelationLoggerAdapter, LoggerSetup
from core.metrics_collector import MetricsCollector
from core.prompt_manager import PromptManager
from core.types.base import DocumentationContext, ProcessingResult
from utils import log_and_raise_error

T = TypeVar("T")

class AIService:
    """
    Manages interactions with the Azure OpenAI API.
    """

    def __init__(
        self,
        config: Optional[AIConfig] = None,
        correlation_id: Optional[str] = None,
    ) -> None:

        self.config: AIConfig = config
        self.correlation_id = correlation_id
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(
                f"{__name__}.{self.__class__.__name__}",
                correlation_id=self.correlation_id,
            ),
            extra={"correlation_id": self.correlation_id},
        )

        self.prompt_manager = PromptManager(correlation_id=correlation_id)
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)

        # Token manager and other components can be initialized here or via dependency injection
        from core.dependency_injection import Injector  # local import to avoid circular dependency

        self.token_manager = Injector.get("token_manager")

        self.semaphore = asyncio.Semaphore(self.config.api_call_semaphore_limit)
        self._client_session: Optional[aiohttp.ClientSession] = None

    async def close(self) -> None:
        """Close the aiohttp client session."""
        if self._client_session:
            await self._client_session.close()
            self.logger.info("AI Service client session closed")

    async def start(self) -> None:
        """Start the aiohttp client session if not already started."""
        if self._client_session is None:
            self._client_session = aiohttp.ClientSession()
            self.logger.info("AI Service client session initialized")

    async def _make_api_call_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        log_extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an API call with retry logic following Azure best practices."""

        headers = {
            "Content-Type": "application/json",
            "api-key": self.config.api_key,
        }
        request_body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_completion_tokens,
            "temperature": self.config.temperature,
        }
        if "model" in request_body:
            del request_body["model"]
        self.logger.info(f"Request body: {request_body}")
        url = f"{self.config.endpoint}openai/deployments/{self.config.deployment_name}/chat/completions?api-version={self.config.api_version}"
        self.logger.info(f"API URL: {url}")
        url = f"{self.config.endpoint}openai/deployments/{self.config.deployment_name}/chat/completions?api-version={self.config.api_version}"
        self.logger.debug(f"API URL: {url}")
        if hasattr(self.config, "functions") and self.config.functions:
            request_body["functions"] = self.config.functions
        if hasattr(self.config, "function_call") and self.config.function_call:
            request_body["function_call"] = self.config.function_call

        if self.correlation_id:
            headers["x-correlation-id"] = self.correlation_id

        request_body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_completion_tokens,
            "temperature": self.config.temperature,
        }

        # Construct the API URL
        url = f"{self.config.endpoint}openai/deployments/{self.config.deployment_name}/chat/completions?api-version={self.config.api_version}"

        self.logger.debug(f"Constructed API URL: {url}")
        self.logger.debug(f"Request headers: {headers}")
        self.logger.debug(f"Endpoint: {self.config.endpoint}")
        self.logger.debug(f"Deployment Name: {self.config.deployment_name}")
        self.logger.debug(f"API Version: {self.config.api_version}")
        retry_count = 0
        last_error = None

        while retry_count <= max_retries:
            try:
                if not self._client_session:
                    await self.start()

                async with self._client_session.post(
                    url,
                    headers=headers,
                    json=request_body,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        self.logger.info("API call succeeded", extra=log_extra)
                        return response_json
                    else:
                        error_text = await response.text()
                        self.logger.error(
                            f"API call failed with status {response.status}: {error_text}",
                            extra={
                                "status_code": response.status,
                                "correlation_id": self.correlation_id,
                            },
                        )
                        if response.status == 404:
                            raise APICallError("Resource not found. Check your endpoint and deployment name.")
                        else:
                            raise APICallError(f"API call failed with status {response.status}: {error_text}")

            except Exception as e:
                last_error = e
                self.logger.error(
                    f"API call failed (attempt {retry_count + 1}/{max_retries}): {str(e)}"
                )
                retry_count += 1
                if retry_count <= max_retries:
                    backoff = 2 ** retry_count
                    self.logger.info(f"Retrying in {backoff} seconds...")
                    await asyncio.sleep(backoff)
                else:
                    break  # Exceeded max retries

        raise APICallError(f"API call failed after {max_retries} retries") from last_error

    async def generate_documentation(
        self, context: DocumentationContext
    ) -> ProcessingResult:
        """Generate documentation for the provided source code context."""
        start_time = time.time()
        log_extra = {
            "correlation_id": self.correlation_id,
            "module": context.metadata.get("module_name", ""),
            "file": context.metadata.get("file_path", ""),
        }

        try:
            # Create the documentation prompt
            prompt_result = await self.prompt_manager.create_documentation_prompt(context)
            prompt = prompt_result.content["prompt"]

            # Validate prompt size
            is_valid, metrics, message = self.token_manager.validate_request(
                prompt, self.config.max_completion_tokens
            )

            if not is_valid:
                self.logger.warning(f"Prompt exceeds token limit: {message}")
                # Try optimizing the prompt
                optimized_prompt, usage = self.token_manager.optimize_prompt(
                    prompt,
                    max_tokens=self.config.max_tokens - self.config.max_completion_tokens,
                    preserve_sections=["summary", "description", "parameters", "returns"],
                )
                prompt = optimized_prompt

                # Re-validate after optimization
                is_valid, metrics, message = self.token_manager.validate_request(
                    prompt, self.config.max_completion_tokens
                )

                if not is_valid:
                    self.logger.error(f"Unable to reduce prompt size: {message}")
                    raise DocumentationError("Prompt size exceeds token limit even after optimization.")

            # Make the API call
            response = await self._make_api_call_with_retry(
                prompt,
                max_retries=self.config.api_call_max_retries,
                log_extra=log_extra,
            )

            # Parse and process the response
            # Assuming there's a method to parse the response
            # For now, we'll extract the content from the response
            choices = response.get("choices", [])
            if not choices:
                raise ResponseParsingError("No choices found in the API response.")

            message_content = choices[0]["message"]["content"]

            processing_time = time.time() - start_time
            return ProcessingResult(
                content=message_content,
                usage=response.get("usage", {}),
                metrics={"processing_time": processing_time},
                validation_status=True,
                validation_errors=[],
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
