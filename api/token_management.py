"""Token management module for interacting with the OpenAI API."""

import tiktoken
from typing import Any, Tuple, Union

from core.config import AIConfig
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.types import TokenUsage
from core.exceptions import ProcessingError
from core.console import print_info, print_success, print_error
from core.metrics_collector import MetricsCollector

class TokenManager:
    """Manages token usage and cost estimation for OpenAI API interactions."""

    def __init__(
        self,
        model: str,
        config: AIConfig | None = None,
        correlation_id: str | None = None,
        metrics_collector: MetricsCollector | None = None
    ) -> None:
        """Initialize TokenManager with model and configuration.

        Args:
            model: The model name.
            config: The AI configuration.
            correlation_id: Optional correlation ID for logging.
            metrics_collector: Optional metrics collector instance.
        """
        self.logger = CorrelationLoggerAdapter(
            logger=LoggerSetup.get_logger(__name__),
            extra={"correlation_id": correlation_id}
        )
        self.config = config if config else AIConfig.from_env()
        self.model = model
        self.deployment_id = self.config.deployment
        self.metrics_collector = metrics_collector or MetricsCollector(correlation_id=correlation_id)
        self.correlation_id = correlation_id

        try:
            base_model = self._get_base_model_name(self.model)
            self.encoding = tiktoken.encoding_for_model(base_model)
        except KeyError:
            self.logger.warning(
                f"Model {self.model} not found. Falling back to 'cl100k_base' encoding."
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self.model_config = self.config.model_limits.get(
            self.model, self.config.model_limits["gpt-4"]
        )

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        print_info("TokenManager initialized.", {"model": model, "correlation_id": correlation_id})

    def _get_base_model_name(self, model_name: str) -> str:
        """
        Get the base model name from a deployment model name.

        Args:
            model_name (str): The model name or deployment name.

        Returns:
            str: The base model name for token encoding.
        """
        model_mappings = {
            "gpt-4": "gpt-4",
            "gpt-35-turbo": "gpt-3.5-turbo",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
        }

        base_name = model_name.split('-')[0].lower()

        for key, value in model_mappings.items():
            if key.startswith(base_name):
                return value

        self.logger.warning(
            f"Unknown model {model_name}, defaulting to gpt-4 for token encoding"
        )
        return "gpt-4"

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.

        Args:
            text (str): The text to estimate tokens for.

        Returns:
            int: Estimated number of tokens.
        """
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            self.logger.error(f"Error estimating tokens: {e}", exc_info=True)
            return len(text) // 4

    def _calculate_usage(self, prompt_tokens: int, completion_tokens: int) -> TokenUsage:
        """Calculate token usage statistics.

        Args:
            prompt_tokens (int): Number of tokens in the prompt
            completion_tokens (int): Number of tokens in the completion

        Returns:
            TokenUsage: Token usage statistics including cost calculation
        """
        total_tokens = prompt_tokens + completion_tokens
        cost_per_token = self.model_config.cost_per_token
        estimated_cost = total_tokens * cost_per_token

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost
        )

    async def validate_and_prepare_request(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None
    ) -> dict[str, Any]:
        """
        Validate and prepare a request with token management.

        Args:
            prompt (str): The prompt to send to the API.
            max_tokens (Optional[int]): Optional maximum tokens for completion.
            temperature (Optional[float]): Optional temperature setting.

        Returns:
            Dict[str, Any]: Validated request parameters.

        Raises:
            ProcessingError: If request preparation fails.
        """
        try:
            prompt_tokens = self._estimate_tokens(prompt)
            available_tokens = self.model_config.max_tokens - prompt_tokens

            if prompt_tokens > self.model_config.max_tokens:
                raise ValueError(f"Prompt exceeds maximum token limit for model {self.model}: {prompt_tokens} > {self.model_config.max_tokens}")

            if max_tokens:
                max_completion = min(max_tokens, available_tokens)
            else:
                max_completion = min(
                    available_tokens, self.model_config.chunk_size
                )

            max_completion = max(1, max_completion)

            if max_completion == 0:
                self.logger.warning(
                    f"Estimated prompt tokens ({prompt_tokens}) close to or exceeding model's max tokens limit ({self.model_config.max_tokens}).",
                    extra={"correlation_id": self.correlation_id}
                )

            if max_completion < available_tokens:
                self.logger.debug(
                    f"Adjusted completion tokens to {max_completion} (prompt: {prompt_tokens}, "
                    f"available: {available_tokens})",
                    extra={"correlation_id": self.correlation_id}
                )

            total_tokens = prompt_tokens + max_completion
            self.logger.debug(
                f"Token calculation: prompt={prompt_tokens}, max_completion={max_completion}, total={total_tokens}",
                extra={"correlation_id": self.correlation_id}
            )

            request_params = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_completion,
                "temperature": temperature or self.config.temperature,
            }

            self.track_request(prompt_tokens, max_completion)

            print_info(f"Validated request: {prompt_tokens} tokens in prompt, {available_tokens} available for completion.")
            return request_params

        except Exception as e:
            self.logger.error(f"Error preparing request: {e}", exc_info=True)
            print_error(f"Failed to prepare request: {str(e)}")
            raise ProcessingError(f"Failed to prepare request: {str(e)}")

    def get_usage_stats(self) -> dict[str, Union[int, float]]:
        """
        Get current token usage statistics.

        Returns:
            Dict[str, Union[int, float]]: Current token usage and estimated cost.
        """
        usage = self._calculate_usage(
            self.total_prompt_tokens, self.total_completion_tokens
        )
        stats = {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "estimated_cost": usage.estimated_cost,
        }
        print_info("Token usage stats retrieved.", stats)
        return stats

    def track_request(self, prompt_tokens: int, max_completion: int) -> None:
        """
        Track token usage for a request.

        Args:
            prompt_tokens (int): Number of tokens in the prompt.
            max_completion (int): Number of tokens allocated for completion.
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += max_completion
        self.logger.info(
            f"Tracked request - Prompt Tokens: {prompt_tokens}, Max Completion Tokens: {max_completion}",
            extra={"correlation_id": self.correlation_id}
        )
        print_success(f"Tokens tracked: {prompt_tokens + max_completion} total tokens.")

    async def process_completion(self, completion: Any) -> Tuple[str, dict[str, Any]]:
        """
        Process completion response and track token usage.

        Args:
            completion (Any): The completion response from the API.

        Returns:
            Tuple[str, Dict[str, int]]: Completion content and usage statistics.

        Raises:
            ProcessingError: If processing the completion fails.
        """
        try:
            message = completion["choices"][0]["message"]

            if "function_call" in message:
                content = message["function_call"]["arguments"]
            else:
                content = message.get("content", "")

            usage = completion.get("usage", {})

            if usage:
                self.total_completion_tokens += usage.get(
                    "completion_tokens", 0)
                self.total_prompt_tokens += usage.get("prompt_tokens", 0)

                if self.metrics_collector:
                    await self.metrics_collector.track_operation(
                        "token_usage",
                        success=True,
                        duration=0,
                        usage=self._calculate_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)).__dict__,
                        metadata={
                            "model": self.model,
                            "deployment_id": self.deployment_id,
                            "correlation_id": self.correlation_id
                        },
                    )

                self.logger.info(
                    f"Processed completion - Content Length: {len(content)}, Usage: {usage}",
                    extra={"correlation_id": self.correlation_id}
                )
                print_success("Completion processed successfully.")

            return content, usage

        except Exception as e:
            self.logger.error(f"Error processing completion: {e}", exc_info=True)
            print_error(f"Failed to process completion: {str(e)}")
            raise ProcessingError(f"Failed to process completion: {str(e)}")
