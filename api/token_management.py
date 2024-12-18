"""Token management module for interacting with the OpenAI API."""

import tiktoken
import json
from typing import Any, Tuple, Union, Optional
import time
import asyncio

from core.config import AIConfig
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.types import TokenUsage
from core.exceptions import ProcessingError
from core.console import print_info, print_success, print_error
from core.metrics_collector import MetricsCollector


class TokenManager:
    """Manages token usage and cost estimation for Azure OpenAI API interactions."""

    def __init__(
        self,
        model: str,
        config: Optional[AIConfig] = None,
        correlation_id: Optional[str] = None,
        metrics_collector: Optional['MetricsCollector'] = None
    ) -> None:
        """Initialize TokenManager with Azure OpenAI configurations."""
        self.logger = CorrelationLoggerAdapter(
            logger=LoggerSetup.get_logger(__name__),
            extra={"correlation_id": correlation_id}
        )
        self.config = config if config else AIConfig.from_env()
        self.model = model
        self.deployment_id = self.config.deployment
        self.metrics_collector = metrics_collector or MetricsCollector(
            correlation_id=correlation_id
        )
        self.correlation_id = correlation_id

        # Initialize token tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Initialize encoding
        try:
            base_model = self._get_base_model_name(self.model)
            self.encoding = tiktoken.encoding_for_model(base_model)
        except KeyError:
            self.logger.warning(
                f"Model {self.model} not found. Using cl100k_base encoding."
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self.model_config = self.config.model_limits.get(
            self.model,
            self.config.model_limits["gpt-4o-2024-11-20"]
        )

        self._initialize_rate_limiting()

    def _get_base_model_name(self, model_name: str) -> str:
        """
        Get the base model name from a deployment model name.
        
        The gpt-4o-2024-11-20 model has specific capabilities and requirements
        that we need to account for in our token encoding and management.
        """
        model_mappings = {
            "gpt-4o": "gpt-4o-2024-11-20",  # New primary model
            "gpt-35-turbo": "gpt-3.5-turbo",  # Keep fallback options
            "gpt-3.5-turbo": "gpt-3.5-turbo",
        }

        for key, value in model_mappings.items():
            if key in model_name.lower():
                return value

        self.logger.warning(
            f"Unknown model {model_name}, defaulting to gpt-4o for token encoding"
        )
        return "gpt-4o-2024-11-20"  # Default to our primary model

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

    def _initialize_rate_limiting(self) -> None:
        """Initialize rate limiting for Azure OpenAI API."""
        self.requests_this_minute = 0
        self.minute_start = time.time()
        self.rate_limit_per_minute = getattr(self.model_config, 'rate_limit', 10)  # Default to 10 if not specified
        self.request_times: list[float] = []

    async def validate_and_prepare_request(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
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
            # Check rate limits
            await self._check_rate_limits()

            # Estimate tokens
            prompt_tokens = self._estimate_tokens(prompt)
            available_tokens = self.model_config.max_tokens - prompt_tokens

            if prompt_tokens > self.model_config.max_tokens:
                raise ValueError(
                    f"Prompt exceeds Azure OpenAI token limit: {prompt_tokens} > "
                    f"{self.model_config.max_tokens}"
                )

            # Calculate max completion tokens
            max_completion = self._calculate_max_completion(
                available_tokens, 
                max_tokens
            )

            # Prepare request parameters
            request_params = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_completion,
                "temperature": temperature or self.config.temperature,
            }

            # Track token usage
            self.track_request(prompt_tokens, max_completion)

            # Log the input sent to the AI
            with open("ai_input.log", "a") as log_file:
                log_file.write(f"{json.dumps(request_params, indent=2)}\n")
            return request_params

        except Exception as e:
            self.logger.error(f"Error preparing request: {e}", exc_info=True)
            raise ProcessingError(f"Failed to prepare request: {str(e)}")
            print_error(f"Failed to prepare request: {str(e)}")
            raise ProcessingError(f"Failed to prepare request: {str(e)}")

    async def _check_rate_limits(self) -> None:
        """Check and enforce Azure OpenAI rate limits."""
        current_time = time.time()
        
        # Clean old request times
        self.request_times = [
            t for t in self.request_times 
            if current_time - t < 60
        ]

        # Check rate limit
        if len(self.request_times) >= self.rate_limit_per_minute:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                self.logger.warning(
                    f"Rate limit reached. Waiting {wait_time:.2f} seconds."
                )
                await asyncio.sleep(wait_time)

        self.request_times.append(current_time)

    def _calculate_max_completion(
        self,
        available_tokens: int,
        max_tokens: Optional[int] = None
    ) -> int:
        """Calculate the maximum completion tokens based on availability and config."""
        if max_tokens:
            max_completion = min(max_tokens, available_tokens)
        else:
            max_completion = min(
                available_tokens,
                self.model_config.chunk_size
            )

        max_completion = max(1, max_completion)
        if max_completion < available_tokens:
            self.logger.debug(
                f"Adjusted completion tokens to {max_completion} (available: {available_tokens})"
            )
        return max_completion

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
