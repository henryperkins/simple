"""Token management module for interacting with the OpenAI API."""

import tiktoken
import sentry_sdk
from typing import Any, Tuple, Union, Optional
import time
import asyncio

from core.config import AIConfig
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.types import TokenUsage
from core.exceptions import ProcessingError
from core.metrics_collector import MetricsCollector
from core.console import (
    print_section_break,
    display_metrics,
    print_success,
    print_warning,
    print_error,
)


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
        except KeyError as e:
            self.logger.warning(
                f"Model {self.model} not found. Using cl100k_base encoding.",
                exc_info=True,
                extra={"model": self.model}
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")
            sentry_sdk.capture_exception(e)

        self.model_config = self.config.model_limits.get(
            self.model,
            self.config.model_limits["gpt-4o-2024-11-20"]
        )

        self._initialize_rate_limiting()

    def _get_base_model_name(self, model_name: str) -> str:
        """
        Get the base model name from a deployment model name.
        """
        model_mappings = {
            "gpt-4o": "gpt-4o-2024-11-20",  # New primary model
            "gpt-35-turbo": "gpt-3.5-turbo",  # Keep fallback options
            "gpt-3.5-turbo": "gpt-3.5-turbo",
        }

        for key, value in model_mappings.items():
            if key in model_name.lower():
                return value

        print_warning(f"⚠️ Unknown model '{model_name}', defaulting to gpt-4o for token encoding.")
        self.logger.warning(
            f"Unknown model {model_name}, defaulting to gpt-4o for token encoding",
             extra={"model": model_name}
        )
        return "gpt-4o-2024-11-20"  # Default to our primary model

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        """
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            self.logger.error(f"Error estimating tokens: {e}", exc_info=True, extra={"text_snippet": text[:50]})
            print_error(f"🔥 Error estimating tokens. Using fallback estimate.") # Added user-facing error
            return len(text) // 4

    def _calculate_usage(self, prompt_tokens: int, completion_tokens: int) -> TokenUsage:
        """Calculate token usage statistics."""
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
        self.rate_limit_per_minute = getattr(
            self.model_config, 'rate_limit', 10)  # Default to 10 if not specified
        self.request_times: list[float] = []

    async def validate_and_prepare_request(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        truncation_strategy: Optional[str] = None,
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        response_format: Optional[str] = None,
        stream_options: Optional[dict] = None
    ) -> dict[str, Any]:
        """
        Validate and prepare a request with token management.

        Args:
            prompt: The input prompt for the API.
            max_tokens: The maximum tokens for the completion.
            temperature: Sampling temperature for the model.
            truncation_strategy: Strategy for truncating input if needed.
            tool_choice: Tool selection for function calling.
            parallel_tool_calls: Whether to allow parallel tool calls.
            response_format: Expected response format.
            stream_options: Options for streaming responses.

        Returns:
            A dictionary of request parameters for the API call.

        Raises:
            ValueError: If the total token count exceeds the model's limit.
        """
        try:
            # Check rate limits
            await self._check_rate_limits()

            # Estimate tokens
            prompt_tokens = self._estimate_tokens(prompt)
            available_tokens = self.model_config.max_tokens - prompt_tokens

            if prompt_tokens > self.model_config.max_tokens:
                error_msg = f"Prompt exceeds Azure OpenAI token limit: {prompt_tokens} > {self.model_config.max_tokens}"
                print_error(f"🔥 {error_msg}")
                raise ValueError(error_msg)

            # Calculate max completion tokens
            max_completion = min(max_tokens or available_tokens, available_tokens)

            if prompt_tokens + max_completion > self.model_config.max_tokens:
                error_msg = f"Total token count exceeds limit: {prompt_tokens + max_completion} > {self.model_config.max_tokens}"
                print_error(f"🔥 {error_msg}")
                raise ValueError(error_msg)

            # Prepare request parameters
            request_params = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_completion,
                "temperature": temperature or self.config.temperature,
            }

            # Track token usage
            self.track_request(prompt_tokens, max_completion)

            # Log the input sent to the AI
            self.logger.debug(
                "Prepared Request Parameters",
                extra={"request_params": request_params}
            )
            return request_params

        except Exception as e:
            self.logger.error(f"Error preparing request: {e}", exc_info=True, extra={"prompt_snippet": prompt[:50]})
            print_error(f"🔥 Failed to prepare request: {e}")
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
                print_warning(f"⏳ Rate limit reached. Waiting {wait_time:.2f} seconds before the next request.")
                self.logger.warning(
                    f"Rate limit reached. Waiting {wait_time:.2f} seconds.",
                    extra={"wait_time": wait_time}
                )
                await asyncio.sleep(wait_time)

        self.request_times.append(current_time)

    def get_usage_stats(self) -> dict[str, Union[int, float]]:
        """
        Get current token usage statistics.

        Returns:
            A dictionary containing token usage statistics.
        """
        usage = self._calculate_usage(
            self.total_prompt_tokens, self.total_completion_tokens
        )
        stats = {
            "Total Prompt Tokens": self.total_prompt_tokens,
            "Total Completion Tokens": self.total_completion_tokens,
            "Total Tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "Estimated Cost": f"${usage.estimated_cost:.4f}",
        }
        print_section_break()
        print_info("📊 Current Token Usage Statistics:")
        display_metrics(stats)
        print_section_break()
        self.logger.info("Token usage stats retrieved.", stats)
        return stats

    def track_request(self, prompt_tokens: int, max_completion: int) -> None:
        """
        Track token usage for a request.

        Args:
            prompt_tokens: Number of tokens in the prompt.
            max_completion: Number of tokens allocated for the completion.
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += max_completion
        print_success(f"✅ Tracked Tokens - Prompt: {prompt_tokens}, Completion: {max_completion} (Total: {prompt_tokens + max_completion})")
        self.logger.info(
            f"Tracked request - Prompt Tokens: {prompt_tokens}, Max Completion Tokens: {max_completion}",
            extra={"correlation_id": self.correlation_id, "prompt_tokens": prompt_tokens, "max_completion": max_completion}
        )

    async def process_completion(self, completion: Any) -> Tuple[str, dict[str, Any]]:
        """
        Process completion response and track token usage.

        Args:
            completion: The raw completion response from the API.

        Returns:
            A tuple containing the processed content and usage statistics.

        Raises:
            ProcessingError: If there is an error processing the completion response.
        """
        try:
            message = completion["choices"][0]["message"]

            if "function_call" in message:
                content = message["function_call"]["arguments"]
            else:
                content = message.get("content", "")

            usage = completion.get("usage", {})
            
            if usage:
                self.total_completion_tokens += usage.get("completion_tokens", 0)
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
                    extra={"correlation_id": self.correlation_id, "content_length": len(content), "usage": usage}
                )
                print_success(f"✅ Completion processed successfully - Usage: {usage}") # More informative success

            return content, usage if isinstance(usage, dict) else {}

        except Exception as e:
            self.logger.error(f"Error processing completion: {e}", exc_info=True)
            print_error(f"🔥 Failed to process completion: {e}")
            raise ProcessingError(f"Failed to process completion: {str(e)}")