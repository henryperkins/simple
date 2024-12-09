"""
Token Management Module.

Centralizes all token-related operations for Azure OpenAI API.
"""

from typing import Optional, Dict, Any, Tuple, Union
from core.config import AIConfig
from core.logger import LoggerSetup, log_debug, log_error, log_info
from utils import (
    TokenCounter,
    serialize_for_logging,
    get_env_var
)
from core.types import TokenUsage
from core.exceptions import ProcessingError
import tiktoken


class TokenManager:
    """Manages all token-related operations for Azure OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4",
        deployment_id: Optional[str] = None,
        config: Optional[AIConfig] = None,
        metrics_collector: Optional[Any] = None,
    ) -> None:
        """
        Initialize the TokenManager.

        Args:
            model (str): The model name to use. Defaults to "gpt-4".
            deployment_id (Optional[str]): The deployment ID for the model.
            config (Optional[AIConfig]): Configuration for Azure OpenAI.
            metrics_collector (Optional[Any]): Collector for metrics.
        """
        self.logger = LoggerSetup.get_logger(__name__)
        self.config = config or AIConfig.from_env()
        # Use deployment from config if not explicitly provided
        self.deployment_id = deployment_id or self.config.deployment
        # Use model from config if not explicitly provided
        self.model = model or self.config.model
        self.metrics_collector = metrics_collector

        try:
            # For Azure OpenAI, we'll use the base model name for encoding
            base_model = self._get_base_model_name(self.model)
            self.encoding = tiktoken.encoding_for_model(base_model)
        except KeyError:
            self.logger.warning(f"Model {self.model} not found. Falling back to 'cl100k_base' encoding.")
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self.model_config = self.config.model_limits.get(
            self.model, self.config.model_limits["gpt-4"]
        )

        # Initialize counters
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.logger.info("TokenManager initialized.")

    def _get_base_model_name(self, model_name: str) -> str:
        """
        Get the base model name from a deployment model name.

        Args:
            model_name (str): The model name or deployment name.

        Returns:
            str: The base model name for token encoding.
        """
        # Map Azure OpenAI deployment names to base model names
        model_mappings = {
            "gpt-4": "gpt-4",
            "gpt-35-turbo": "gpt-3.5-turbo",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
        }
        
        # Remove any version numbers or suffixes
        base_name = model_name.split('-')[0].lower()
        
        # Try to match with known models
        for key, value in model_mappings.items():
            if key.startswith(base_name):
                return value
                
        # Default to gpt-4 if unknown
        self.logger.warning(f"Unknown model {model_name}, defaulting to gpt-4 for token encoding")
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
            self.logger.error(f"Error estimating tokens: {e}")
            return len(text) // 4  # Rough fallback estimate

    def _calculate_usage(self, prompt_tokens: int, completion_tokens: int) -> TokenUsage:
        """Calculate token usage statistics.
        
        Args:
            prompt_tokens (int): Number of tokens in the prompt
            completion_tokens (int): Number of tokens in the completion
            
        Returns:
            TokenUsage: Token usage statistics including cost calculation
        """
        total_tokens = prompt_tokens + completion_tokens
        
        # Calculate costs based on model config
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
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
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
            max_completion = max_tokens or min(
                self.model_config.max_tokens - prompt_tokens,
                self.model_config.chunk_size,
            )
            max_completion = max(1, max_completion)
            total_tokens = prompt_tokens + max_completion
            if total_tokens > self.model_config.max_tokens:
                max_completion = max(1, self.model_config.max_tokens - prompt_tokens)
                self.logger.warning(
                    f"Total tokens ({total_tokens}) exceed model max tokens ({self.model_config.max_tokens}). Adjusting max_completion to {max_completion}."
                )

            self.logger.debug(
                f"Token calculation: prompt={prompt_tokens}, max_completion={max_completion}, total={total_tokens}"
            )

            # For Azure OpenAI, we use the deployment_id as the model
            request_params = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_completion,
                "temperature": temperature or self.config.temperature,
            }

            self.track_request(prompt_tokens, max_completion)

            return request_params

        except Exception as e:
            self.logger.error(f"Error preparing request: {e}", exc_info=True)
            raise ProcessingError(f"Failed to prepare request: {str(e)}")

    def get_usage_stats(self) -> Dict[str, Union[int, float]]:
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
        self.logger.info(f"Current Usage Stats: {stats}")
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
            f"Tracked request - Prompt Tokens: {prompt_tokens}, "
            f"Max Completion Tokens: {max_completion}"
        )

    async def process_completion(self, completion: Any) -> Tuple[str, Dict[str, int]]:
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
            
            # Handle both regular responses and function call responses
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
                        duration=0,  # Duration can be updated based on actual metrics
                        usage=usage,
                        metadata={
                            "model": self.model,
                            "deployment_id": self.deployment_id,
                        },
                    )

                self.logger.info(
                    f"Processed completion - Content Length: {len(content)}, Usage: {usage}"
                )

            return content, usage

        except Exception as e:
            self.logger.error(f"Error processing completion: {e}", exc_info=True)
            raise ProcessingError(f"Failed to process completion: {str(e)}")
