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
from exceptions import TokenLimitError
from core.types import TokenUsage
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
            config (Optional[AzureOpenAIConfig]): Configuration for Azure OpenAI.
            metrics_collector (Optional[Any]): Collector for metrics.
        """
        self.logger = LoggerSetup.get_logger(__name__)
        self.config = config or AzureOpenAIConfig.from_env()
        self.model = self._get_model_name(deployment_id, model)
        self.deployment_id = deployment_id
        self.metrics_collector = metrics_collector

        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
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

    def _get_model_name(self, deployment_id: Optional[str], model: str) -> str:
        """
        Determine the model name based on deployment ID or default model.

        Args:
            deployment_id (Optional[str]): The deployment ID for the model.
            model (str): The default model name.

        Returns:
            str: The resolved model name.
        """
        resolved_model = deployment_id or model
        self.logger.debug(f"Resolved model name: {resolved_model}")
        return resolved_model

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
            TokenLimitError: If request exceeds token limits.
        """
        try:
            prompt_tokens = self.estimate_tokens(prompt)
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

            request_params = {
                "model": self.deployment_id or self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_completion,
                "temperature": temperature or 0.7,
            }

            self.track_request(prompt_tokens, max_completion)

            return request_params

        except Exception as e:
            self.logger.error(f"Error preparing request: {e}", exc_info=True)
            raise

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
        """
        try:
            content = completion.choices[0].message.content
            usage = (
                completion.usage.model_dump() if hasattr(completion, "usage") else {}
            )

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
            raise
