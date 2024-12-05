"""Token Management Module.

Centralizes all token-related operations for Azure OpenAI API.
"""

from typing import Optional, Dict, Any, Tuple
from functools import lru_cache
from core.config import AzureOpenAIConfig
from core.logger import LoggerSetup
from core.metrics import MetricsCollector
from exceptions import TokenLimitError

import tiktoken

logger = LoggerSetup.get_logger(__name__)

class TokenManager:
    """Manages all token-related operations for Azure OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4",
        deployment_id: Optional[str] = None,
        config: Optional[AzureOpenAIConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ) -> None:
        self.logger = LoggerSetup.get_logger(__name__)
        self.config = config or AzureOpenAIConfig.from_env()
        self.model = self._get_model_name(deployment_id, model)
        self.deployment_id = deployment_id
        self.metrics_collector = metrics_collector

        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self.model_config = self.config.model_limits.get(
            self.model, self.config.model_limits["gpt-4"]
        )

        # Initialize counters
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _get_model_name(self, deployment_id: Optional[str], model: str) -> str:
        """Determine the model name based on deployment ID or default model."""
        return deployment_id or model

    async def validate_and_prepare_request(
        self, 
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Validate and prepare a request with token management.

        Args:
            prompt: The prompt to send to the API
            max_tokens: Optional maximum tokens for completion
            temperature: Optional temperature setting

        Returns:
            Dict containing the validated request parameters

        Raises:
            TokenLimitError: If request exceeds token limits
        """
        try:
            prompt_tokens = self.estimate_tokens(prompt)
            max_completion = max_tokens or (self.model_config["chunk_size"] - prompt_tokens)
            total_tokens = prompt_tokens + max_completion

            if total_tokens > self.model_config["max_tokens"]:
                raise TokenLimitError("Request exceeds model token limit")

            request_params = {
                "model": self.deployment_id or self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_completion,
                "temperature": temperature or 0.3
            }

            # Track the request
            self.track_request(prompt_tokens, max_completion)

            return request_params

        except Exception as e:
            self.logger.error(f"Error preparing request: {e}")
            raise

    async def process_completion(
        self, 
        completion: Any
    ) -> Tuple[str, Dict[str, int]]:
        """Process completion response and track token usage.

        Args:
            completion: The completion response from the API

        Returns:
            Tuple of (completion content, usage statistics)
        """
        try:
            content = completion.choices[0].message.content
            usage = completion.usage.model_dump() if hasattr(completion, 'usage') else {}

            # Update token counts
            if usage:
                self.total_completion_tokens += usage.get('completion_tokens', 0)
                self.total_prompt_tokens += usage.get('prompt_tokens', 0)

                if self.metrics_collector:
                    await self.metrics_collector.track_operation(
                        "token_usage",
                        True,
                        0,
                        usage,
                        metadata={
                            "model": self.model,
                            "deployment_id": self.deployment_id
                        }
                    )

            return content, usage

        except Exception as e:
            self.logger.error(f"Error processing completion: {e}")
            raise

    @lru_cache(maxsize=1024)
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        try:
            tokens = len(self.encoding.encode(text))
            return tokens
        except Exception as e:
            self.logger.error(f"Error estimating tokens: {e}")
            raise

    def track_request(self, prompt_tokens: int, max_completion: int) -> None:
        """Track token usage for a request."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += max_completion