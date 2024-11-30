"""
Token Management Module

Handles token counting, optimization, and management for Azure OpenAI API
requests. Provides efficient token estimation and prompt optimization.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import asyncio  # Import for asynchronous operations
import tiktoken

from core.logger import LoggerSetup
from core.config import AzureOpenAIConfig
from core.metrics_collector import MetricsCollector


# Initialize logger
logger = LoggerSetup.get_logger(__name__)


@dataclass
class TokenUsage:
    """Token usage statistics and cost calculation."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float


class TokenManager:
    """
    Manages token counting, optimization, and cost calculation for Azure OpenAI
    API requests. Handles different models and their token limits and pricing.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        deployment_name: Optional[str] = None,
        config: Optional[AzureOpenAIConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ) -> None:
        """
        Initialize TokenManager with model configuration.

        Args:
            model (str): The model name to use for token management.
            deployment_name (Optional[str]): Azure deployment name if different
                                             from model.
            config (Optional[AzureOpenAIConfig]): Configuration object.
            metrics_collector (Optional[MetricsCollector]): Metrics collector
                                                            for tracking
                                                            operations.
        """
        self.config = config or AzureOpenAIConfig.from_env()
        self.model = self._get_model_name(deployment_name, model)
        self.deployment_name = deployment_name
        self.metrics_collector = metrics_collector

        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self.model_config = self.config.model_limits.get(
            self.model, self.config.model_limits["gpt-4"]
        )
        logger.debug(f"TokenManager initialized for model: {self.model}, "
                     f"deployment: {deployment_name}")

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _get_model_name(self, deployment_name: Optional[str],
                        default_model: str) -> str:
        """
        Get the model name based on deployment name or default model.

        Args:
            deployment_name (Optional[str]): Deployment name to check.
            default_model (str): Default model name to use if deployment name
                                 is not found.

        Returns:
            str: The model name to use.
        """
        if deployment_name:
            return self.config.model_limits.get(deployment_name, default_model)
        return default_model

    @lru_cache(maxsize=1024)
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text (str): Text to estimate tokens for.

        Returns:
            int: Estimated token count.
        """
        try:
            tokens = len(self.encoding.encode(text))  # Simplified encoding
            logger.debug(f"Estimated {tokens} tokens for text")
            return tokens
        except Exception as e:
            logger.error(f"Error estimating tokens: {e}")
            return 0

    async def validate_request(
        self,
        prompt: str,
        max_completion_tokens: Optional[int] = None
    ) -> Tuple[bool, Dict[str, Union[int, float]], str]:
        """
        Validate if request is within token limits.

        Args:
            prompt (str): The prompt text to validate.
            max_completion_tokens (Optional[int]): Maximum allowed completion
                                                   tokens.

        Returns:
            Tuple[bool, Dict[str, Union[int, float]], str]: Validation result,
            metrics, and message.
        """
        try:
            prompt_tokens = self.estimate_tokens(prompt)
            max_completion = max_completion_tokens or (
                self.model_config["chunk_size"] - prompt_tokens
            )
            total_tokens = prompt_tokens + max_completion

            metrics = {
                "prompt_tokens": prompt_tokens,
                "max_completion_tokens": max_completion,
                "total_tokens": total_tokens,
                "model_limit": self.model_config["max_tokens"],
                "chunk_size": self.model_config["chunk_size"]
            }

            if total_tokens > self.model_config["chunk_size"]:
                chunks = self.chunk_text(prompt)
                if len(chunks) > 1:
                    message = (f"Input split into {len(chunks)} chunks due to "
                               "token limit")
                    logger.info(message)
                    return True, metrics, message

            logger.info("Request validated successfully")
            return True, metrics, "Request validated successfully"
        except Exception as e:
            logger.error(f"Error validating request: {e}")
            return False, {}, str(e)

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks that fit within token limits.

        Args:
            text (str): Text to split into chunks.

        Returns:
            List[str]: List of text chunks.
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_size = self.model_config["chunk_size"]

        sentences = text.split('. ')
        for sentence in sentences:
            sent_tokens = self.estimate_tokens(sentence)
            if current_tokens + sent_tokens <= chunk_size:
                current_chunk.append(sentence)
                current_tokens += sent_tokens
            else:
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_tokens = sent_tokens

        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        return chunks

    def calculate_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached: bool = False
    ) -> TokenUsage:
        """
        Calculate token usage and cost.

        Args:
            prompt_tokens (int): Number of tokens in the prompt.
            completion_tokens (int): Number of tokens in the completion.
            cached (bool): Whether to use cached pricing rates.

        Returns:
            TokenUsage: Token usage statistics including cost.
        """
        total_tokens = prompt_tokens + completion_tokens
        prompt_cost = (
            (prompt_tokens / 1000) *
            self.model_config.get("cached_cost_per_1k_prompt",
                                  self.model_config["cost_per_1k_prompt"])
            if cached else
            (prompt_tokens / 1000) * self.model_config["cost_per_1k_prompt"]
        )
        completion_cost = (
            (completion_tokens / 1000) *
            self.model_config.get("cached_cost_per_1k_completion",
                                  self.model_config["cost_per_1k_completion"])
            if cached else
            (completion_tokens / 1000) *
            self.model_config["cost_per_1k_completion"]
        )
        return TokenUsage(prompt_tokens, completion_tokens, total_tokens,
                          prompt_cost + completion_cost)

    def track_request(self, request_tokens: int, response_tokens: int) -> None:
        """
        Track token usage for a request.

        Args:
            request_tokens (int): Number of tokens in the request.
            response_tokens (int): Number of tokens in the response.
        """
        try:
            self.total_prompt_tokens += request_tokens
            self.total_completion_tokens += response_tokens
            logger.debug(f"Tracked request: {request_tokens} prompt, "
                         f"{response_tokens} completion tokens")

            if self.metrics_collector:
                usage = {"prompt_tokens": request_tokens,
                         "completion_tokens": response_tokens}
                asyncio.create_task(
                    self.metrics_collector.track_operation("token_usage",
                                                           True, 0, usage)
                )

        except Exception as e:
            logger.error(f"Error tracking request: {e}")
            raise

    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get current token usage statistics.

        Returns:
            Dict[str, int]: Total prompt and completion tokens.
        """
        return {"total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens}

    def get_model_limits(self) -> Dict[str, int]:
        """
        Get token limits for the current model.

        Returns:
            Dict[str, int]: Dictionary containing model token limits.
        """
        try:
            return {
                "max_tokens": self.model_config["max_tokens"],
                "chunk_size": self.model_config["chunk_size"],
                "max_prompt_tokens": self.model_config["chunk_size"],
                "max_completion_tokens": (self.model_config["max_tokens"] -
                                          self.model_config["chunk_size"]),
            }
        except Exception as e:
            logger.error(f"Error getting model limits: {e}")
            return {"max_tokens": 0, "chunk_size": 0, "max_prompt_tokens": 0,
                    "max_completion_tokens": 0}

    def get_token_costs(self, cached: bool = False) -> Dict[str, float]:
        """
        Get token costs for the current model.

        Args:
            cached (bool): Whether to return cached pricing rates.

        Returns:
            Dict[str, float]: Dictionary containing token costs per 1k tokens.
        """
        try:
            if cached and "cached_cost_per_1k_prompt" in self.model_config:
                return {
                    "prompt_cost_per_1k": self.model_config[
                        "cached_cost_per_1k_prompt"],
                    "completion_cost_per_1k": self.model_config[
                        "cached_cost_per_1k_completion"],
                }
            return {
                "prompt_cost_per_1k": self.model_config["cost_per_1k_prompt"],
                "completion_cost_per_1k": self.model_config[
                    "cost_per_1k_completion"],
            }
        except Exception as e:
            logger.error(f"Error getting token costs: {e}")
            return {"prompt_cost_per_1k": 0.0, "completion_cost_per_1k": 0.0}

    def reset_cache(self) -> None:
        """Reset the token estimation cache."""
        self.estimate_tokens.cache_clear()
        logger.debug("Token estimation cache cleared")


# Helper functions (outside the class)
def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate token count for text.

    Args:
        text (str): Text to estimate tokens for.
        model (str): Model name to use for estimation.

    Returns:
        int: Estimated token count.
    """
    manager = TokenManager(model=model)
    return manager.estimate_tokens(text)


def chunk_text(text: str, model: str = "gpt-4") -> List[str]:
    """
    Split text into chunks that fit within token limits.

    Args:
        text (str): Text to split into chunks.
        model (str): Model name to use for chunking.

    Returns:
        List[str]: List of text chunks.
    """
    manager = TokenManager(model=model)
    return manager.chunk_text(text)
