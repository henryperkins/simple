"""
Token Management Module

Handles token counting, optimization, and management for Azure OpenAI API
requests. Provides efficient token estimation and prompt optimization.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import tiktoken

from core.logger import LoggerSetup
from core.config import AzureOpenAIConfig

# Initialize logger and config
logger = LoggerSetup.get_logger(__name__)
config = AzureOpenAIConfig.from_env()


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
        deployment_name: Optional[str] = None
    ):
        """
        Initialize TokenManager with model configuration.
        
        Args:
            model (str): The model name to use for token management
            deployment_name (Optional[str]): Azure deployment name if different from model
        """
        self.model = self._get_model_name(deployment_name, model)
        self.deployment_name = deployment_name

        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
        self.model_config = config.model_limits.get(
            self.model, config.model_limits["gpt-4"]
        )
        logger.debug(
            f"TokenManager initialized for model: {self.model}, "
            f"deployment: {deployment_name}"
        )

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _get_model_name(self, deployment_name: Optional[str], default_model: str) -> str:
        """
        Get the appropriate model name based on deployment name or default model.
        
        Args:
            deployment_name (Optional[str]): The deployment name
            default_model (str): The default model name
            
        Returns:
            str: The resolved model name
        """
        if deployment_name:
            return config.model_limits.get(deployment_name, default_model)
        return default_model

    @lru_cache(maxsize=1024)
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text (str): Text to estimate tokens for
            
        Returns:
            int: Estimated token count
        """
        try:
            tokens = len(self.encoding.encode(
                text,
                disallowed_special=(),
                allowed_special={'<|endoftext|>'}
            ))
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
            prompt (str): The prompt text to validate
            max_completion_tokens (Optional[int]): Maximum allowed completion tokens
            
        Returns:
            Tuple[bool, Dict[str, Union[int, float]], str]: Validation result,
            metrics, and message
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
                    message = (
                        f"Input split into {len(chunks)} chunks due to "
                        f"token limit"
                    )
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
            text (str): Text to split into chunks
            
        Returns:
            List[str]: List of text chunks
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
            prompt_tokens (int): Number of tokens in the prompt
            completion_tokens (int): Number of tokens in the completion
            cached (bool): Whether to use cached pricing rates

        Returns:
            TokenUsage: Token usage statistics including cost
        """
        total_tokens = prompt_tokens + completion_tokens
        
        if cached and "cached_cost_per_1k_prompt" in self.model_config:
            prompt_cost = (prompt_tokens / 1000) * self.model_config[
                "cached_cost_per_1k_prompt"
            ]
            completion_cost = (completion_tokens / 1000) * self.model_config[
                "cached_cost_per_1k_completion"
            ]
        else:
            prompt_cost = (prompt_tokens / 1000) * self.model_config[
                "cost_per_1k_prompt"
            ]
            completion_cost = (completion_tokens / 1000) * self.model_config[
                "cost_per_1k_completion"
            ]

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=prompt_cost + completion_cost
        )

    def track_request(self, request_tokens: int, response_tokens: int) -> None:
        """
        Track token usage for a request.

        Args:
            request_tokens (int): Number of tokens in the request
            response_tokens (int): Number of tokens in the response
        """
        try:
            self.total_prompt_tokens += request_tokens
            self.total_completion_tokens += response_tokens
            logger.debug(
                f"Tracked request: {request_tokens} prompt tokens, "
                f"{response_tokens} completion tokens"
            )
        except Exception as e:
            logger.error(f"Error tracking request: {e}")
            raise

    def get_usage_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get current token usage statistics.

        Returns:
            Dict[str, int]: Total prompt and completion tokens
        """
        try:
            return {
                "total_prompt_tokens": int(self.total_prompt_tokens),
                "total_completion_tokens": int(self.total_completion_tokens)
            }
        except Exception as e:
            logger.error(f"Error getting usage stats: {e}")
            return {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            }

    def get_model_limits(self) -> Dict[str, int]:
        """
        Get the token limits for the current model.

        Returns:
            Dict[str, int]: Dictionary containing model token limits
        """
        try:
            return {
                "max_tokens": self.model_config["max_tokens"],
                "chunk_size": self.model_config["chunk_size"],
                "max_prompt_tokens": self.model_config["chunk_size"],
                "max_completion_tokens": (
                    self.model_config["max_tokens"] -
                    self.model_config["chunk_size"]
                )
            }
        except Exception as e:
            logger.error(f"Error getting model limits: {e}")
            return {
                "max_tokens": 0,
                "chunk_size": 0,
                "max_prompt_tokens": 0,
                "max_completion_tokens": 0
            }

    def get_token_costs(self, cached: bool = False) -> Dict[str, float]:
        """
        Get the token costs for the current model.

        Args:
            cached (bool): Whether to return cached pricing rates

        Returns:
            Dict[str, float]: Dictionary containing token costs per 1k tokens
        """
        try:
            if cached and "cached_cost_per_1k_prompt" in self.model_config:
                return {
                    "prompt_cost_per_1k": self.model_config[
                        "cached_cost_per_1k_prompt"
                    ],
                    "completion_cost_per_1k": self.model_config[
                        "cached_cost_per_1k_completion"
                    ]
                }
            return {
                "prompt_cost_per_1k": self.model_config["cost_per_1k_prompt"],
                "completion_cost_per_1k": self.model_config[
                    "cost_per_1k_completion"
                ]
            }
        except Exception as e:
            logger.error(f"Error getting token costs: {e}")
            return {
                "prompt_cost_per_1k": 0.0,
                "completion_cost_per_1k": 0.0
            }

    def reset_cache(self) -> None:
        """Reset the token estimation cache."""
        try:
            self.estimate_tokens.cache_clear()
            logger.debug("Token estimation cache cleared")
        except Exception as e:
            logger.error(f"Error resetting cache: {e}")


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate token count for text.

    Args:
        text (str): Text to estimate tokens for
        model (str): Model name to use for estimation

    Returns:
        int: Estimated token count
    """
    manager = TokenManager(model)
    return manager.estimate_tokens(text)


def chunk_text(text: str, model: str = "gpt-4") -> List[str]:
    """
    Split text into chunks that fit within token limits.

    Args:
        text (str): Text to split into chunks
        model (str): Model name to use for chunking

    Returns:
        List[str]: List of text chunks
    """
    manager = TokenManager(model)
    return manager.chunk_text(text)