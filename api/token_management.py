"""
Token Management Module

Handles token counting, optimization, and management for Azure OpenAI API requests.
Provides efficient token estimation and prompt optimization with caching.

Version: 1.2.0
Author: Development Team
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import tiktoken

from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.metrics_collector import MetricsCollector

@dataclass
class TokenUsage:
    """Token usage statistics and cost calculation."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float

class TokenManager:
    """
    Manages token counting, optimization, and cost calculation for Azure OpenAI API requests.
    Handles different models and their specific token limits and pricing.
    """

    # Token limits and pricing for different models
    MODEL_LIMITS = {
        "gpt-4": {
            "max_tokens": 8192,
            "cost_per_1k_prompt": 0.03,
            "cost_per_1k_completion": 0.06
        },
        "gpt-4-32k": {
            "max_tokens": 32768,
            "cost_per_1k_prompt": 0.06,
            "cost_per_1k_completion": 0.12
        },
        "gpt-3.5-turbo": {
            "max_tokens": 4096,
            "cost_per_1k_prompt": 0.0015,
            "cost_per_1k_completion": 0.002
        },
        "gpt-3.5-turbo-16k": {
            "max_tokens": 16384,
            "cost_per_1k_prompt": 0.003,
            "cost_per_1k_completion": 0.004
        },
        "gpt-4o-2024-08-06": {
            "max_tokens": 8192,
            "cost_per_1k_prompt": 2.50,
            "cost_per_1k_completion": 10.00,
            "cached_cost_per_1k_prompt": 1.25,
            "cached_cost_per_1k_completion": 5.00
        }
    }

    # Mapping of deployment names to model names
    DEPLOYMENT_TO_MODEL = {
        "gpt-4o": "gpt-4o-2024-08-06",
        "gpt-4": "gpt-4",
        "gpt-4-32k": "gpt-4-32k",
        "gpt-35-turbo": "gpt-3.5-turbo",
        "gpt-35-turbo-16k": "gpt-3.5-turbo-16k"
    }

    def __init__(self, model: str = "gpt-4", deployment_name: Optional[str] = None, correlation_id: Optional[str] = None):
        """
        Initialize TokenManager with model configuration.

        Args:
            model (str): The model name to use for token management
            deployment_name (Optional[str]): The deployment name, which may differ from model name
            correlation_id (Optional[str]): Correlation ID for logging
        """
        # Map deployment name to model if provided
        if deployment_name:
            self.model = self.DEPLOYMENT_TO_MODEL.get(deployment_name, model)
        else:
            self.model = model

        self.deployment_name = deployment_name

        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback to cl100k_base for newer models
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
        # Get model configuration
        self.model_config = self.MODEL_LIMITS.get(self.model, self.MODEL_LIMITS["gpt-4"])
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__),
            extra={"correlation_id": correlation_id}
        )
        self.logger.debug(f"TokenManager initialized for model: {self.model}, deployment: {deployment_name}")

        # Initialize token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Initialize metrics collector
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)

    @lru_cache(maxsize=128)
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text with caching.

        Args:
            text (str): Text to estimate tokens for

        Returns:
            int: Estimated token count
        """
        try:
            tokens = len(self.encoding.encode(text))
            self.logger.debug(f"Estimated {tokens} tokens for text")
            return tokens
        except Exception as e:
            self.logger.error(f"Error estimating tokens: {e}")
            return 0

    def optimize_prompt(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        preserve_sections: Optional[List[str]] = None
    ) -> Tuple[str, TokenUsage]:
        """
        Optimize prompt to fit within token limits while preserving essential sections.

        Args:
            text (str): Text to optimize
            max_tokens (Optional[int]): Maximum allowed tokens
            preserve_sections (Optional[List[str]]): Sections that must be preserved

        Returns:
            Tuple[str, TokenUsage]: Optimized text and token usage statistics
        """
        max_tokens = max_tokens or int(self.model_config["max_tokens"] * 0.75)
        current_tokens = self.estimate_tokens(text)

        if current_tokens <= max_tokens:
            return text, self._calculate_usage(current_tokens, 0)

        try:
            # Split into sections
            sections = text.split('\n\n')
            preserved = []
            optional = []
            
            # Identify critical sections to preserve
            preserve_sections = preserve_sections or ["summary", "description", "parameters", "returns", "raises"]
            
            # First pass: Keep only the most critical sections with length limits
            for section in sections:
                if any(p in section.lower() for p in preserve_sections):
                    # Limit size of preserved sections
                    if self.estimate_tokens(section) > (max_tokens // 4):
                        section = self._trim_section(section, max_tokens // 4)
                    preserved.append(section)
                elif len(optional) < 5:  # Limit number of optional sections
                    optional.append(section)

            # Start with preserved content
            optimized = '\n\n'.join(preserved)
            tokens_so_far = self.estimate_tokens(optimized)
            
            if tokens_so_far > max_tokens:
                # If still too large, aggressively trim each section
                preserved_sections = []
                max_section_tokens = max_tokens // len(preserved)
                for section in preserved:
                    trimmed_section = self._trim_section(section, max_section_tokens)
                    if self.estimate_tokens(trimmed_section) <= max_section_tokens:
                        preserved_sections.append(trimmed_section)
                optimized = '\n\n'.join(preserved_sections)
            else:
                # Add optional sections that fit, with size limits
                remaining_tokens = max_tokens - tokens_so_far
                max_optional_tokens = remaining_tokens // 2
                for section in optional:
                    if self.estimate_tokens(section) > max_optional_tokens:
                        section = self._trim_section(section, max_optional_tokens)
                    section_tokens = self.estimate_tokens(section)
                    if section_tokens <= remaining_tokens:
                        optimized = f"{optimized}\n\n{section}"
                        remaining_tokens -= section_tokens

            final_tokens = self.estimate_tokens(optimized)
            if final_tokens > max_tokens:
                # Final failsafe: emergency truncation
                optimized = self._emergency_truncate(optimized, max_tokens)
                final_tokens = self.estimate_tokens(optimized)

            return optimized, self._calculate_usage(final_tokens, 0)

        except Exception as e:
            self.logger.error(f"Error optimizing prompt: {e}")
            return self._emergency_truncate(text, max_tokens), self._calculate_usage(max_tokens, 0)

    def _trim_section(self, section: str, max_tokens: int) -> str:
        """Trim a section to fit within token limit while preserving meaning."""
        current_tokens = self.estimate_tokens(section)
        if current_tokens <= max_tokens:
            return section

        # Split into lines and preserve important parts
        lines = section.split('\n')
        if len(lines) <= 2:  # Very short section
            return section[:int(len(section) * (max_tokens / current_tokens))]

        # Keep first and last lines if possible
        result = [lines[0]]
        remaining_tokens = max_tokens - self.estimate_tokens(lines[0]) - self.estimate_tokens(lines[-1])
        
        # Add middle lines that fit
        for line in lines[1:-1]:
            line_tokens = self.estimate_tokens(line)
            if line_tokens <= remaining_tokens:
                result.append(line)
                remaining_tokens -= line_tokens
        
        result.append(lines[-1])
        return '\n'.join(result)

    def _emergency_truncate(self, text: str, max_tokens: int) -> str:
        """Emergency truncation when other methods fail."""
        tokens = self.estimate_tokens(text)
        if tokens <= max_tokens:
            return text

        try:
            # Try to preserve important sections by keeping first 40% and last 20%
            total_chars = len(text)
            front_chars = int(total_chars * 0.4)
            back_chars = int(total_chars * 0.2)
            
            truncated = text[:front_chars] + "\n...\n" + text[-back_chars:]
            
            while self.estimate_tokens(truncated) > max_tokens and len(truncated) > 100:
                front_chars = int(front_chars * 0.8)  # Reduce by 20% each time
                back_chars = int(back_chars * 0.8)
                truncated = text[:front_chars] + "\n...\n" + text[-back_chars:]
            
            return truncated

        except Exception:
            # If all else fails, do simple truncation
            ratio = max_tokens / tokens
            char_estimate = int(len(text) * ratio)
            return text[:char_estimate] + "..."

    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit while trying to preserve meaning."""
        if self.estimate_tokens(text) <= max_tokens:
            return text

        sections = text.split('\n\n')
        result = []
        current_tokens = 0

        # Always include first section
        result.append(sections[0])
        current_tokens = self.estimate_tokens(sections[0])

        # Try to add complete sections
        for section in sections[1:]:
            section_tokens = self.estimate_tokens(section)
            if current_tokens + section_tokens <= max_tokens:
                result.append(section)
                current_tokens += section_tokens
            else:
                break

        return '\n\n'.join(result)

    def _calculate_usage(self, prompt_tokens: int, completion_tokens: int, cached: bool = False) -> TokenUsage:
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
        
        if cached:
            prompt_cost = (prompt_tokens / 1000) * self.model_config["cached_cost_per_1k_prompt"]
            completion_cost = (completion_tokens / 1000) * self.model_config["cached_cost_per_1k_completion"]
        else:
            prompt_cost = (prompt_tokens / 1000) * self.model_config["cost_per_1k_prompt"]
            completion_cost = (completion_tokens / 1000) * self.model_config["cost_per_1k_completion"]

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
        self.total_prompt_tokens += request_tokens
        self.total_completion_tokens += response_tokens
        self.logger.debug(f"Tracked request: {request_tokens} prompt tokens, {response_tokens} completion tokens")

        # Track token usage in MetricsCollector
        self.metrics_collector.track_token_usage(request_tokens, response_tokens)

    def get_usage_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get current token usage statistics.

        Returns:
            Dict[str, int]: Total prompt and completion tokens.
        """
        return {
            "total_prompt_tokens": int(self.total_prompt_tokens),
            "total_completion_tokens": int(self.total_completion_tokens)
        }

    def validate_request(
        self,
        prompt: str,
        max_completion_tokens: Optional[int] = None
    ) -> Tuple[bool, Dict[str, Union[int, float]], str]:
        """
        Validate if request is within token limits.

        Args:
            prompt (str): The prompt to validate
            max_completion_tokens (Optional[int]): Maximum allowed completion tokens

        Returns:
            Tuple[bool, Dict[str, Union[int, float]], str]: 
                - Boolean indicating if request is valid
                - Dictionary of token metrics
                - Status message
        """
        prompt_tokens = self.estimate_tokens(prompt)
        max_completion = max_completion_tokens or (self.model_config["max_tokens"] - prompt_tokens)
        total_tokens = prompt_tokens + max_completion

        metrics = {
            "prompt_tokens": prompt_tokens,
            "max_completion_tokens": max_completion,
            "total_tokens": total_tokens,
            "model_limit": self.model_config["max_tokens"]
        }

        if total_tokens > self.model_config["max_tokens"]:
            message = f"Total tokens ({total_tokens}) exceeds model limit ({self.model_config['max_tokens']})"
            self.logger.error(message)
            return False, metrics, message

        self.logger.info("Request validated successfully")
        return True, metrics, "Request validated successfully"

    def get_model_limits(self) -> Dict[str, int]:
        """
        Get the token limits for the current model.

        Returns:
            Dict[str, int]: Dictionary containing model token limits
        """
        return {
            "max_tokens": self.model_config["max_tokens"],
            "max_prompt_tokens": self.model_config["max_tokens"] // 2,  # Conservative estimate
            "max_completion_tokens": self.model_config["max_tokens"] // 2
        }

    def get_token_costs(self, cached: bool = False) -> Dict[str, float]:
        """
        Get the token costs for the current model.

        Args:
            cached (bool): Whether to return cached pricing rates

        Returns:
            Dict[str, float]: Dictionary containing token costs per 1k tokens
        """
        if cached and "cached_cost_per_1k_prompt" in self.model_config:
            return {
                "prompt_cost_per_1k": self.model_config["cached_cost_per_1k_prompt"],
                "completion_cost_per_1k": self.model_config["cached_cost_per_1k_completion"]
            }
        return {
            "prompt_cost_per_1k": self.model_config["cost_per_1k_prompt"],
            "completion_cost_per_1k": self.model_config["cost_per_1k_completion"]
        }

    def estimate_cost(
        self,
        prompt_tokens: int,
        estimated_completion_tokens: int,
        cached: bool = False
    ) -> float:
        """
        Estimate the cost for a request.

        Args:
            prompt_tokens (int): Number of tokens in the prompt
            estimated_completion_tokens (int): Estimated number of completion tokens
            cached (bool): Whether to use cached pricing rates

        Returns:
            float: Estimated cost in currency units
        """
        usage = self._calculate_usage(prompt_tokens, estimated_completion_tokens, cached)
        return usage.estimated_cost

    def reset_cache(self) -> None:
        """Reset the token estimation cache."""
        self.estimate_tokens.cache_clear()
        self.logger.debug("Token estimation cache cleared")


# Module-level functions for backward compatibility
@lru_cache(maxsize=128)
def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Legacy function for token estimation.

    Args:
        text (str): Text to estimate tokens for
        model (str): Model name to use for estimation

    Returns:
        int: Estimated token count
    """
    manager = TokenManager(model)
    return manager.estimate_tokens(text)

def optimize_prompt(text: str, max_tokens: int = 4000) -> str:
    """
    Legacy function for prompt optimization.

    Args:
        text (str): Text to optimize
        max_tokens (int): Maximum allowed tokens

    Returns:
        str: Optimized text
    """
    manager = TokenManager()
    optimized_text, _ = manager.optimize_prompt(text, max_tokens)
    return optimized_text