---
Module: token_management
File: simple2/api/token_management.py
Generated: 2024-02-20 12:00:00
---

# Token Management

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Classes](#classes)
  - [Methods](#methods)
- [Functions](#functions)
- [Constants & Variables](#constants--variables)
- [Dependencies](#dependencies)
- [Recent Changes](#recent-changes)
- [Source Code](#source-code)

## Overview

Token management system for Azure OpenAI API requests. Handles token counting, optimization, and cost calculations for different models.

## Installation

This module is part of the core API package. No additional installation required.

## Usage

## Classes

| Class | Inherits From | Complexity Score* |
|-------|---------------|------------------|
| `TokenUsage` | `None` | 0 |
| `TokenManager` | `None` | 24 (warning) |

### Methods

| Class | Method | Parameters | Returns | Complexity Score* |
|-------|---------|------------|---------|------------------|
| `TokenManager` | `estimate_tokens` | `(self: Any, text: str)` | `int` | 3 |
| `TokenManager` | `optimize_prompt` | `(self: Any, text: str, max_tokens: int, preserve_sections: str)` | `TokenUsage` | 10 |
| `TokenManager` | `track_request` | `(self: Any, request_tokens: int, response_tokens: int)` | `unknown` | 1 |
| `TokenManager` | `get_usage_stats` | `(self: Any)` | `int` | 1 |
| `TokenManager` | `validate_request` | `(self: Any, prompt: str, max_completion_tokens: int)` | `ValidationResult` | 3 |
| `TokenManager` | `get_model_limits` | `(self: Any)` | `int` | 1 |
| `TokenManager` | `get_token_costs` | `(self: Any, cached: bool)` | `float` | 3 |
| `TokenManager` | `estimate_cost` | `(self: Any, prompt_tokens: int, estimated_completion_tokens: int, cached: bool)` | `float` | 1 |
| `TokenManager` | `reset_cache` | `(self: Any)` | `unknown` | 1 |

## Source Code

```python
"""
Token Management Module

Handles token counting, optimization, and management for Azure OpenAI API requests.
Provides efficient token estimation and prompt optimization with caching.

Version: 1.2.0
Author: Development Team
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union, TypeAlias
import tiktoken

from core.logger import log_debug, log_error, log_info, log_exception


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

    def __init__(self, model: str = "gpt-4", deployment_name: Optional[str] = None):
        """
        Initialize TokenManager with model configuration.

        Args:
            model (str): The model name to use for token management
            deployment_name (Optional[str]): The deployment name, which may differ from model name
        """
        if deployment_name:
            self.model = self.DEPLOYMENT_TO_MODEL.get(deployment_name, model)
        else:
            self.model = model

        self.deployment_name = deployment_name

        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self.model_config = self.MODEL_LIMITS.get(
            self.model, self.MODEL_LIMITS["gpt-4"])
        log_debug(
            f"TokenManager initialized for model: {self.model}, deployment: {deployment_name}")

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

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
            log_debug(f"Estimated {tokens} tokens for text")
            return tokens
        except Exception as e:
            log_exception(f"Error estimating tokens: {e}")
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
        max_tokens = max_tokens or int(self.model_config["max_tokens"] // 2)
        current_tokens = self.estimate_tokens(text)

        if current_tokens <= max_tokens:
            log_info("Prompt is within token limits, no optimization needed.")
            return text, self._calculate_usage(current_tokens, 0)

        try:
            sections = text.split('\n\n')
            preserved = []
            optional = []

            for section in sections:
                if preserve_sections and any(p in section for p in preserve_sections):
                    preserved.append(section)
                else:
                    optional.append(section)

            optimized = '\n\n'.join(preserved)
            remaining_tokens = max_tokens - self.estimate_tokens(optimized)

            for section in optional:
                section_tokens = self.estimate_tokens(section)
                if remaining_tokens >= section_tokens:
                    optimized = f"{optimized}\n\n{section}"
                    remaining_tokens -= section_tokens

            final_tokens = self.estimate_tokens(optimized)
            log_info(
                f"Prompt optimized from {current_tokens} to {final_tokens} tokens")
            return optimized, self._calculate_usage(final_tokens, 0)

        except Exception as e:
            log_exception(f"Error optimizing prompt: {e}")
            return text, self._calculate_usage(current_tokens, 0)

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
            prompt_cost = (prompt_tokens / 1000) * \
                self.model_config["cached_cost_per_1k_prompt"]
            completion_cost = (completion_tokens / 1000) * \
                self.model_config["cached_cost_per_1k_completion"]
        else:
            prompt_cost = (prompt_tokens / 1000) * \
                self.model_config["cost_per_1k_prompt"]
            completion_cost = (completion_tokens / 1000) * \
                self.model_config["cost_per_1k_completion"]

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
        log_debug(
            f"Tracked request: {request_tokens} prompt tokens, {response_tokens} completion tokens")

    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get current token usage statistics.

        Returns:
            Dict[str, int]: Total prompt and completion tokens.
        """
        return {
            "total_prompt_tokens": int(self.total_prompt_tokens),
            "total_completion_tokens": int(self.total_completion_tokens)
        }

    # Type aliases for better readability
    MetricsDict: TypeAlias = Dict[str, Union[int, float]]
    ValidationResult: TypeAlias = Tuple[bool, MetricsDict, str]

    def validate_request(
        self,
        prompt: str,
        max_completion_tokens: Optional[int] = None
    ) -> ValidationResult:
        """
        Validate if request is within token limits.

        Args:
            prompt (str): The prompt to validate
            max_completion_tokens (Optional[int]): Maximum allowed completion tokens

        Returns:
            ValidationResult: 
                - Boolean indicating if request is valid
                - Dictionary of token metrics
                - Status message
        """
        prompt_tokens = self.estimate_tokens(prompt)
        max_completion = max_completion_tokens or (
            self.model_config["max_tokens"] - prompt_tokens)
        total_tokens = prompt_tokens + max_completion

        metrics = {
            "prompt_tokens": prompt_tokens,
            "max_completion_tokens": max_completion,
            "total_tokens": total_tokens,
            "model_limit": self.model_config["max_tokens"]
        }

        if total_tokens > self.model_config["max_tokens"]:
            message = f"Total tokens ({total_tokens}) exceeds model limit ({self.model_config['max_tokens']})"
            log_error(message)
            return False, metrics, message

        log_info("Request validated successfully")
        return True, metrics, "Request validated successfully"

    def get_model_limits(self) -> Dict[str, int]:
        """
        Get the token limits for the current model.

        Returns:
            Dict[str, int]: Dictionary containing model token limits
        """
        return {
            "max_tokens": self.model_config["max_tokens"],
            "max_prompt_tokens": self.model_config["max_tokens"] // 2,
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
        usage = self._calculate_usage(
            prompt_tokens, estimated_completion_tokens, cached)
        return usage.estimated_cost

    def reset_cache(self) -> None:
        """Reset the token estimation cache."""
        self.estimate_tokens.cache_clear()
        log_debug("Token estimation cache cleared")


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
```