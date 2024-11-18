"""
Configuration Module for AI Model Integrations

This module centralizes all configuration settings for various AI services.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()


@dataclass
class AIModelConfig:
    """Base configuration for all AI models."""
    model_type: str
    max_tokens: int
    temperature: float
    request_timeout: int
    max_retries: int
    retry_delay: int


@dataclass
class AzureOpenAIConfig(AIModelConfig):
    """Configuration settings for Azure OpenAI."""
    endpoint: str
    api_key: str
    api_version: str
    deployment_name: str
    model_name: str
    cache_enabled: bool = True
    cache_ttl: int = 3600
    max_tokens_per_minute: int = 150000
    token_buffer: int = 100
    docstring_functions: Dict[str, Any] = None

    @classmethod
    def from_env(cls) -> "AzureOpenAIConfig":
        """Create configuration from environment variables."""
        return cls(
            model_type="azure",
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_KEY", ""),
            api_version=os.getenv("AZURE_OPENAI_VERSION", "2024-02-15-preview"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
            model_name=os.getenv("MODEL_NAME", "gpt-4"),
            max_tokens=int(os.getenv("MAX_TOKENS", "4000")),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=int(os.getenv("RETRY_DELAY", "2")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
            max_tokens_per_minute=int(os.getenv("MAX_TOKENS_PER_MINUTE", "150000")),
            token_buffer=int(os.getenv("TOKEN_BUFFER", "100")),
            docstring_functions={}
        )

    def validate(self) -> bool:
        """Validate the configuration settings.

        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        required_fields = [
            self.endpoint, self.api_key, self.api_version, self.deployment_name
        ]
        missing_fields = [field for field in required_fields if not field]
        if missing_fields:
            logging.error(f"Missing required configuration fields: {missing_fields}")
            return False
        return True


@dataclass
class OpenAIConfig(AIModelConfig):
    """Configuration for OpenAI API."""
    api_key: str
    organization_id: Optional[str] = None
    model_name: str = "gpt-4"

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        """Create configuration from environment variables."""
        return cls(
            model_type="openai",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            organization_id=os.getenv("OPENAI_ORG_ID"),
            model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-4"),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            request_timeout=int(os.getenv("OPENAI_TIMEOUT", "30")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
            retry_delay=int(os.getenv("OPENAI_RETRY_DELAY", "2"))
        )


@dataclass
class ClaudeConfig(AIModelConfig):
    """Configuration for Claude API."""
    api_key: str
    model_name: str = "claude-3-opus-20240229"

    @classmethod
    def from_env(cls) -> "ClaudeConfig":
        """Create configuration from environment variables."""
        return cls(
            model_type="claude",
            api_key=os.getenv("CLAUDE_API_KEY", ""),
            model_name=os.getenv("CLAUDE_MODEL_NAME", "claude-3-opus-20240229"),
            max_tokens=int(os.getenv("CLAUDE_MAX_TOKENS", "100000")),
            temperature=float(os.getenv("CLAUDE_TEMPERATURE", "0.7")),
            request_timeout=int(os.getenv("CLAUDE_TIMEOUT", "30")),
            max_retries=int(os.getenv("CLAUDE_MAX_RETRIES", "3")),
            retry_delay=int(os.getenv("CLAUDE_RETRY_DELAY", "2"))
        )


@dataclass
class GeminiConfig(AIModelConfig):
    """Configuration for Google Gemini API."""
    api_key: str
    project_id: Optional[str] = None
    model_name: str = "gemini-pro"

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        """Create configuration from environment variables."""
        return cls(
            model_type="gemini",
            api_key=os.getenv("GOOGLE_API_KEY", ""),
            project_id=os.getenv("GOOGLE_PROJECT_ID"),
            model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-pro"),
            max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "2048")),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
            request_timeout=int(os.getenv("GEMINI_TIMEOUT", "30")),
            max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
            retry_delay=int(os.getenv("GEMINI_RETRY_DELAY", "2"))
        )


# Create default configuration instances
try:
    azure_config = AzureOpenAIConfig.from_env()
    openai_config = OpenAIConfig.from_env()
    claude_config = ClaudeConfig.from_env()
    gemini_config = GeminiConfig.from_env()
except ValueError as err:
    logging.error(f"Failed to create configuration: {err}")
