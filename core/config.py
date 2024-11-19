"""
Configuration Module for AI Model Integrations

This module centralizes all configuration settings for various AI services.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging
from pathlib import Path

# Load environment variables
load_dotenv()

@dataclass
class AIModelConfig:
    """Base configuration for all AI models."""
    model_type: str
    max_tokens: int = field(default=4000)
    temperature: float = field(default=0.7)
    request_timeout: int = field(default=30)
    max_retries: int = field(default=3)
    retry_delay: int = field(default=2)

    def validate(self) -> bool:
        """Validate base configuration settings."""
        try:
            if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
                logging.error("Invalid max_tokens value")
                return False
            if not isinstance(self.temperature, float) or not 0 <= self.temperature <= 1:
                logging.error("Invalid temperature value")
                return False
            if not isinstance(self.request_timeout, int) or self.request_timeout <= 0:
                logging.error("Invalid request_timeout value")
                return False
            return True
        except Exception as e:
            logging.error(f"Configuration validation error: {e}")
            return False

@dataclass
class AzureOpenAIConfig(AIModelConfig):
    """Configuration settings for Azure OpenAI."""
    endpoint: str = field(default="")
    api_key: str = field(default="")
    api_version: str = field(default="2024-02-15-preview")
    deployment_name: str = field(default="")
    model_name: str = field(default="gpt-4")
    cache_enabled: bool = field(default=True)
    cache_ttl: int = field(default=3600)
    max_tokens_per_minute: int = field(default=150000)
    token_buffer: int = field(default=100)
    docstring_functions: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "AzureOpenAIConfig":
        """Create configuration from environment variables."""
        try:
            config = cls(
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
                cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
                cache_ttl=int(os.getenv("CACHE_TTL", "3600"))
            )
            
            if not config.validate():
                raise ValueError("Invalid configuration values")
            return config
            
        except Exception as e:
            logging.error(f"Error creating configuration from environment: {e}")
            raise

    def validate(self) -> bool:
        """Validate Azure OpenAI configuration settings."""
        try:
            if not super().validate():
                return False
                
            if not self.endpoint or not self.api_key or not self.deployment_name:
                logging.error("Missing required Azure OpenAI credentials")
                return False
                
            if not isinstance(self.max_tokens_per_minute, int) or self.max_tokens_per_minute <= 0:
                logging.error("Invalid max_tokens_per_minute value")
                return False
                
            if not isinstance(self.cache_ttl, int) or self.cache_ttl <= 0:
                logging.error("Invalid cache_ttl value")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Azure configuration validation error: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "endpoint": self.endpoint,
            "api_version": self.api_version,
            "deployment_name": self.deployment_name,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
            "max_tokens_per_minute": self.max_tokens_per_minute
        }

@dataclass
class OpenAIConfig(AIModelConfig):
    """Configuration for OpenAI API."""
    api_key: str = field(default="")
    organization_id: Optional[str] = field(default=None)
    model_name: str = field(default="gpt-4")

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        """Create configuration from environment variables."""
        try:
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
        except Exception as e:
            logging.error(f"Error creating OpenAI configuration: {e}")
            raise

    def validate(self) -> bool:
        """Validate OpenAI configuration settings."""
        if not super().validate():
            return False
        return bool(self.api_key)

@dataclass
class ClaudeConfig(AIModelConfig):
    """Configuration for Claude API."""
    api_key: str = field(default="")
    model_name: str = field(default="claude-3-opus-20240229")

    @classmethod
    def from_env(cls) -> "ClaudeConfig":
        """Create configuration from environment variables."""
        try:
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
        except Exception as e:
            logging.error(f"Error creating Claude configuration: {e}")
            raise

    def validate(self) -> bool:
        """Validate Claude configuration settings."""
        if not super().validate():
            return False
        return bool(self.api_key)

@dataclass
class GeminiConfig(AIModelConfig):
    """Configuration for Google Gemini API."""
    api_key: str = field(default="")
    project_id: Optional[str] = field(default=None)
    model_name: str = field(default="gemini-pro")

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        """Create configuration from environment variables."""
        try:
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
        except Exception as e:
            logging.error(f"Error creating Gemini configuration: {e}")
            raise

    def validate(self) -> bool:
        """Validate Gemini configuration settings."""
        if not super().validate():
            return False
        return bool(self.api_key)

def load_config(model_type: str) -> AIModelConfig:
    """
    Load configuration for specified model type.

    Args:
        model_type (str): Type of model to load configuration for
            ('azure', 'openai', 'claude', 'gemini')

    Returns:
        AIModelConfig: Configuration instance for specified model type

    Raises:
        ValueError: If invalid model type specified
    """
    try:
        config_map = {
            'azure': AzureOpenAIConfig.from_env,
            'openai': OpenAIConfig.from_env,
            'claude': ClaudeConfig.from_env,
            'gemini': GeminiConfig.from_env
        }

        if model_type not in config_map:
            raise ValueError(f"Invalid model type: {model_type}")

        config = config_map[model_type]()
        if not config.validate():
            raise ValueError(f"Invalid configuration for {model_type}")

        logging.info(f"Successfully loaded configuration for {model_type}")
        return config

    except Exception as e:
        logging.error(f"Error loading configuration for {model_type}: {e}")
        raise

def get_default_config() -> AIModelConfig:
    """
    Get default configuration based on environment settings.

    Returns:
        AIModelConfig: Default configuration instance
    """
    try:
        default_model = os.getenv("DEFAULT_MODEL", "azure")
        return load_config(default_model)
    except Exception as e:
        logging.error(f"Error loading default configuration: {e}")
        raise

# Create default configuration instances
try:
    azure_config = AzureOpenAIConfig.from_env()
    openai_config = OpenAIConfig.from_env()
    claude_config = ClaudeConfig.from_env()
    gemini_config = GeminiConfig.from_env()
except ValueError as err:
    logging.error(f"Failed to create configuration: {err}")

# Export default configurations
__all__ = [
    'AIModelConfig',
    'AzureOpenAIConfig',
    'OpenAIConfig',
    'ClaudeConfig',
    'GeminiConfig',
    'load_config',
    'get_default_config',
    'azure_config',
    'openai_config',
    'claude_config',
    'gemini_config'
]