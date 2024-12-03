"""
Configuration module for Azure OpenAI service integration.
Handles environment variables, validation, and configuration management.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path
import urllib.parse
from core.logger import LoggerSetup

# Configure logging
logger = LoggerSetup.get_logger(__name__)

def get_env_int(var_name: str, default: int) -> int:
    """Get an integer environment variable or return the default."""
    value = os.getenv(var_name, None)
    try:
        return int(value) if value is not None else default
    except ValueError:
        logger.error(
            "Environment variable %s must be an integer. Using default %s.",
            var_name, default
        )
        return default

def get_env_float(var_name: str, default: float) -> float:
    """Get a float environment variable or return the default."""
    value = os.getenv(var_name, None)
    try:
        return float(value) if value is not None else default
    except ValueError:
        logger.error(
            "Environment variable %s must be a float. Using default %s.",
            var_name, default
        )
        return default

def get_env_bool(var_name: str, default: bool) -> bool:
    """Get a boolean environment variable or return the default."""
    value = os.getenv(var_name, str(default)).lower()
    return value in ("true", "1", "yes")

def check_required_env_vars() -> None:
    """Verify required environment variables are set."""
    required_vars = {
        "AZURE_OPENAI_ENDPOINT": "Your Azure OpenAI endpoint URL",
        "AZURE_OPENAI_KEY": "Your Azure OpenAI API key",
        "AZURE_OPENAI_DEPLOYMENT": "Your deployment name"
    }

    missing = [
        f"{var} ({description})"
        for var, description in required_vars.items()
        if not os.getenv(var)
    ]

    if missing:
        raise ValueError(
            "Missing required environment variables:\n" +
            "\n".join(f"- {var}" for var in missing)
        )

@dataclass(frozen=True)
class AzureOpenAIConfig:
    """Configuration settings for Azure OpenAI service."""

    model_type: str = field(default_factory=lambda: os.getenv("MODEL_TYPE", "azure"))
    max_tokens: int = field(default_factory=lambda: get_env_int("MAX_TOKENS", 6000))
    temperature: float = field(
        default_factory=lambda: get_env_float("TEMPERATURE", 0.4)
    )
    request_timeout: int = field(default_factory=lambda: get_env_int("REQUEST_TIMEOUT", 30))
    max_retries: int = field(default_factory=lambda: get_env_int("MAX_RETRIES", 3))
    retry_delay: int = field(default_factory=lambda: get_env_int("RETRY_DELAY", 2))
    cache_enabled: bool = field(default_factory=lambda: get_env_bool("CACHE_ENABLED", False))

    endpoint: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    api_key: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_KEY", ""))
    api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-09-01-preview"))
    deployment_name: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT", ""))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o-2024-08-06"))

    max_tokens_per_minute: int = field(default_factory=lambda: get_env_int("MAX_TOKENS_PER_MINUTE", 150000))
    token_buffer: int = field(default_factory=lambda: get_env_int("TOKEN_BUFFER", 100))
    batch_size: int = field(default_factory=lambda: get_env_int("BATCH_SIZE", 5))

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "DEBUG"))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    log_directory: str = field(default_factory=lambda: os.getenv("LOG_DIRECTORY", "logs"))

    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: get_env_int("REDIS_PORT", 6379))
    redis_db: int = field(default_factory=lambda: get_env_int("REDIS_DB", 0))
    redis_password: str = field(default_factory=lambda: os.getenv("REDIS_PASSWORD", ""))

    model_limits: Dict[str, Any] = field(default_factory=lambda: {
        "gpt-4": {
            "max_tokens": 8192,
            "cost_per_1k_prompt": 0.03,
            "cost_per_1k_completion": 0.06,
            "chunk_size": 6144
        }
    })

    output_directory: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", "docs"))

    @classmethod
    def from_env(cls) -> "AzureOpenAIConfig":
        """Create configuration from environment variables."""
        logger.debug("Loading Azure OpenAI configuration from environment variables")
        try:
            check_required_env_vars()
            config = cls()
            if not config.validate():
                raise ValueError("Invalid configuration values.")
            logger.debug("Successfully loaded Azure OpenAI configuration")
            return config
        except ValueError as ve:
            logger.error("Configuration error: %s", ve)
            raise
        except Exception as e:
            logger.error("Unexpected error creating Azure OpenAI configuration: %s", e)
            raise

    def validate(self) -> bool:
        """Validate the Azure OpenAI configuration settings."""
        if not self.model_type:
            logger.error("Model type is required")
            return False

        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            logger.error("Invalid max_tokens value: %s", self.max_tokens)
            return False

        if not isinstance(self.temperature, (int, float)) or not (0 <= self.temperature <= 1):
            logger.error("Invalid temperature value: %s", self.temperature)
            return False

        if not self.endpoint:
            logger.error("Azure OpenAI endpoint is required")
            return False
        else:
            parsed_url = urllib.parse.urlparse(self.endpoint)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                logger.error("Invalid Azure OpenAI endpoint URL: %s", self.endpoint)
                return False

        if not self.api_key:
            logger.error("Azure OpenAI API key is required")
            return False

        if not self.deployment_name:
            logger.error("Azure OpenAI deployment name is required")
            return False

        if not isinstance(self.redis_port, int) or self.redis_port <= 0:
            logger.error("Invalid Redis port value: %s", self.redis_port)
            return False

        try:
            Path(self.output_directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration settings to a dictionary, excluding sensitive information."""
        return {
            "model_type": self.model_type,
            "endpoint": self.endpoint,
            "api_version": self.api_version,
            "deployment_name": self.deployment_name,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "cache_enabled": self.cache_enabled,
            "max_tokens_per_minute": self.max_tokens_per_minute,
            "token_buffer": self.token_buffer,
            "batch_size": self.batch_size,
            "log_level": self.log_level,
            "log_format": self.log_format,
            "log_directory": self.log_directory,
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "redis_db": self.redis_db,
            "model_limits": self.model_limits,
            "output_directory": self.output_directory
        }