import os
from dataclasses import dataclass, field
from typing import Dict, Any
from core.logger import LoggerSetup

# Configure logging
logger = LoggerSetup.get_logger(__name__)

@dataclass
class AzureOpenAIConfig:
    """Configuration settings for Azure OpenAI service."""

    # Base configuration
    model_type: str = field(default="azure")
    max_tokens: int = field(default=1000)
    temperature: float = field(default=0.7)
    request_timeout: int = field(default=30)
    max_retries: int = field(default=3)
    retry_delay: int = field(default=2)
    cache_enabled: bool = field(default=False)

    # Azure-specific configuration
    endpoint: str = field(default="")
    api_key: str = field(default="")
    api_version: str = field(default="2024-02-15-preview")
    deployment_name: str = field(default="")
    model_name: str = field(default="gpt-4")

    # Additional Azure-specific parameters
    max_tokens_per_minute: int = field(default=150000)
    token_buffer: int = field(default=100)
    batch_size: int = field(default=5)

    # Logging configuration
    log_level: str = field(default="DEBUG")
    log_format: str = field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_directory: str = field(default="logs")

    # Redis connection parameters
    redis_host: str = field(
        default_factory=lambda: os.getenv("REDIS_HOST", "localhost")
    )
    redis_port: int = field(
        default_factory=lambda: int(os.getenv("REDIS_PORT", "6379"))
    )
    redis_db: int = field(
        default_factory=lambda: int(os.getenv("REDIS_DB", "0"))
    )
    redis_password: str = field(
        default_factory=lambda: os.getenv("REDIS_PASSWORD", "")
    )

    # Model limits and pricing (if applicable)
    model_limits: Dict[str, Any] = field(default_factory=lambda: {
        "gpt-4": {
            "max_tokens": 8192,
            "cost_per_1k_prompt": 0.03,
            "cost_per_1k_completion": 0.06,
            "chunk_size": 6144
        },
        # Add other models as needed
    })

    @classmethod
    def from_env(cls) -> "AzureOpenAIConfig":
        """Create an instance of AzureOpenAIConfig from environment variables."""
        try:
            config = cls(
                model_type="azure",
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_key=os.getenv("AZURE_OPENAI_KEY", ""),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                model_name=os.getenv("MODEL_NAME", "gpt-4"),
                max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
                temperature=float(os.getenv("TEMPERATURE", "0.7")),
                request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
                max_retries=int(os.getenv("MAX_RETRIES", "3")),
                retry_delay=int(os.getenv("RETRY_DELAY", "2")),
                cache_enabled=os.getenv("CACHE_ENABLED", "False").lower() in ("true", "1"),
                max_tokens_per_minute=int(os.getenv("MAX_TOKENS_PER_MINUTE", "150000")),
                token_buffer=int(os.getenv("TOKEN_BUFFER", "100")),
                batch_size=int(os.getenv("BATCH_SIZE", "5")),
                log_level=os.getenv("LOG_LEVEL", "DEBUG"),
                log_format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                log_directory=os.getenv("LOG_DIRECTORY", "logs"),
                redis_host=os.getenv("REDIS_HOST", "localhost"),
                redis_port=int(os.getenv("REDIS_PORT", "6379")),
                redis_db=int(os.getenv("REDIS_DB", "0")),
                redis_password=os.getenv("REDIS_PASSWORD", "")
            )

            # Validate configuration
            if not config.validate():
                raise ValueError("Invalid Azure OpenAI configuration")

            logger.debug("Successfully loaded Azure OpenAI configuration")
            return config

        except Exception as e:
            logger.error(f"Error creating Azure OpenAI configuration: {e}")
            raise

    def validate(self) -> bool:
        """Validate the Azure OpenAI configuration settings."""
        if not self.model_type:
            logger.error("Model type is required")
            return False

        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            logger.error("Invalid max_tokens value")
            return False

        if not isinstance(self.temperature, float) or not (0 <= self.temperature <= 1):
            logger.error("Invalid temperature value")
            return False

        if not self.endpoint:
            logger.error("Missing Azure OpenAI endpoint")
            return False

        if not self.api_key:
            logger.error("Missing Azure OpenAI API key")
            return False

        if not self.deployment_name:
            logger.error("Missing Azure OpenAI deployment name")
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration settings to a dictionary."""
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
            "redis_password": self.redis_password,
            "model_limits": self.model_limits
        }