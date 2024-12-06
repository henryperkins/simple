import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from core.cache import Cache
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)


def get_env_int(var_name: str, default: int) -> int:
    value = os.getenv(var_name)
    try:
        return int(value) if value is not None else default
    except ValueError:
        logger.error(
            "Environment variable %s must be an integer. Using default %s",
            var_name, default
        )
        return default


def get_env_float(var_name: str, default: float) -> float:
    """Retrieve and convert an environment variable to float type.

    Args:
        var_name (str): The name of the environment variable to retrieve.
        default (float): The default value to return if the environment variable is not set
            or cannot be converted to float.

    Returns:
        float: The environment variable value converted to float if successful,
        otherwise returns the default value.

    Example:
        >>> get_env_float("TIMEOUT", 5.0)
        5.0
        >>> # With environment variable TIMEOUT=10.5
        >>> get_env_float("TIMEOUT", 5.0)
        10.5
    """
    value = os.getenv(var_name)
    try:
        return float(value) if value is not None else default
    except ValueError:
        logger.error(
            "Environment variable %s must be a float. Using default %s",
            var_name, default
        )
        return default


def get_env_bool(var_name: str, default: bool) -> bool:
    """Get boolean value from environment variable.

    Retrieves a boolean value from an environment variable, with a default fallback.
    The function considers 'true', '1', and 'yes' (case-insensitive) as True values.

    Args:
        var_name (str): Name of the environment variable to retrieve
        default (bool): Default value if environment variable is not set

    Returns:
        bool: The boolean value from environment variable or default if not set

    Example:
        >>> os.environ['DEBUG'] = 'true'
        >>> get_env_bool('DEBUG', False)
        True
        >>> get_env_bool('NONEXISTENT', False)
        False
    """
    value = os.getenv(var_name, str(default)).lower()
    return value in ("true", "1", "yes")


@dataclass
class OpenAIModelConfig:
    """Configuration for OpenAI model limits and costs."""
    max_tokens: int = 8192
    cost_per_1k_prompt: float = 0.03
    cost_per_1k_completion: float = 0.06
    chunk_size: int = 6144


@dataclass
class AzureOpenAIConfig:
    """Configuration settings for Azure OpenAI service."""

    # Model settings
    model_type: str = "gpt"
    endpoint: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", "https://openai-hp.openai.azure.com"))
    api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"))
    deployment_id: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", "gpt-4op-deployment"))
    model_name: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o-2024-08-06"))

    # Token settings
    max_tokens: int = field(default_factory=lambda: get_env_int("MAX_TOKENS", 8192))
    temperature: float = field(default_factory=lambda: get_env_float("TEMPERATURE", 0.7))
    request_timeout: int = field(default_factory=lambda: get_env_int("REQUEST_TIMEOUT", 30))
    max_retries: int = field(default_factory=lambda: get_env_int("MAX_RETRIES", 3))
    retry_delay: int = field(default_factory=lambda: get_env_int("RETRY_DELAY", 5))

    # Cache settings
    cache_enabled: bool = field(default_factory=lambda: get_env_bool("CACHE_ENABLED", False))
    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: get_env_int("REDIS_PORT", 6379))
    redis_db: int = field(default_factory=lambda: get_env_int("REDIS_DB", 0))
    redis_password: Optional[str] = field(
        default_factory=lambda: os.getenv("REDIS_PASSWORD", None)
    )
    cache_ttl: int = field(default_factory=lambda: get_env_int("CACHE_TTL", 3600))

    # API Key
    api_key: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_KEY", "40c4befe4179411999c14239f386e24d"))

    # Logging settings
    output_directory: str = field(default_factory=lambda: os.getenv("OUTPUT_DIRECTORY", "output"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    log_directory: str = field(default_factory=lambda: os.getenv("LOG_DIRECTORY", "logs"))

    # Model Limits
    model_limits: Dict[str, OpenAIModelConfig] = field(default_factory=lambda: {
        "gpt-4": OpenAIModelConfig(),
        "gpt-3.5-turbo": OpenAIModelConfig(max_tokens=4096, cost_per_1k_prompt=0.02, cost_per_1k_completion=0.04)
    })

    def __post_init__(self):
        """Initialize cache if enabled."""
        self.cache: Optional[Cache] = None
        if self.cache_enabled:
            try:
                self.cache = Cache(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=self.redis_db,
                    password=self.redis_password,
                    enabled=True,
                    ttl=self.cache_ttl
                )
                logger.info("Cache initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize cache: {e}")
                self.cache = None

    @classmethod
    def from_env(cls) -> "AzureOpenAIConfig":
        """
        Create configuration from environment variables.

        Returns:
            AzureOpenAIConfig instance

        Raises:
            ValueError: If configuration is invalid
        """
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
        """Validate required configuration settings."""
        required_fields = ['endpoint', 'api_key', 'deployment_id']
        missing = [field for field in required_fields if not getattr(self, field, None)]
        if missing:
            logger.error(f"Missing required configuration fields: {missing}")
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration settings to a dictionary excluding sensitive info."""
        config_dict = {
            "model_type": self.model_type,
            "endpoint": self.endpoint,
            "api_version": self.api_version,
            "deployment_id": self.deployment_id,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "cache_enabled": self.cache_enabled,
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "redis_db": self.redis_db,
            "cache_ttl": self.cache_ttl,
            "model_limits": self.model_limits,
            "output_directory": self.output_directory,
            "log_level": self.log_level,
            "log_format": self.log_format,
            "log_directory": self.log_directory,
            # Exclude sensitive information
        }
        return config_dict


def check_required_env_vars() -> None:
    """
    Verify required environment variables are set.

    Raises:
        ValueError: If required variables are missing
    """
    required_vars = {
        "AZURE_OPENAI_ENDPOINT": "Your Azure OpenAI endpoint URL",
        "AZURE_OPENAI_KEY": "Your Azure OpenAI API key",
        "AZURE_OPENAI_DEPLOYMENT_ID": "Your deployment ID"
    }

    missing = [
        var
        for var, description in required_vars.items()
        if not os.getenv(var)
    ]

    if missing:
        raise ValueError(
            "Missing required environment variables:\n" +
            "\n".join(f"- {var} ({required_vars[var]})" for var in missing)
        )