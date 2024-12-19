"""Configuration module for AI documentation service."""

import os
from dataclasses import dataclass, field
from typing import Any, Optional
from dotenv import load_dotenv
import uuid
from pathlib import Path
from typing import Literal


# Load environment variables
load_dotenv()

# Define base paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPOS_DIR = ROOT_DIR / "repos"
DOCS_OUTPUT_DIR = ROOT_DIR / "docs_output"


def get_env_var(
    key: str, default: Any = None, var_type: type = str, required: bool = False
) -> Any:
    """Get environment variable with type conversion and validation.

    Args:
        key: Environment variable key
        default: Default value if not found
        var_type: Type to convert the value to
        required: Whether the variable is required

    Returns:
        The environment variable value converted to the specified type

    Raises:
        ValueError: If a required variable is missing or type conversion fails
    """
    value = os.getenv(key)

    if value is None:
        if required:
            raise ValueError(f"Required environment variable {key} is not set")
        return default

    try:
        if var_type == bool:
            return value.lower() in ("true", "1", "yes", "on")
        return var_type(value)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Failed to convert {key}={value} to type {var_type.__name__}: {str(e)}"
        )


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    max_tokens: int
    chunk_size: int
    cost_per_token: float
    rate_limit: int = 10000  # Requests per minute


@dataclass
class AIConfig:
    """Azure OpenAI service configuration."""

    api_key: str
    endpoint: str
    deployment: str
    model: str = "gpt-4o"  # Using the general model name
    azure_api_version: str = "2024-10-01-preview"  # Updated API version
    max_tokens: int = 128000
    temperature: float = 0.7
    timeout: int = 30
    api_call_semaphore_limit: int = 10
    api_call_max_retries: int = 3
    max_completion_tokens: Optional[int] = None
    truncation_strategy: Optional[dict[str, Any]] = None
    tool_choice: Optional[str | dict[str, Any]] = None
    parallel_tool_calls: Optional[bool] = True
    response_format: Optional[dict[str, str]] = None
    stream_options: Optional[dict[str, bool]] = None

    # Azure-specific settings
    azure_api_base: str = field(default_factory=lambda: os.getenv("AZURE_API_BASE", ""))
    azure_deployment_name: str = field(
        default_factory=lambda: os.getenv("AZURE_DEPLOYMENT_NAME", "")
    )

    # Model configurations including Azure-specific limits
    model_limits: dict[str, ModelConfig] = field(
        default_factory=lambda: {
            "gpt-4o": ModelConfig(
                max_tokens=128000,
                chunk_size=4096,
                cost_per_token=0.00003,
                rate_limit=10000,
            ),
            "gpt-3.5-turbo": ModelConfig(
                max_tokens=4096,
                chunk_size=2048,
                cost_per_token=0.000002,
                rate_limit=10000,
            ),
            "gpt-4o-2024-11-20": ModelConfig(
                max_tokens=128000,
                chunk_size=4096,
                cost_per_token=0.00003,
                rate_limit=10000,
            ),
        }
    )

    @staticmethod
    def from_env() -> "AIConfig":
        """Create configuration from environment variables with Azure defaults."""
        return AIConfig(
            api_key=get_env_var("AZURE_OPENAI_KEY", required=True),
            endpoint=get_env_var("AZURE_OPENAI_ENDPOINT", required=True),
            deployment=get_env_var(
                "AZURE_OPENAI_DEPLOYMENT",
                required=True,
                default=None,
                var_type=str,
            ),
            model=get_env_var("AZURE_OPENAI_MODEL", "gpt-4o"),
            azure_api_version=get_env_var("AZURE_API_VERSION", "2024-10-01-preview"),
            max_tokens=get_env_var("AZURE_MAX_TOKENS", 128000, int),
            temperature=get_env_var("TEMPERATURE", 0.7, float),
            timeout=get_env_var("TIMEOUT", 30, int),
            api_call_semaphore_limit=get_env_var("API_CALL_SEMAPHORE_LIMIT", 10, int),
            api_call_max_retries=get_env_var("API_CALL_MAX_RETRIES", 3, int),
            azure_api_base=get_env_var("AZURE_API_BASE", ""),
            azure_deployment_name=get_env_var("AZURE_DEPLOYMENT_NAME", ""),
            max_completion_tokens=get_env_var(
                "AZURE_MAX_COMPLETION_TOKENS", None, int, False
            ),
            truncation_strategy=get_env_var("TRUNCATION_STRATEGY", None, dict, False),
            tool_choice=get_env_var("TOOL_CHOICE", None, str, False),
            parallel_tool_calls=get_env_var("PARALLEL_TOOL_CALLS", True, bool, False),
            response_format=get_env_var("RESPONSE_FORMAT", None, dict, False),
            stream_options=get_env_var("STREAM_OPTIONS", None, dict, False),
        )


@dataclass
class AppConfig:
    """Application configuration."""

    debug: bool = False
    log_level: str = "INFO"
    repos_dir: Path = REPOS_DIR
    docs_output_dir: Path = DOCS_OUTPUT_DIR
    log_dir: Path = ROOT_DIR / "logs"
    use_cache: bool = False
    cache_ttl: int = 3600

    @staticmethod
    def from_env() -> "AppConfig":
        """Create configuration from environment variables."""
        return AppConfig(
            debug=get_env_var("DEBUG", False, bool),
            log_level=get_env_var("LOG_LEVEL", "INFO"),
            repos_dir=Path(get_env_var("REPOS_DIR", str(REPOS_DIR))),
            docs_output_dir=Path(get_env_var("DOCS_OUTPUT_DIR", str(DOCS_OUTPUT_DIR))),
            log_dir=Path(get_env_var("LOG_DIR", "logs")),
            use_cache=get_env_var("USE_CACHE", False, bool),
            cache_ttl=get_env_var("CACHE_TTL", 3600, int),
        )

    def ensure_directories(self):
        """Ensure all required directories exist."""
        self.repos_dir.mkdir(exist_ok=True)
        self.docs_output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)


class Config:
    """Main configuration class combining all config sections."""

    def __init__(self):
        """Initialize configuration from environment."""
        self.ai = AIConfig.from_env()
        self.app = AppConfig.from_env()
        self.correlation_id = str(uuid.uuid4())
        self.app.ensure_directories()
        self.project_root = (
            Path.cwd()
        )  # Set project_root to the current working directory

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "ai": {
                "api_key": "[REDACTED]",
                "endpoint": self.ai.endpoint,
                "deployment": self.ai.deployment,
                "model": self.ai.model,
                "max_tokens": self.ai.max_tokens,
                "temperature": self.ai.temperature,
                "timeout": self.ai.timeout,
                "api_call_semaphore_limit": self.ai.api_call_semaphore_limit,
                "api_call_max_retries": self.ai.api_call_max_retries,
                "azure_api_version": self.ai.azure_api_version,
                "model_limits": {
                    model: {
                        "max_tokens": config.max_tokens,
                        "chunk_size": config.chunk_size,
                        "cost_per_token": config.cost_per_token,
                        "rate_limit": config.rate_limit,
                    }
                    for model, config in self.ai.model_limits.items()
                },
                "max_completion_tokens": self.ai.max_completion_tokens,
                "truncation_strategy": self.ai.truncation_strategy,
                "tool_choice": self.ai.tool_choice,
                "parallel_tool_calls": self.ai.parallel_tool_calls,
                "response_format": self.ai.response_format,
                "stream_options": self.ai.stream_options,
            },
            "app": {
                "debug": self.app.debug,
                "log_level": self.app.log_level,
                "repos_dir": str(self.app.repos_dir),
                "docs_output_dir": str(self.app.docs_output_dir),
                "log_dir": str(self.app.log_dir),
                "use_cache": self.app.use_cache,
                "cache_ttl": self.app.cache_ttl,
            },
            "correlation_id": self.correlation_id,
            "project_root": str(self.project_root),
        }


# Create global configuration instance
config = Config()
