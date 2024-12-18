"""Configuration module for AI documentation service."""

import os
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv
import uuid
from pathlib import Path


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


@dataclass
class AIConfig:
    """Core AI service configuration."""

    api_key: str
    endpoint: str
    deployment: str
    model: str = "gpt-4"
    max_tokens: int = 8192
    temperature: float = 0.7
    timeout: int = 30
    api_call_semaphore_limit: int = 10 # Added this line
    api_call_max_retries: int = 3 # Added this line
    model_limits: dict[str, ModelConfig] = field(default_factory=lambda: {
        "gpt-4": ModelConfig(max_tokens=8192, chunk_size=4096, cost_per_token=0.00003),
        "gpt-3.5-turbo": ModelConfig(max_tokens=4096, chunk_size=2048, cost_per_token=0.000002),
    })

    @staticmethod
    def from_env() -> "AIConfig":
        """Create configuration from environment variables."""
        return AIConfig(
            api_key=get_env_var("AZURE_OPENAI_KEY", required=True),
            endpoint=get_env_var("AZURE_OPENAI_ENDPOINT", required=True),
            deployment=get_env_var("AZURE_OPENAI_DEPLOYMENT", required=True),
            model=get_env_var("MODEL_NAME", "gpt-4"),
            max_tokens=get_env_var("MAX_TOKENS", 8192, int),
            temperature=get_env_var("TEMPERATURE", 0.7, float),
            timeout=get_env_var("TIMEOUT", 30, int),
            api_call_semaphore_limit=get_env_var("API_CALL_SEMAPHORE_LIMIT", 10, int), # Added this line
            api_call_max_retries=get_env_var("API_CALL_MAX_RETRIES", 3, int) # Added this line
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
            cache_ttl=get_env_var("CACHE_TTL", 3600, int)
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
        self.project_root = Path.cwd()  # Set project_root to the current working directory

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
                "model_limits": {
                    model: {
                        "max_tokens": config.max_tokens,
                        "chunk_size": config.chunk_size,
                        "cost_per_token": config.cost_per_token,
                    }
                    for model, config in self.ai.model_limits.items()
                },
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
            "project_root": str(self.project_root),  # Add project_root to the dictionary
        }


# Create global configuration instance
config = Config()