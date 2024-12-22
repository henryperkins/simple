"""Configuration module for AI documentation service."""

import os
import logging  # Move import to top
from dataclasses import dataclass, field
from typing import Any, Optional
from dotenv import load_dotenv
import uuid
from pathlib import Path
import jsonschema
from core.console import print_error
from core.logger import LoggerSetup
from core.exceptions import ConfigurationError
from utils import log_and_raise_error

# Load environment variables
load_dotenv()

# Define base paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPOS_DIR = ROOT_DIR / "repos"
DOCS_OUTPUT_DIR = ROOT_DIR / "docs_output"


def get_env_var(
    key: str,
    default: Any = None,
    var_type: type = str,
    required: bool = False,
    validation_schema: Optional[dict] = None,
) -> Any:
    """Get environment variable with type conversion and validation.

    Args:
        key: Environment variable key
        default: Default value if not found
        var_type: Type to convert the value to
        required: Whether the variable is required
        validation_schema: Optional JSON schema for value validation

    Returns:
        The environment variable value converted to the specified type

    Raises:
        ValueError: If a required variable is missing or type conversion fails
    """
    logger = LoggerSetup.get_logger(__name__)
    value = os.getenv(key)

    if value is None:
        if required:
            log_and_raise_error(
                logger,
                ValueError(f"Required environment variable {key} is not set"),
                ConfigurationError,
                "Missing required environment variable",
                key=key,
            )
        return default

    try:
        if var_type == bool:
            converted_value = value.lower() in ("true", "1", "yes", "on")
        else:
            converted_value = var_type(value)

        if validation_schema:
            jsonschema.validate(instance=converted_value, schema=validation_schema)
        return converted_value
    except (ValueError, TypeError, jsonschema.ValidationError) as e:
        log_and_raise_error(
            logger,
            e,
            ConfigurationError,
            "Failed to convert or validate environment variable",
            key=key,
            value=value,
            var_type=var_type.__name__,
        )
        return default


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

    def __post_init__(self):
        self.logger = LoggerSetup.get_logger(__name__)
        if not self.azure_deployment_name:
            self.logger.warning(
                "⚠️ Warning: The 'AZURE_DEPLOYMENT_NAME' environment variable is not set."
            )
        import logging
        LoggerSetup.log_once(
            self.logger, logging.INFO, "AIConfig initialized successfully"
        )

    @staticmethod
    def from_env() -> "AIConfig":
        """Create configuration from environment variables with Azure defaults."""
        try:
            deployment_name = get_env_var("AZURE_DEPLOYMENT_NAME", required=True)
            config = AIConfig(
                api_key=get_env_var("AZURE_OPENAI_KEY", required=True),
                endpoint=get_env_var("AZURE_OPENAI_ENDPOINT", required=True),
                deployment=deployment_name,
                model=get_env_var(
                    "AZURE_OPENAI_MODEL",
                    "gpt-4o",
                    validation_schema={
                        "type": "string",
                        "enum": ["gpt-4o", "gpt-3.5-turbo", "gpt-4o-2024-11-20"],
                    },
                ),
                azure_api_version=get_env_var(
                    "AZURE_API_VERSION", "2024-10-01-preview"
                ),
            )
            config.azure_deployment_name = deployment_name
            config.max_tokens = get_env_var(
                "AZURE_MAX_TOKENS",
                128000,
                int,
                validation_schema={"type": "integer", "minimum": 1000},
            )
            config.temperature = get_env_var(
                "TEMPERATURE",
                0.7,
                float,
                validation_schema={"type": "number", "minimum": 0, "maximum": 1},
            )
            config.timeout = get_env_var(
                "TIMEOUT",
                30,
                int,
                validation_schema={"type": "integer", "minimum": 10},
            )
            config.api_call_semaphore_limit = get_env_var(
                "API_CALL_SEMAPHORE_LIMIT",
                10,
                int,
                validation_schema={"type": "integer", "minimum": 1},
            )
            config.api_call_max_retries = get_env_var(
                "API_CALL_MAX_RETRIES",
                3,
                int,
                validation_schema={"type": "integer", "minimum": 1},
            )
            config.azure_api_base = get_env_var("AZURE_API_BASE", "")
            config.max_completion_tokens = get_env_var(
                "AZURE_MAX_COMPLETION_TOKENS",
                None,
                int,
                False,
                validation_schema={"type": "integer", "minimum": 100},
            )
            config.truncation_strategy = get_env_var(
                "TRUNCATION_STRATEGY", None, dict, False
            )
            config.tool_choice = get_env_var("TOOL_CHOICE", None, str, False)
            config.parallel_tool_calls = get_env_var(
                "PARALLEL_TOOL_CALLS", True, bool, False
            )
            config.response_format = get_env_var("RESPONSE_FORMAT", None, dict, False)
            config.stream_options = get_env_var("STREAM_OPTIONS", None, dict, False)

            LoggerSetup.log_once(
                config.logger, logging.INFO, "AIConfig initialized successfully"
            )
            return config
        except Exception as e:
            print_error(f"Failed to initialize AIConfig: {e}")
            raise

@dataclass
class AppConfig:
    """Application configuration."""

    debug: bool = False
    log_level: str = "INFO"
    verbose: bool = False
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
            log_level=get_env_var(
                "LOG_LEVEL",
                "INFO",
                validation_schema={
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                },
            ),
            repos_dir=Path(get_env_var("REPOS_DIR", str(REPOS_DIR))),
            docs_output_dir=Path(get_env_var("DOCS_OUTPUT_DIR", str(DOCS_OUTPUT_DIR))),
            log_dir=Path(get_env_var("LOG_DIR", "logs")),
            use_cache=get_env_var("USE_CACHE", False, bool),
            cache_ttl=get_env_var(
                "CACHE_TTL",
                3600,
                int,
                validation_schema={"type": "integer", "minimum": 0},
            ),
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
        try:
            self.ai = AIConfig.from_env()
        except Exception as e:
            print_error(f"Error initializing AIConfig: {e}")
            self.ai = AIConfig(
                api_key="",
                endpoint="",
                deployment="",
                model="",
                azure_api_version="",
                max_tokens=0,
                temperature=0.0,
                timeout=0,
                api_call_semaphore_limit=0,
                api_call_max_retries=0,
            )  # Provide a fallback AIConfig instance to avoid NoneType errors
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
