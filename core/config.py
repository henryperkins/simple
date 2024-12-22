"""Configuration module for AI documentation service."""
# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

from dataclasses import dataclass
from typing import Any, Optional, Union
from dotenv import load_dotenv
import uuid
from pathlib import Path
import jsonschema
from core.console import print_error
from core.logger import LoggerSetup
from core.exceptions import ConfigurationError
from utils import log_and_raise_error

# Base paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPOS_DIR = ROOT_DIR / "repos"
DOCS_OUTPUT_DIR = ROOT_DIR / "docs_output"

# Helper functions
def get_env_var(
    key: str,
    default: Any = None,
    var_type: type = str,
    required: bool = False,
    validation_schema: Optional[dict] = None,
) -> Any:
    """Get environment variable with type conversion and validation."""
    logger = LoggerSetup.get_logger(__name__)
    value = os.getenv(key)

    if value is None:
        if required:
            log_and_raise_error(
                logger,
                ValueError(f"Required environment variable {key} is not set"),
                ConfigurationError,
                f"Missing required environment variable: {key}",
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
            f"Failed to validate {key}: {str(e)}",
        )

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    max_tokens: int
    chunk_size: int
    cost_per_token: float
    rate_limit: int = 10000

@dataclass
class AIConfig:
    """Azure OpenAI service configuration."""
    api_key: str
    endpoint: str
    deployment: str
    azure_api_base: str
    azure_deployment_name: str
    model: str = "gpt-4"
    azure_api_version: str = "2023-05-15"
    max_tokens: int = 8000
    temperature: float = 0.7
    timeout: int = 30
    api_call_semaphore_limit: int = 10
    api_call_max_retries: int = 3
    max_completion_tokens: Optional[int] = None
    truncation_strategy: Optional[dict[str, Any]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = True
    response_format: Optional[dict[str, str]] = None
    stream_options: Optional[dict[str, bool]] = None

    @staticmethod
    def from_env() -> "AIConfig":
        """Create configuration from environment variables."""
        try:
            # Get deployment name and API base first
            deployment_name = get_env_var(
                "AZURE_DEPLOYMENT_NAME",
                required=True,
                validation_schema={
                    "type": "string",
                    "minLength": 1,
                    "pattern": "^[a-zA-Z0-9-]+$"
                }
            )

            azure_api_base = get_env_var(
                "AZURE_API_BASE",
                required=True,
                validation_schema={
                    "type": "string",
                    "minLength": 1
                }
            )

            # Create config object
            config = AIConfig(
                api_key=get_env_var("AZURE_OPENAI_KEY", required=True),
                endpoint=get_env_var("AZURE_OPENAI_ENDPOINT", required=True),
                deployment=deployment_name,
                azure_api_base=azure_api_base,
                azure_deployment_name=deployment_name,
                model=get_env_var("MODEL_NAME", "gpt-4"),
                max_tokens=get_env_var("MAX_TOKENS", 8000, int),
                temperature=get_env_var("TEMPERATURE", 0.7, float),
                azure_api_version=get_env_var("AZURE_API_VERSION", "2023-05-15")
            )

            return config
        except Exception as e:
            print_error(f"Error initializing AIConfig: {str(e)}")
            raise ConfigurationError(f"Failed to initialize AIConfig: {str(e)}") from e


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

    def ensure_directories(self):
        """Ensure all required directories exist."""
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        self.docs_output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def from_env() -> "AppConfig":
        """Create configuration from environment variables."""
        return AppConfig(
            debug=get_env_var("DEBUG", False, bool),
            log_level=get_env_var("LOG_LEVEL", "INFO"),
            repos_dir=Path(get_env_var("REPOS_DIR", str(REPOS_DIR))),
            docs_output_dir=Path(get_env_var("DOCS_OUTPUT_DIR", str(DOCS_OUTPUT_DIR))),
            log_dir=Path(get_env_var("LOG_DIR", str(ROOT_DIR / "logs"))),
            use_cache=get_env_var("USE_CACHE", False, bool),
            cache_ttl=get_env_var("CACHE_TTL", 3600, int)
        )

class Config:
    """Main configuration class combining all config sections."""
    def __init__(self):
        """Initialize configuration from environment."""
        try:
            # Environment variables are already loaded at module level with load_dotenv()

            # Initialize AI config first
            self.ai = AIConfig.from_env()
            
            # Initialize app config
            self.app = AppConfig.from_env()
            self.app.ensure_directories()
            
            # Set correlation ID and project root
            self.correlation_id = str(uuid.uuid4())
            self.project_root = Path.cwd()

        except Exception as e:
            print_error(f"Error initializing config: {str(e)}")
            raise ConfigurationError(f"Failed to initialize configuration: {str(e)}") from e

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
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
                "max_completion_tokens": self.ai.max_completion_tokens,
                "truncation_strategy": self.ai.truncation_strategy,
                "tool_choice": self.ai.tool_choice,
                "parallel_tool_calls": self.ai.parallel_tool_calls,
                "response_format": self.ai.response_format,
                "stream_options": self.ai.stream_options,
                "azure_api_base": self.ai.azure_api_base,
                "azure_deployment_name": self.ai.azure_deployment_name,
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

# Instantiate the global configuration if needed
# config = Config()