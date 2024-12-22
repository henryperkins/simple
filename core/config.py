# config.py
"""Configuration module for AI documentation service."""

import os
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path
import uuid

from core.logger import LoggerSetup
from core.logging_utils import get_logger  # Import from new module
# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
REPOS_DIR = ROOT_DIR / "repos"
DOCS_OUTPUT_DIR = ROOT_DIR / "docs_output"

logger = get_logger(__name__)

# Helper functions
def get_env_var(
    key: str,
    default: Any = None,
    var_type: type = str,
    required: bool = False,
) -> Any:
    """Get environment variable with type conversion and validation."""
    value = os.getenv(key)
    if value is None:
        if required:
            logger.error(f"Required environment variable {key} is not set.")
            raise ValueError(f"Required environment variable {key} is not set")
        return default

    try:
        if var_type == bool:
            return value.lower() in ("true", "1", "yes", "on")
        return var_type(value)
    except ValueError as e:
        logger.error(f"Error converting environment variable {key} to type {var_type.__name__}: {e}")
        raise

@dataclass
class AIConfig:
    """Azure OpenAI service configuration."""
    api_key: str
    endpoint: str
    deployment_name: str
    api_version: str = "2023-12-01-preview"
    max_tokens: int = 8192  
    temperature: float = 0.7
    timeout: int = 30
    api_call_semaphore_limit: int = 10
    api_call_max_retries: int = 3
    max_completion_tokens: int = 1000  

    @staticmethod
    def from_env() -> "AIConfig":
        """Create configuration from environment variables."""
        config = AIConfig(
            api_key=get_env_var("AZURE_OPENAI_KEY", required=True),
            endpoint=get_env_var("AZURE_OPENAI_ENDPOINT", required=True),
            deployment_name=get_env_var("AZURE_DEPLOYMENT_NAME", required=True),
            max_tokens=get_env_var("AZURE_MAX_TOKENS", 8192, int),
            temperature=get_env_var("TEMPERATURE", 0.7, float),
            timeout=get_env_var("TIMEOUT", 30, int),
            api_call_semaphore_limit=get_env_var("API_CALL_SEMAPHORE_LIMIT", 10, int),
            api_call_max_retries=get_env_var("API_CALL_MAX_RETRIES", 3, int),
            max_completion_tokens=get_env_var("AZURE_MAX_COMPLETION_TOKENS", 1000, int),
        )
        # Check that the environment variables were loaded correctly.
        print("--- Loaded AIConfig ---")
        print(f"  api_key: {'*' * (len(config.api_key)-4)}{config.api_key[-4:]}") # Only display last 4 digits of the key
        print(f"  endpoint: {config.endpoint}")
        print(f"  deployment_name: {config.deployment_name}")
        print(f"  api_version: {config.api_version}")
        print("-----------------------")
        return config

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
            verbose=get_env_var("VERBOSE", False, bool),
            repos_dir=Path(get_env_var("REPOS_DIR", str(REPOS_DIR))),
            docs_output_dir=Path(get_env_var("DOCS_OUTPUT_DIR", str(DOCS_OUTPUT_DIR))),
            log_dir=Path(get_env_var("LOG_DIR", str(ROOT_DIR / "logs"))),
            use_cache=get_env_var("USE_CACHE", False, bool),
            cache_ttl=get_env_var("CACHE_TTL", 3600, int),
        )

class Config:
    """Main configuration class combining all config sections."""
    def __init__(self):
        """Initialize configuration from environment."""
        self.ai = AIConfig.from_env()
        self.app = AppConfig.from_env()
        self.app.ensure_directories()
        self.correlation_id = str(uuid.uuid4())
        self.project_root = ROOT_DIR

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "ai": {
                "api_key": "********",  # Redact API key for security
                "endpoint": self.ai.endpoint,
                "deployment_name": self.ai.deployment_name,
                "api_version": self.ai.api_version,
                "max_tokens": self.ai.max_tokens,
                "temperature": self.ai.temperature,
                "timeout": self.ai.timeout,
                "api_call_semaphore_limit": self.ai.api_call_semaphore_limit,
                "api_call_max_retries": self.ai.api_call_max_retries,
                "max_completion_tokens": self.ai.max_completion_tokens
            },
            "app": {
                "debug": self.app.debug,
                "log_level": self.app.log_level,
                "verbose": self.app.verbose,
                "repos_dir": str(self.app.repos_dir),
                "docs_output_dir": str(self.app.docs_output_dir),
                "log_dir": str(self.app.log_dir),
                "use_cache": self.app.use_cache,
                "cache_ttl": self.app.cache_ttl,
            },
            "correlation_id": self.correlation_id,
            "project_root": str(self.project_root)
        }