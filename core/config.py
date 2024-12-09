"""Configuration module for AI documentation service."""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

def get_env_var(key: str, default: Any = None, var_type: type = str, required: bool = False) -> Any:
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
            return value.lower() in ('true', '1', 'yes', 'on')
        return var_type(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to convert {key}={value} to type {var_type.__name__}: {str(e)}")

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
    model_limits: Dict[str, ModelConfig] = field(default_factory=lambda: {
        "gpt-4": ModelConfig(
            max_tokens=8192,
            chunk_size=4096,
            cost_per_token=0.00003
        ),
        "gpt-3.5-turbo": ModelConfig(
            max_tokens=4096,
            chunk_size=2048,
            cost_per_token=0.000002
        )
    })

    @classmethod
    def from_env(cls) -> 'AIConfig':
        """Create configuration from environment variables."""
        return cls(
            api_key=get_env_var("AZURE_OPENAI_KEY", required=True),
            endpoint=get_env_var("AZURE_OPENAI_ENDPOINT", required=True),
            deployment=get_env_var("AZURE_OPENAI_DEPLOYMENT", required=True),
            model=get_env_var("MODEL_NAME", "gpt-4"),
            max_tokens=get_env_var("MAX_TOKENS", 8192, int),
            temperature=get_env_var("TEMPERATURE", 0.7, float),
            timeout=get_env_var("TIMEOUT", 30, int)
        )

@dataclass
class AppConfig:
    """Application configuration."""
    debug: bool = False
    log_level: str = "INFO"
    output_dir: str = "docs"
    use_cache: bool = False
    cache_ttl: int = 3600

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create configuration from environment variables."""
        return cls(
            debug=get_env_var("DEBUG", False, bool),
            log_level=get_env_var("LOG_LEVEL", "INFO"),
            output_dir=get_env_var("OUTPUT_DIR", "docs"),
            use_cache=get_env_var("USE_CACHE", False, bool),
            cache_ttl=get_env_var("CACHE_TTL", 3600, int)
        )

class Config:
    """Main configuration class combining all config sections."""
    def __init__(self):
        """Initialize configuration from environment."""
        self.ai = AIConfig.from_env()
        self.app = AppConfig.from_env()
        self.correlation_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "ai": {
                "endpoint": self.ai.endpoint,
                "deployment": self.ai.deployment,
                "model": self.ai.model,
                "max_tokens": self.ai.max_tokens,
                "temperature": self.ai.temperature,
                "timeout": self.ai.timeout,
                "model_limits": {
                    model: {
                        "max_tokens": config.max_tokens,
                        "chunk_size": config.chunk_size,
                        "cost_per_token": config.cost_per_token
                    }
                    for model, config in self.ai.model_limits.items()
                }
            },
            "app": {
                "debug": self.app.debug,
                "log_level": self.app.log_level,
                "output_dir": self.app.output_dir,
                "use_cache": self.app.use_cache,
                "cache_ttl": self.app.cache_ttl
            },
            "correlation_id": self.correlation_id
        }

# Create global configuration instance
config = Config()
