import os  
from dataclasses import dataclass, field  
from typing import Dict, Any  
from core.logger import LoggerSetup  
import urllib.parse  # For URL validation  

# Configure logging  
logger = LoggerSetup.get_logger(__name__)  

def get_env_int(var_name: str, default: int) -> int:  
    """Get an integer environment variable or return the default."""  
    value = os.getenv(var_name, None)  
    try:  
        return int(value) if value is not None else default  
    except ValueError:  
        logger.error("Environment variable %s must be an integer. Using default %s.", var_name, default)  
        return default  
  
def get_env_float(var_name: str, default: float) -> float:  
    """Get a float environment variable or return the default."""  
    value = os.getenv(var_name, None)  
    try:  
        return float(value) if value is not None else default  
    except ValueError:  
        logger.error("Environment variable %s must be a float. Using default %s.", var_name, default)  
        return default  
  
def get_env_bool(var_name: str, default: bool) -> bool:  
    """Get a boolean environment variable or return the default."""  
    value = os.getenv(var_name, str(default)).lower()  
    return value in ("true", "1", "yes")  
  
@dataclass(frozen=True)  
class AzureOpenAIConfig:  
    """Configuration settings for Azure OpenAI service."""  
  
    # Base configuration  
    model_type: str = field(default_factory=lambda: os.getenv("MODEL_TYPE", "azure"))  
    max_tokens: int = field(default_factory=lambda: get_env_int("MAX_TOKENS", 1000))  
    temperature: float = field(default_factory=lambda: get_env_float("TEMPERATURE", 0.7))  
    request_timeout: int = field(default_factory=lambda: get_env_int("REQUEST_TIMEOUT", 30))  
    max_retries: int = field(default_factory=lambda: get_env_int("MAX_RETRIES", 3))  
    retry_delay: int = field(default_factory=lambda: get_env_int("RETRY_DELAY", 2))  
    cache_enabled: bool = field(default_factory=lambda: get_env_bool("CACHE_ENABLED", False))  
  
    # Azure-specific configuration  
    endpoint: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))  
    api_key: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_KEY", ""))  
    api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"))  
    deployment_name: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT", ""))  
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4"))  
  
    # Additional Azure-specific parameters  
    max_tokens_per_minute: int = field(default_factory=lambda: get_env_int("MAX_TOKENS_PER_MINUTE", 150000))  
    token_buffer: int = field(default_factory=lambda: get_env_int("TOKEN_BUFFER", 100))  
    batch_size: int = field(default_factory=lambda: get_env_int("BATCH_SIZE", 5))  
  
    # Logging configuration  
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "DEBUG"))  
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))  
    log_directory: str = field(default_factory=lambda: os.getenv("LOG_DIRECTORY", "logs"))  
  
    # Redis connection parameters  
    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))  
    redis_port: int = field(default_factory=lambda: get_env_int("REDIS_PORT", 6379))  
    redis_db: int = field(default_factory=lambda: get_env_int("REDIS_DB", 0))  
    redis_password: str = field(default_factory=lambda: os.getenv("REDIS_PASSWORD", ""))  
  
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
        logger.debug("Loading Azure OpenAI configuration from environment variables")  
        try:  
            config = cls()  
            if not config.validate():  
                raise ValueError("Invalid Azure OpenAI configuration")  
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
        # Validate model_type  
        if not self.model_type:  
            logger.error("Model type is required")  
            return False  
  
        # Validate max_tokens  
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:  
            logger.error("Invalid max_tokens value: %s", self.max_tokens)  
            return False  
  
        # Validate temperature  
        if not isinstance(self.temperature, (int, float)) or not (0 <= self.temperature <= 1):  
            logger.error("Invalid temperature value: %s", self.temperature)  
            return False  
  
        # Validate endpoint  
        if not self.endpoint:  
            logger.error("Azure OpenAI endpoint is required")  
            return False  
        else:  
            parsed_url = urllib.parse.urlparse(self.endpoint)  
            if not all([parsed_url.scheme, parsed_url.netloc]):  
                logger.error("Invalid Azure OpenAI endpoint URL: %s", self.endpoint)  
                return False  
  
        # Validate api_key without logging its value  
        if not self.api_key:  
            logger.error("Azure OpenAI API key is required")  
            return False  
  
        # Validate deployment_name  
        if not self.deployment_name:  
            logger.error("Azure OpenAI deployment name is required")  
            return False  
  
        # Validate redis_port  
        if not isinstance(self.redis_port, int) or self.redis_port <= 0:  
            logger.error("Invalid Redis port value: %s", self.redis_port)  
            return False  
  
        # Add additional validations as needed  
  
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
            # Exclude sensitive fields like 'api_key' and 'redis_password'  
            "model_limits": self.model_limits  
        }  