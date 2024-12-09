"""
Configuration module for AI documentation service.

Provides essential configuration settings for Azure OpenAI service
interaction and basic application settings.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from utils import get_env_var
from core.logger import LoggerSetup, CorrelationLoggerAdapter
import uuid

# Initialize the logger with a correlation ID
base_logger = LoggerSetup.get_logger(__name__)
correlation_id = str(uuid.uuid4())  # Generate a unique correlation ID
logger = CorrelationLoggerAdapter(base_logger, correlation_id=correlation_id)

@dataclass
class AIConfig:
    """Core AI service configuration."""
    
    # Required settings
    api_key: str
    endpoint: str
    api_version: str
    
    # Optional settings with sensible defaults
    model: str = "gpt-4"
    max_tokens: int = 8192
    temperature: float = 0.7
    timeout: int = 30
    
    @classmethod
    def from_env(cls) -> 'AIConfig':
        """Create configuration from environment variables."""
        try:
            logger.info("Loading AI configuration from environment", extra={'correlation_id': logger.correlation_id})
            config = cls(
                api_key=get_env_var("OPENAI_API_KEY", required=True),
                api_version=get_env_var("OPENAI_API_VERSION", "2020-05-03"),
                endpoint=get_env_var("OPENAI_ENDPOINT", required=True),
                model=get_env_var("OPENAI_MODEL", "gpt-4"),
                max_tokens=get_env_var("MAX_TOKENS", 8192, int),
                temperature=get_env_var("TEMPERATURE", 0.7, float),
                timeout=get_env_var("TIMEOUT", 30, int)
            )
            logger.info("AI configuration loaded successfully", extra={'correlation_id': logger.correlation_id})
            return config
        except ValueError as e:
            logger.error(f"Configuration error: {e}", exc_info=True, extra={'correlation_id': logger.correlation_id})
            raise

@dataclass
class AppConfig:
    """Application configuration."""
    
    # Basic settings
    debug: bool = False
    log_level: str = "INFO"
    output_dir: str = "docs"
    
    # Cache settings (simplified)
    use_cache: bool = False
    cache_ttl: int = 3600  # 1 hour
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create application config from environment variables."""
        logger.info("Loading application configuration from environment", extra={'correlation_id': logger.correlation_id})
        config = cls(
            debug=get_env_var("DEBUG", False, bool),
            log_level=get_env_var("LOG_LEVEL", "INFO"),
            output_dir=get_env_var("OUTPUT_DIR", "docs"),
            use_cache=get_env_var("USE_CACHE", False, bool),
            cache_ttl=get_env_var("CACHE_TTL", 3600, int)
        )
        logger.info("Application configuration loaded successfully", extra={'correlation_id': logger.correlation_id})
        return config

class Config:
    """Main configuration container."""
    
    def __init__(self):
        """Initialize configuration."""
        logger.info("Initializing main configuration", extra={'correlation_id': logger.correlation_id})
        self.ai = AIConfig.from_env()
        self.app = AppConfig.from_env()
        logger.info("Main configuration initialized successfully", extra={'correlation_id': logger.correlation_id})
        
    @property
    def debug(self) -> bool:
        """Get debug mode status."""
        return self.app.debug
    
    @property
    def is_cache_enabled(self) -> bool:
        """Get cache status."""
        return self.app.use_cache
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary, excluding sensitive data."""
        logger.debug("Converting configuration to dictionary", extra={'correlation_id': logger.correlation_id})
        return {
            "ai": {
                "model": self.ai.model,
                "max_tokens": self.ai.max_tokens,
                "temperature": self.ai.temperature,
                "timeout": self.ai.timeout
            },
            "app": {
                "debug": self.app.debug,
                "log_level": self.app.log_level,
                "output_dir": self.app.output_dir,
                "use_cache": self.app.use_cache,
                "cache_ttl": self.app.cache_ttl
            }
        }