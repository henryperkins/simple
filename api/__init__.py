"""
API package for Azure OpenAI service integration.

Provides:
- Token management and usage tracking
- API request handling
- Rate limiting and throttling
- Error handling and retries
"""

from typing import Optional
from core.logger import LoggerSetup
from .token_management import TokenManager
from core.types import TokenUsage

logger = LoggerSetup.get_logger(__name__)


def initialize_api(api_key: Optional[str] = None) -> None:
    """Initialize API with optional configuration."""
    try:
        TokenManager.configure(api_key)
        logger.info("API initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}", exc_info=True)
        raise

__version__ = "1.0.0"
__all__ = [
    "TokenManager",
    "TokenUsage",
    "initialize_api"
]
