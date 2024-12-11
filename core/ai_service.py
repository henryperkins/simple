"""AI service module for interacting with OpenAI API."""

import json
from typing import Dict, Any, List, Optional
import aiohttp
import asyncio
from datetime import datetime
from urllib.parse import urljoin
from pathlib import Path

from core.logger import LoggerSetup
from core.config import AIConfig
from core.cache import Cache
from core.exceptions import ProcessingError
from core.docstring_processor import DocstringProcessor
from core.response_parsing import ResponseParsingService
from core.prompt_manager import PromptManager
from core.types.base import DocumentationContext, ProcessingResult, DocumentationData
from api.token_management import TokenManager
from core.types.base import Injector


class AIService:
    """Service for interacting with OpenAI API."""

    def __init__(
        self,
        config: AIConfig,
        correlation_id: Optional[str] = None
    ) -> None:
        """Initialize AI service with dependency injection.

        Args:
            config: AI service configuration
            correlation_id: Optional correlation ID for tracking related operations
        """
        self.config = config
        self.correlation_id = correlation_id
        self.logger = LoggerSetup.get_logger(__name__)
        self.cache = Cache()
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent API calls
        self._client = None

        # Inject dependencies
        self.docstring_processor = Injector.get('docstring_parser')
        self.response_parser = Injector.get('response_parser')
        self.token_manager = Injector.get('token_manager')
        self.prompt_manager = Injector.get('prompt_manager')

    # ... rest of the class remains unchanged ...
