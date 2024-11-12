# api_client.py

import os
import time
from typing import Optional

from dotenv import load_dotenv

from core.logger import LoggerSetup
from openai import AzureOpenAI, OpenAI
from anthropic import Anthropic

# Initialize logger
logger = LoggerSetup.get_logger("api_client")

# Load environment variables
load_dotenv()


class APIClient:
    """Unified API client for multiple LLM providers."""

    def __init__(self):
        # Initialize API clients
        logger.info("Initializing API clients")
        self.azure_client = self._init_azure_client()
        self.openai_client = self._init_openai_client()
        self.anthropic_client = self._init_anthropic_client()

        # Configuration
        self.azure_deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4")
        self.openai_model = "gpt-4-turbo-preview"  # Updated to latest model
        self.claude_model = "claude-3-opus-20240229"

    def _init_azure_client(self) -> Optional[AzureOpenAI]:
        """Initialize Azure OpenAI client with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if os.getenv("AZURE_OPENAI_API_KEY"):
                    logger.debug("Initializing Azure OpenAI client")
                    return AzureOpenAI(
                        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://api.azure.com"),
                        api_version=os.getenv("AZURE_API_VERSION", "2024-02-15-preview"),
                        azure_deployment=os.getenv("DEPLOYMENT_NAME", "gpt-4"),
                        azure_ad_token=os.getenv("AZURE_AD_TOKEN"),
                        azure_ad_token_provider=None  # Add token provider if needed
                    )
                logger.warning("Azure OpenAI API key not found")
                return None
            except Exception as e:
                logger.error(f"Error initializing Azure client: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff

    def _init_openai_client(self) -> Optional[OpenAI]:
        """Initialize OpenAI client."""
        try:
            if os.getenv("OPENAI_API_KEY"):
                logger.debug("Initializing OpenAI client")
                return OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_API_BASE"),  # Updated from api_base
                    timeout=60.0,
                    max_retries=3
                )
            logger.warning("OpenAI API key not found")
            return None
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            return None

    def _init_anthropic_client(self) -> Optional[Anthropic]:
        """Initialize Anthropic client."""
        try:
            if os.getenv("ANTHROPIC_API_KEY"):
                logger.debug("Initializing Anthropic client")
                return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            logger.warning("Anthropic API key not found")
            return None
        except Exception as e:
            logger.error(f"Error initializing Anthropic client: {e}")
            return None