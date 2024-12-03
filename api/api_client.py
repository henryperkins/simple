"""
API Client Module

Provides an interface to interact with the Azure OpenAI API, using centralized response parsing.
"""

from typing import Any, Dict, List, Optional, Tuple
from openai import AsyncAzureOpenAI
from core.config import AzureOpenAIConfig
from core.logger import LoggerSetup
from core.response_parsing import ResponseParsingService, ParsedResponse

class APIClient:
    """Client to interact with Azure OpenAI API with centralized response parsing."""

    def __init__(
        self,
        config: Optional[AzureOpenAIConfig] = None,
        response_parser: Optional[ResponseParsingService] = None
    ) -> None:
        """
        Initialize API client with configuration and response parser.

        Args:
            config: Configuration settings for Azure OpenAI service
            response_parser: Optional response parsing service instance
        """
        self.logger = LoggerSetup.get_logger(__name__)
        try:
            self.config = config or AzureOpenAIConfig.from_env()
            self.client = AsyncAzureOpenAI(
                api_key=self.config.api_key,
                api_version=self.config.api_version,
                azure_endpoint=self.config.endpoint
            )
            self.response_parser = response_parser or ResponseParsingService()
            self.logger.info("APIClient initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize APIClient: {e}")
            raise

    async def process_request(
        self,
        prompt: str,
        temperature: float = 0.4,
        max_tokens: int = 6000,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, str]] = None,
        expected_format: str = 'json'
    ) -> Tuple[Optional[ParsedResponse], Optional[Dict[str, Any]]]:
        """
        Process request to Azure OpenAI API with response parsing.

        Args:
            prompt: The prompt to send to the API
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            tools: Optional function calling tools
            tool_choice: Optional tool choice
            expected_format: Expected response format

        Returns:
            Tuple of parsed response and usage statistics
        """
        try:
            self.logger.debug("Processing API request", 
                            extra={"max_tokens": max_tokens, "temperature": temperature})

            completion = await self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice
            )

            # Parse response
            parsed_response = await self.response_parser.parse_response(
                response=completion.choices[0].message.content,
                expected_format=expected_format
            )

            # Get usage statistics
            usage = completion.usage.model_dump() if hasattr(completion, 'usage') else {}

            self.logger.debug("API request successful",
                           extra={"usage": usage, "parsing_success": parsed_response.validation_success})

            return parsed_response, usage

        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            raise

    async def close(self) -> None:
        """Close API client resources."""
        try:
            await self.client.close()
            self.logger.info("APIClient closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing APIClient: {e}")
            raise

    async def __aenter__(self) -> 'APIClient':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()