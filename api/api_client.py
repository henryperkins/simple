"""
API Client Module

Provides an interface to interact with the Azure OpenAI API, handling
requests and responses, including support for function calling and structured outputs.

Usage Example:
    ```python
    from api.api_client import APIClient
    from core.config import AzureOpenAIConfig

    config = AzureOpenAIConfig.from_env()
    client = APIClient(config)

    async def main():
        prompt = "Translate the following English text to French: 'Hello, world!'"
        response, usage = await client.process_request(prompt)
        print(response)

    import asyncio
    asyncio.run(main())
    ```
"""

from typing import Any, Dict, List, Optional, Tuple
from openai import AsyncAzureOpenAI
from core.config import AzureOpenAIConfig
from core.logger import LoggerSetup

class APIClient:
    """
    Client to interact with Azure OpenAI API.

    Attributes:
        config (AzureOpenAIConfig): Configuration settings for Azure OpenAI service.
        client (AsyncAzureOpenAI): Asynchronous client for Azure OpenAI API.
        logger (Logger): Logger instance for logging.
    """

    def __init__(self, config: Optional[AzureOpenAIConfig] = None) -> None:
        """
        Initialize API client with configuration.

        Args:
            config (Optional[AzureOpenAIConfig]): Configuration settings for Azure OpenAI service.

        Raises:
            Exception: If initialization fails.
        """
        self.logger = LoggerSetup.get_logger(__name__)
        try:
            self.config = config or AzureOpenAIConfig.from_env()
            self.client = AsyncAzureOpenAI(
                api_key=self.config.api_key,
                api_version=self.config.api_version,
                azure_endpoint=self.config.endpoint
            )
            self.logger.info("APIClient initialized successfully",
                             extra={"api_version": self.config.api_version})
        except Exception as e:
            self.logger.error(f"Failed to initialize APIClient: {e}")
            raise

    async def process_request(
        self,
        prompt: str,
        temperature: float = 0.4,
        max_tokens: int = 6000,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, str]] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Process request to Azure OpenAI API.

        Args:
            prompt (str): The prompt text to send to the API.
            temperature (float): Sampling temperature for the API response.
            max_tokens (int): Maximum number of tokens in the response.
            tools (Optional[List[Dict[str, Any]]]): Optional tools for the API request.
            tool_choice (Optional[Dict[str, str]]): Optional tool choice for the API request.

        Returns:
            Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]: API response and usage information.

        Raises:
            Exception: If the API request fails.
        """
        try:
            self.logger.debug("Processing API request",
                              extra={"max_tokens": max_tokens, "temperature": temperature})

            response = await self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice
            )

            usage = response.usage.model_dump() if hasattr(response, 'usage') else {}
            self.logger.debug("API request successful", extra={"usage": usage})

            return response, usage

        except Exception as e:
            self.logger.error(f"API request failed: {e}",
                              extra={"prompt_length": len(prompt)})
            raise

    async def close(self) -> None:
        """
        Close API client resources.

        Raises:
            Exception: If closing the client fails.
        """
        try:
            await self.client.close()
            self.logger.info("APIClient closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing APIClient: {e}")
            raise