"""
This module provides the APIClient class to interact with the Azure OpenAI API.

The APIClient handles connection setup, request processing with token tracking and usage calculation,
response parsing, and resource management. It utilizes asynchronous operations and encapsulates error
handling and logging functionality.
"""

import aiohttp
from typing import Optional, Dict, Any
from openai import AsyncAzureOpenAI
from core.config import AzureOpenAIConfig
from core.logger import LoggerSetup
from core.response_parsing import ResponseParsingService
from api.token_management import TokenManager

class APIClient:
    """Client to interact with the Azure OpenAI API with centralized response parsing."""

    def __init__(
        self,
        config: Optional[AzureOpenAIConfig] = None,
        response_parser: Optional[ResponseParsingService] = None,
        token_manager: Optional[TokenManager] = None
    ) -> None:
        """Initialize the APIClient."""
        self.logger = LoggerSetup.get_logger(__name__)
        try:
            self.config = config or AzureOpenAIConfig.from_env()
            self.client = AsyncAzureOpenAI(
                api_key=self.config.api_key,
                api_version=self.config.api_version,
                azure_endpoint=self.config.endpoint
            )
            self.response_parser = response_parser or ResponseParsingService()
            self.token_manager = token_manager or TokenManager()
            self.logger.info("APIClient initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize APIClient: {e}")
            raise

    def get_client(self) -> AsyncAzureOpenAI:
        """Get the initialized OpenAI client."""
        return self.client

    async def process_request(self, prompt: str) -> Dict[str, Any]:
        """Process a request to the Azure OpenAI API with token tracking."""
        try:
            # Enhance prompt generation with context-specific information
            context_info = "Provide detailed and context-specific information."
            enhanced_prompt = f"{context_info}\n\n{prompt}"

            request_tokens = self.token_manager.estimate_tokens(enhanced_prompt)
            request_params = await self.token_manager.validate_and_prepare_request(enhanced_prompt)
            
            response = await self.client.chat.completions.create(
                model=self.config.deployment_id,
                messages=[{"role": "user", "content": enhanced_prompt}],
                max_tokens=request_params.get("max_tokens", 1000),
                temperature=request_params.get("temperature", 0.7)
            )

            response_content = response.choices[0].message.content
            response_tokens = self.token_manager.estimate_tokens(response_content)
            
            self.token_manager.track_request(request_tokens, response_tokens)
            usage = self.token_manager.calculate_usage(request_tokens, response_tokens)
            
            self.logger.info(f"Processed request with usage: {usage}")
            return {"response": response_content, "usage": usage}
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            # Add error handling and fallback mechanism
            fallback_response = {
                "response": "An error occurred while processing the request. Please try again later.",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost": 0.0}
            }
            return fallback_response

    async def close(self) -> None:
        """Close the API client resources."""
        try:
            await self.client.close()
            self.logger.info("APIClient closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing APIClient: {e}")
            raise

    async def test_connection(self) -> None:
        """Tests the connection to the Azure OpenAI endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.config.endpoint, 
                    timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
                    headers={"api-key": self.config.api_key}
                ) as response:
                    if response.status != 200:
                        self.logger.error(f"Connection test failed with status code: {response.status}")
                        response_text = await response.text()
                        self.logger.error(f"Response: {response_text}")
                        raise ConnectionError(f"Failed to connect to Azure OpenAI endpoint: {response_text}")
            self.logger.info("Connection to Azure OpenAI endpoint successful")
        except Exception as e:
            self.logger.error(f"Error testing connection to Azure OpenAI endpoint: {e}")
            raise

    async def __aenter__(self) -> 'APIClient':
        """
        Asynchronously enters the runtime context and tests the connection.

        Returns:
            APIClient: The APIClient instance.

        Raises:
            Exception: If entering the context or connection test fails.
        """
        await self.test_connection()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Asynchronously exits the runtime context and closes resources.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception value if raised.
            exc_tb: Exception traceback if raised.

        Raises:
            Exception: If closing resources fails.
        """
        await self.close()
