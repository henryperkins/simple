import aiohttp  # Ensure aiohttp is imported
from typing import Optional, Dict, Any
from openai import AsyncAzureOpenAI
from core.config import AzureOpenAIConfig
from core.logger import LoggerSetup
from core.response_parsing import ResponseParsingService
from api.token_management import TokenManager

class APIClient:
    """Client to interact with Azure OpenAI API with centralized response parsing."""

    def __init__(
        self,
        config: Optional[AzureOpenAIConfig] = None,
        response_parser: Optional[ResponseParsingService] = None,
        token_manager: Optional[TokenManager] = None
    ) -> None:
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

    async def test_connection(self):
        """Test connection to Azure OpenAI endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.config.endpoint, timeout=self.config.request_timeout) as response:
                    if response.status != 200:
                        self.logger.error(f"Connection test failed with status code: {response.status}")
                        raise ConnectionError("Failed to connect to Azure OpenAI endpoint")
            self.logger.info("Connection to Azure OpenAI endpoint successful")
        except Exception as e:
            self.logger.error(f"Error testing connection to Azure OpenAI endpoint: {e}")
            raise

    async def process_request(self, prompt: str) -> Dict[str, Any]:
        """Process a request with token tracking and usage calculation."""
        try:
            request_tokens = self.token_manager.estimate_tokens(prompt)
            request_params = await self.token_manager.validate_and_prepare_request(prompt)
            
            response = await self.client.chat.completions.create(**request_params)
            response_content = response.choices[0].message.content if response.choices else ""
            response_tokens = self.token_manager.estimate_tokens(response_content)
            
            # Track usage
            self.token_manager.track_request(request_tokens, response_tokens)
            usage = self.token_manager._calculate_usage(request_tokens, response_tokens)
            
            self.logger.info(f"Processed request with usage: {usage}")
            return {"response": response_content, "usage": usage}
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
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
        await self.test_connection()  # Ensure connection is tested when entering the context
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()