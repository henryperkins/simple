"""
Simplified Azure OpenAI API client wrapper.
Provides basic API interaction functionality used by AIInteractionHandler.
"""

from typing import Optional, Dict, Any, List
from openai import AsyncAzureOpenAI
from openai import OpenAIError
from core.config import AzureOpenAIConfig
from core.logger import log_info, log_error, log_debug

class AzureOpenAIClient:
    """
    Simple client wrapper for Azure OpenAI API interactions.
    Handles basic communication with the API, leaving higher-level operations
    to AIInteractionHandler.
    """

    def __init__(self, config: AzureOpenAIConfig):
        """Initialize with Azure OpenAI configuration."""
        if not config.validate():
            raise ValueError("Invalid Azure OpenAI configuration")
            
        self.config = config
        self.client = AsyncAzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint
        )
        log_info("Azure OpenAI client initialized")

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a completion from the API.

        Args:
            messages: List of message dictionaries
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Returns:
            Optional[Dict[str, Any]]: API response or None if failed
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens
            )
            
            return {
                "content": response.choices[0].message.content if response.choices else None,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                } if response.usage else None
            }

        except OpenAIError as e:
            log_error(f"API error: {str(e)}")
            return None
        except Exception as e:
            log_error(f"Unexpected error: {str(e)}")
            return None

    async def health_check(self) -> Dict[str, Any]:
        """
        Check API health with a simple request.

        Returns:
            Dict[str, Any]: Health check results
        """
        try:
            start_time = time.time()
            response = await self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            latency = time.time() - start_time
            
            return {
                "status": "healthy" if response else "unhealthy",
                "latency": round(latency, 3),
                "model": self.config.deployment_name
            }
        except Exception as e:
            log_error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.config.deployment_name
            }

    async def close(self):
        """Close the API client."""
        try:
            await self.client.close()
            log_info("API client closed")
        except Exception as e:
            log_error(f"Error closing client: {str(e)}")

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()