"""  
API Client Module  
  
Provides an interface to interact with the Azure OpenAI API, handling  
requests and responses, including support for function calling and structured outputs.  
"""  
  
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncAzureOpenAI

from core.config import AzureOpenAIConfig
from core.logger import LoggerSetup


class APIClient:  
    """Client to interact with Azure OpenAI API."""  
  
    def __init__(self, config: Optional[AzureOpenAIConfig] = None):  
        """Initialize API client with configuration."""  
        self.logger = LoggerSetup.get_logger(__name__)  # Initialize self.logger  
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
        """Process request to Azure OpenAI API."""  
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
            return None, None  
  
    async def close(self) -> None:  
        """Close API client resources."""  
        try:  
            await self.client.close()  
            self.logger.info("APIClient closed successfully")  
        except Exception as e:  
            self.logger.error(f"Error closing APIClient: {e}")  