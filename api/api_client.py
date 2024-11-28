from typing import Optional, Dict, Any, Tuple
import aiohttp
import json
from core.config import AzureOpenAIConfig
from core.logger import LoggerSetup
from exceptions import APIError, ConfigurationError

# Initialize logger
logger = LoggerSetup.get_logger(__name__)

class APIClient:
    """API Client for managing interactions with the Azure OpenAI API."""
    
    def __init__(self):
        """Initialize the API client with configuration."""
        try:
            self.config = AzureOpenAIConfig.from_env()
            self.api_key = self.config.api_key
            self.endpoint = self.config.endpoint
            self.deployment_name = self.config.deployment_name
            self.api_version = self.config.api_version
            
            if not all([self.api_key, self.endpoint, self.deployment_name]):
                raise ConfigurationError("API key, endpoint, or deployment name not properly configured.")
            
            self._session = None
            logger.info("APIClient initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize APIClient: {str(e)}")
            raise ConfigurationError("Failed to initialize the APIClient.")

    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={
                    "api-key": self.api_key,
                    "Content-Type": "application/json"
                }
            )

    async def close(self):
        """Close the API client session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def process_request(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        tools: Optional[list] = None,
        tool_choice: Optional[dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Process a request to the Azure OpenAI API.

        Args:
            prompt (str): The input prompt
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens to generate
            tools (Optional[list]): List of tools/functions to use
            tool_choice (Optional[dict]): Tool choice configuration

        Returns:
            Tuple[Dict[str, Any], Dict[str, int]]: Response and usage data

        Raises:
            APIError: If the request fails
        """
        try:
            await self._ensure_session()
            
            url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"
            
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            if tools:
                payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

            async with self._session.post(url, json=payload) as response:
                response.raise_for_status()
                data = await response.json()

                if "choices" not in data or not data["choices"]:
                    raise APIError("No choices in API response")

                usage = data.get("usage", {})
                return data["choices"][0], usage

        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {str(e)}")
            raise APIError(f"API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {str(e)}")
            raise APIError(f"Failed to parse API response: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in API request: {str(e)}")
            raise APIError(f"Unexpected error in API request: {str(e)}")

    def send_request(self, prompt: str, max_tokens: Optional[int] = 150) -> Optional[Dict[str, Any]]:
        """
        Send a prompt to the Azure OpenAI API and receive a response.

        Args:
            prompt (str): The input prompt to send to the API.
            max_tokens (Optional[int]): The maximum number of tokens to generate in the response.

        Returns:
            Optional[Dict[str, Any]]: Parsed JSON response from the API or None in case of failure.

        Raises:
            APIError: If the request fails or the response is invalid.
        """
        try:
            url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/completions?api-version=2023-06-01-preview"
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 1.0,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
            
            logger.info(f"Sending request to Azure OpenAI with payload: {payload}")
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()

            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                logger.info("Received response from Azure OpenAI.")
                return data["choices"][0]
            else:
                logger.error("Unexpected response format received from API.")
                raise APIError("Unexpected response format received from API.")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request to Azure OpenAI failed: {str(req_err)}")
            raise APIError(f"Request to Azure OpenAI failed: {str(req_err)}")
        except ValueError as val_err:
            logger.error(f"Error parsing response: {str(val_err)}")
            raise APIError(f"Error parsing response: {str(val_err)}")
        except Exception as e:
            logger.error(f"Unexpected error occurred: {str(e)}")
            raise APIError(f"Unexpected error occurred: {str(e)}")

    def validate_api_key(self) -> bool:
        """
        Validate the Azure OpenAI API key by making a simple request.

        Returns:
            bool: True if the API key is valid, False otherwise.
        """
        try:
            url = f"{self.endpoint}/openai/deployments?api-version=2023-06-01-preview"
            logger.info("Validating Azure OpenAI API key...")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            logger.info("API key validation successful.")
            return True
        except requests.exceptions.RequestException as req_err:
            logger.error(f"API key validation failed: {str(req_err)}")
            return False
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        
# Example usage
if __name__ == "__main__":
    try:
        client = APIClient()
        prompt_text = "Explain the concept of recursion in programming."
        response = client.send_request(prompt_text)
        if response:
            print("Generated Response:")
            print(response.get("text"))
        else:
            print("No response received.")
    except ConfigurationError as conf_err:
        print(f"Configuration Error: {str(conf_err)}")
    except APIError as api_err:
        print(f"API Error: {str(api_err)}")