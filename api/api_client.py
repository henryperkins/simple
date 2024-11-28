"""  
Azure OpenAI API Client Module  
  
Provides a client wrapper for Azure OpenAI API interactions, handling requests,  
responses, and error management.  
"""  
  
from typing import Optional, Dict, Any, List, Tuple  
import openai  
from openai.error import OpenAIError  
from datetime import datetime  
  
from core.logger import LoggerSetup, log_error, log_info  
from core.monitoring import MetricsCollector  
from config import AzureOpenAIConfig  
from exceptions import APIError, ConfigurationError  
  
# Initialize logger and config  
logger = LoggerSetup.get_logger(__name__)  
config = AzureOpenAIConfig.from_env()  
  
  
class AzureOpenAIClient:  
    """  
    Client wrapper for Azure OpenAI API interactions.  
  
    Handles API requests, responses, and error management with proper logging  
    and metrics collection.  
    """  
  
    def __init__(  
        self,  
        token_manager,  
        metrics_collector: Optional[MetricsCollector] = None  
    ):  
        """  
        Initialize the Azure OpenAI API client.  
  
        Args:  
            token_manager: Token management instance for tracking usage  
            metrics_collector (Optional[MetricsCollector]): Collector for tracking metrics  
        """  
        try:  
            self.config = config  
            self.token_manager = token_manager  
            self.metrics = metrics_collector  
  
            # Setup OpenAI with Azure settings  
            openai.api_type = "azure"  
            openai.api_base = self.config.endpoint  
            openai.api_version = self.config.api_version  
            openai.api_key = self.config.api_key  
  
            logger.info("Azure OpenAI client initialized successfully")  
  
        except Exception as e:  
            logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")  
            raise ConfigurationError(f"Client initialization failed: {str(e)}")  
  
    async def generate_completion(  
        self,  
        messages: List[Dict[str, str]],  
        temperature: Optional[float] = None,  
        max_tokens: Optional[int] = None,  
        stream: bool = False  
    ) -> Optional[Dict[str, Any]]:  
        """  
        Generate a completion from the API.  
  
        Args:  
            messages (List[Dict[str, str]]): List of message dictionaries  
            temperature (Optional[float]): Sampling temperature  
            max_tokens (Optional[int]): Maximum tokens to generate  
            stream (bool): Whether to stream the response  
  
        Returns:  
            Optional[Dict[str, Any]]: API response containing completion and usage info  
  
        Raises:  
            APIError: If the API request fails  
        """  
        operation_start = datetime.now()  
  
        try:  
            response = await openai.ChatCompletion.acreate(  
                engine=self.config.deployment_name,  
                messages=messages,  
                temperature=temperature or self.config.temperature,  
                max_tokens=max_tokens or self.config.max_tokens,  
                stream=stream  
            )  
  
            if stream:  
                # Handle streaming if necessary  
                # For simplicity, not implemented here  
                logger.warning("Streaming not implemented in this client.")  
                return None  
  
            completion_data = {  
                "content": response.choices[0].message.content if response.choices else None,  
                "usage": response.usage.to_dict() if response.usage else {}  
            }  
  
            if self.metrics:  
                duration = (datetime.now() - operation_start).total_seconds()  
                await self.metrics.track_operation(  
                    operation_type='completion',  
                    success=True,  
                    duration=duration  
                )  
  
            return completion_data  
  
        except OpenAIError as e:  
            await self._handle_api_error(e, operation_start)  
            return None  
  
        except Exception as e:  
            await self._handle_unexpected_error(e, operation_start)  
            return None  
  
    async def process_request(  
        self,  
        prompt: str,  
        temperature: Optional[float] = None,  
        max_tokens: Optional[int] = None  
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:  
        """  
        Process a request with token tracking and usage calculation.  
  
        Args:  
            prompt (str): The prompt text to process  
            temperature (Optional[float]): Sampling temperature  
            max_tokens (Optional[int]): Maximum tokens to generate  
  
        Returns:  
            Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:  
                Tuple of (response, usage statistics)  
        """  
        try:  
            # Validate request  
            is_valid, metrics, message = await self.token_manager.validate_request(  
                prompt, max_tokens  
            )  
  
            if not is_valid:  
                logger.error(f"Request validation failed: {message}")  
                return None, None  
  
            # Track token usage  
            request_tokens = self.token_manager.estimate_tokens(prompt)  
  
            # Generate completion  
            response = await self.generate_completion(  
                messages=[{"role": "user", "content": prompt}],  
                temperature=temperature,  
                max_tokens=max_tokens  
            )  
  
            if not response:  
                return None, None  
  
            # Get the actual usage from the API response if available  
            usage_data = response.get('usage', {})  
            prompt_tokens = usage_data.get('prompt_tokens', request_tokens)  
            completion_tokens = usage_data.get('completion_tokens', self.token_manager.estimate_tokens(  
                response["content"]  
            ))  
  
            # Update token tracking  
            self.token_manager.track_request(  
                prompt_tokens, completion_tokens  
            )  
  
            # Calculate usage statistics  
            usage = self.token_manager.calculate_usage(  
                prompt_tokens=prompt_tokens,  
                completion_tokens=completion_tokens,  
                cached=self.config.cache_enabled  
            )  
  
            return response, usage.__dict__  
  
        except Exception as e:  
            logger.error(f"Error processing request: {str(e)}")  
            return None, None  
  
    async def _handle_api_error(  
        self,  
        error: OpenAIError,  
        start_time: datetime  
    ) -> None:  
        """  
        Handle Azure OpenAI API errors.  
  
        Args:  
            error (OpenAIError): The API error  
            start_time (datetime): Operation start time  
        """  
        log_error(f"API error: {str(error)}")  
  
        if self.metrics:  
            duration = (datetime.now() - start_time).total_seconds()  
            await self.metrics.track_operation(  
                operation_type='completion',  
                success=False,  
                duration=duration,  
                error=str(error)  
            )  
  
        raise APIError(f"Azure OpenAI API error: {str(error)}") from error  
  
    async def _handle_unexpected_error(  
        self,  
        error: Exception,  
        start_time: datetime  
    ) -> None:  
        """  
        Handle unexpected errors during API operations.  
  
        Args:  
            error (Exception): The unexpected error  
            start_time (datetime): Operation start time  
        """  
        logger.error(f"Unexpected error: {str(error)}")  
  
        if self.metrics:  
            duration = (datetime.now() - start_time).total_seconds()  
            await self.metrics.track_operation(  
                operation_type='completion',  
                success=False,  
                duration=duration,  
                error=str(error)  
            )  
  
        raise APIError(f"Unexpected error during API operation: {str(error)}") from error  
  
    async def close(self) -> None:  
        """Close the API client and cleanup resources."""  
        # The OpenAI library does not require closing any connections  
        try:  
            logger.info("API client closed successfully")  
        except Exception as e:  
            logger.error(f"Error closing client: {str(e)}")  
  
    async def __aenter__(self) -> 'AzureOpenAIClient':  
        """Async context manager entry."""  
        return self  
  
    async def __aexit__(  
        self,  
        exc_type: Optional[type],  
        exc_val: Optional[Exception],  
        exc_tb: Optional[Any]  
    ) -> None:  
        """Async context manager exit."""  
        await self.close()  