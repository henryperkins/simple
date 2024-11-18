"""
Azure OpenAI Client Module

This module provides a high-level client for interacting with Azure OpenAI
to generate docstrings with proper async support and error handling.
"""

import asyncio
from typing import List, Tuple, Optional, Dict, Any
from openai import AsyncAzureOpenAI
from openai import OpenAIError
from core.cache import Cache
from core.config import AzureOpenAIConfig
from api.token_management import TokenManager
from api.api_interaction import APIInteraction
from core.logger import log_info, log_error, log_debug
from core.exceptions import TooManyRetriesError
from core.monitoring import SystemMonitor



class AzureOpenAIClient:
    """
    Client for interacting with Azure OpenAI to generate docstrings.

    This class manages the configuration and initializes the components necessary
    for API interaction. It provides a high-level interface for generating docstrings
    and managing the cache.
    """

    def __init__(self, config: Optional[AzureOpenAIConfig] = None):
        """
        Initialize the AzureOpenAIClient with necessary configuration.

        Args:
            config (Optional[AzureOpenAIConfig]): Configuration instance for Azure OpenAI.
                If not provided, will load from environment variables.

        Raises:
            ValueError: If the configuration is invalid
        """
        self.config = config or AzureOpenAIConfig.from_env()
        if not self.config.validate():
            raise ValueError("Invalid Azure OpenAI configuration")

        self.token_manager = TokenManager(
            model=self.config.model_name,
            deployment_name=self.config.deployment_name
        )
        self.cache = Cache()
        
        # Initialize Azure OpenAI client
        self._client = AsyncAzureOpenAI(
            api_key=self.config.api_key,
            api_version=self.config.api_version,
            azure_endpoint=self.config.endpoint
        )
        
        # Initialize API interaction handler
        self.api_interaction = APIInteraction(
            self.config,
            self.token_manager,
            self.cache,
            SystemMonitor()  # Pass the SystemMonitor instance
        )

        log_info("Azure OpenAI client initialized successfully")

    async def generate_docstring(
        self,
        func_name: str,
        params: List[Tuple[str, str]],
        return_type: str,
        complexity_score: int,
        existing_docstring: str,
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a docstring for a function using Azure OpenAI.

        Args:
            func_name: Name of the function
            params: List of parameter names and types
            return_type: Return type of the function
            complexity_score: Complexity score of the function
            existing_docstring: Existing docstring if any
            decorators: List of decorators
            exceptions: List of exceptions

        Returns:
            Optional[Dict[str, Any]]: Generated docstring and metadata, or None if failed.
                The dictionary contains:
                - content: The generated docstring content
                - usage: Token usage statistics
                - metadata: Additional generation metadata

        Raises:
            TooManyRetriesError: If maximum retry attempts are exceeded
        """
        try:
            log_debug(f"Generating docstring for function: {func_name}")
            
            response = await self.api_interaction.get_docstring(
                func_name=func_name,
                params=params,
                return_type=return_type,
                complexity_score=complexity_score,
                existing_docstring=existing_docstring,
                decorators=decorators,
                exceptions=exceptions
            )
            
            if response and isinstance(response, dict):
                log_info(f"Successfully generated docstring for {func_name}")
                return response
                
            log_error(f"Failed to generate docstring for {func_name}")
            return None
            
        except OpenAIError as e:
            log_error(f"OpenAI API error for {func_name}: {str(e)}")
            return None
        except TooManyRetriesError as e:
            log_error(f"Max retries exceeded for {func_name}: {e}")
            raise
        except Exception as e:
            log_error(f"Error generating docstring for {func_name}: {e}")
            return None

    async def batch_generate_docstrings(
        self,
        functions: List[Dict[str, Any]],
        batch_size: Optional[int] = None
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Generate docstrings for multiple functions in batches.

        Args:
            functions: List of function metadata dictionaries, each containing:
                - func_name: Name of the function
                - params: List of parameter tuples (name, type)
                - return_type: Function return type
                - complexity_score: Function complexity score
                - existing_docstring: Existing docstring if any
                - decorators: Optional list of decorators
                - exceptions: Optional list of exceptions
            batch_size: Number of functions to process concurrently.
                       Defaults to config setting if not specified.

        Returns:
            List[Optional[Dict[str, Any]]]: List of generated docstrings and metadata
        """
        batch_size = batch_size or getattr(self.config, 'batch_size', 10)
        results = []
        
        for i in range(0, len(functions), batch_size):
            batch = functions[i:i + batch_size]
            try:
                log_debug(f"Processing batch {i//batch_size + 1}, size {len(batch)}")
                
                batch_results = await asyncio.gather(*[
                    self.generate_docstring(**func) for func in batch
                ], return_exceptions=True)
                
                for func, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        log_error(f"Error processing {func.get('func_name', 'unknown')}: {result}")
                        results.append(None)
                    else:
                        results.append(result)
                
            except Exception as e:
                log_error(f"Batch processing error: {str(e)}")
                results.extend([None] * len(batch))
        
        return results

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the Azure OpenAI service.

        Returns:
            Dict[str, Any]: Health check results including status, latency, and metrics
        """
        return await self.api_interaction.health_check()

    async def close(self):
        """Close the client and release any resources."""
        try:
            if self.api_interaction:
                await self.api_interaction.close()
            if self._client:
                await self._client.close()
        except Exception as e:
            log_error(f"Error closing API client: {str(e)}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
