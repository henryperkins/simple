"""
API Interaction Module

This module handles interactions with the Azure OpenAI API, including making requests,
handling retries, managing rate limits, and validating connections.
"""

import asyncio
import json
from typing import List, Tuple, Optional, Dict, Any, Union, Mapping, Iterable
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, Function
from core.logger import log_info, log_error, log_debug, log_warning
from api.token_management import TokenManager, TokenUsage
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.exceptions import TooManyRetriesError
from docs.docstring_utils import DocstringValidator
from core.monitoring import SystemMonitor  # Assuming this is where the monitor is defined

class APIInteraction:
    """Handles interactions with the Azure OpenAI API."""

    def __init__(
        self, config: AzureOpenAIConfig, token_manager: TokenManager, cache: Cache, monitor: SystemMonitor
    ):
        """Initializes the APIInteraction with necessary components."""
        log_debug("Initializing APIInteraction with Azure OpenAI configuration")
        
        # Initialize Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint
        )
        
        self.token_manager = token_manager
        self.cache = cache
        self.config = config
        self.monitor = monitor
        self.current_retry = 0
        self.validator = DocstringValidator()  # Add validator instance
        log_info("APIInteraction initialized successfully.")

    def _log_token_usage(self, func_name: str, token_usage: TokenUsage, response_time: float = 0.0, error: Optional[str] = None):
        """Logs token usage for a function."""
        log_info(f"Token usage for {func_name}: {token_usage.total_tokens} tokens used, response time: {response_time}s")
        if error:
            log_error(f"Error during token usage logging for {func_name}: {error}")

    def _get_docstring_function(self) -> Function:
        """Enhanced function schema for docstring generation."""
        return {
            "name": "generate_docstring",
            "description": "Generate a structured docstring for a function",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief description of the function"
                    },
                    "parameters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "description": {"type": "string"},
                                "optional": {"type": "boolean"},
                                "default": {"type": ["string", "number", "boolean", "null"]}
                            },
                            "required": ["name", "type", "description"]
                        }
                    },
                    "returns": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["type", "description"]
                    },
                    "raises": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "exception": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["exception", "description"]
                        }
                    },
                    "examples": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["code"]
                        }
                    }
                },
                "required": ["summary", "parameters", "returns"]
            }
        }

    async def get_docstring(
        self,
        func_name: str,
        params: List[Tuple[str, str]],
        return_type: str,
        complexity_score: int,
        existing_docstring: str,
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generates a docstring for a function using Azure OpenAI with token management."""
        log_debug(f"Generating docstring for function: {func_name}")
        
        # Convert lists to tuples for hashable contexts
        cache_key = f"docstring:{func_name}:{hash(tuple(params))}:{hash(return_type)}"

        try:
            # Check cache first
            cached_response = await self.cache.get_cached_docstring(cache_key)
            if cached_response:
                log_info(f"Cache hit for function: {func_name}")
                return json.loads(cached_response)

            # Create messages
            messages: List[ChatCompletionMessageParam] = [
                {
                    "role": "system",
                    "content": "You are a technical documentation expert. Generate comprehensive and accurate function documentation."
                },
                {
                    "role": "user",
                    "content": self._create_prompt(
                        func_name, params, return_type, complexity_score,
                        existing_docstring, decorators, exceptions
                    )
                }
            ]

            # Validate token limits before making request
            prompt_text = json.dumps(messages)
            is_valid, metrics, validation_message = self.token_manager.validate_request(prompt_text)
            
            if not is_valid:
                log_error(f"Token validation failed: {validation_message}")
                return None

            # Optimize prompt if needed
            optimized_messages = await self._optimize_prompt(messages, {k: int(v) for k, v in metrics.items()})
            if optimized_messages is None:
                return None

            # Make API request with retry logic and token tracking
            for attempt in range(self.config.max_retries):
                try:
                    # Get token usage before the API call
                    token_usage_before = self.token_manager.estimate_tokens(
                        self._create_prompt(func_name, params, return_type, complexity_score, existing_docstring, decorators, exceptions)
                    )

                    start_time = asyncio.get_event_loop().time()
                    response = await self._make_api_request(optimized_messages, attempt)
                    response_time = asyncio.get_event_loop().time() - start_time

                    if response:
                        # Parse and validate the response
                        parsed_response = await self._process_response(response, {'function': func_name})
                        if parsed_response:
                            # Get token usage after the API call
                            token_usage_after = self.token_manager.estimate_tokens(response.choices[0].message.content)
                            token_usage = self.token_manager._calculate_usage(token_usage_before, token_usage_after)
                            
                            # Log token usage with monitoring system
                            self._log_token_usage(func_name, token_usage, response_time)

                            # Cache the successful response
                            await self.cache.save_docstring(
                                cache_key,
                                parsed_response,
                                ttl=self.config.cache_ttl
                            )
                            return parsed_response

                except Exception as e:
                    # Log token usage even on failure
                    token_usage_before = self.token_manager.estimate_tokens(
                        self._create_prompt(func_name, params, return_type, complexity_score, existing_docstring, decorators, exceptions)
                    )
                    token_usage = self.token_manager._calculate_usage(token_usage_before, 0)
                    self._log_token_usage(func_name, token_usage, error=str(e))
                    if not await self._handle_api_error(e, attempt):
                        break

            log_warning(f"Failed to generate docstring for {func_name} after {self.config.max_retries} attempts")
            return None

        except Exception as e:
            log_error(f"Error in get_docstring for {func_name}: {e}")
            return None

    async def _optimize_prompt(
        self, 
        messages: List[ChatCompletionMessageParam], 
        metrics: Mapping[str, int]
    ) -> Optional[List[ChatCompletionMessageParam]]:
        """Optimizes the prompt to fit within token limits."""
        try:
            optimized_messages, token_usage = self.token_manager.optimize_prompt(
                json.dumps(messages),
                max_tokens=self.config.max_tokens,
                preserve_sections=['parameters', 'returns']
            )
            
            log_info(f"Optimized prompt tokens: {token_usage.prompt_tokens}")
            return json.loads(optimized_messages)

        except Exception as e:
            log_error(f"Error optimizing prompt: {e}")
            return None

    async def _make_api_request(
        self, 
        messages: List[ChatCompletionMessageParam], 
        attempt: int
    ) -> Optional[ChatCompletion]:
        """Makes an API request with token tracking."""
        try:
            log_debug(f"Making API request, attempt {attempt + 1}")
            
            # Pre-request token check
            estimated_tokens = self.token_manager.estimate_tokens(json.dumps(messages))
            if estimated_tokens > self.config.max_tokens:
                log_warning("Estimated tokens exceed maximum limit")
                return None

            response = await self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=messages,
                functions=[Function(**self._get_docstring_function())],
                function_call={"name": "generate_docstring"},
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Track token usage
            if response.usage:
                self.token_manager.track_request(
                    request_tokens=response.usage.prompt_tokens,
                    response_tokens=response.usage.completion_tokens
                )
            
            log_debug("API request successful")
            return response

        except Exception as e:
            log_error(f"API request failed: {str(e)}")
            raise

    async def _process_response(self, response: ChatCompletion, 
                              error_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process and validate API response."""
        try:
            if not response.choices:
                return None

            function_call = response.choices[0].message.function_call
            if not function_call:
                return None

            parsed_args = json.loads(function_call.arguments)
            
            # Validate response content
            is_valid, validation_errors = self.validator.validate_docstring(parsed_args)
            
            if not is_valid:
                log_error(
                    f"Response validation failed for {error_context['function']}: "
                    f"{validation_errors}"
                )
                return None

            return {
                "content": parsed_args,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }

        except Exception as e:
            log_error(f"Response processing error: {e}")
            error_context['last_error'] = str(e)
            return None

    async def _handle_api_error(self, error: Exception, attempt: int) -> bool:
        """Handles API errors and determines if retry is appropriate."""
        log_warning(f"API error on attempt {attempt + 1}: {str(error)}")
        
        if attempt < self.config.max_retries - 1:
            retry_delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
            log_info(f"Retrying in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
            return True
        return False

    def _create_prompt(
        self,
        func_name: str,
        params: List[Tuple[str, str]],
        return_type: str,
        complexity_score: int,
        existing_docstring: str,
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[str]] = None,
    ) -> str:
        """Creates the prompt for the API request."""
        return f"""
Generate documentation for the following Python function:

Function Name: {func_name}
Parameters: {', '.join(f'{name}: {type_}' for name, type_ in params)}
Return Type: {return_type}
Decorators: {', '.join(decorators) if decorators else 'None'}
Exceptions: {', '.join(exceptions) if exceptions else 'None'}
Complexity Score: {complexity_score}
Existing Docstring: {existing_docstring if existing_docstring else 'None'}

Generate a comprehensive docstring that includes:
1. A clear summary of the function's purpose
2. Detailed parameter descriptions with types
3. Return value description with type
4. Usage examples
5. Time and space complexity analysis
6. Any relevant exceptions and their conditions

Use the generate_docstring function to provide structured output.
"""

    async def validate_connection(self) -> bool:
        """Validates the connection to Azure OpenAI service."""
        log_debug("Validating connection to Azure OpenAI service")
        try:
            # Make a minimal API request to verify connection
            response = await self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=1
            )
            log_info("Connection to Azure OpenAI API validated successfully")
            return True
        except Exception as e:
            log_error(f"Connection validation failed: {str(e)}")
            raise ConnectionError(f"Connection validation failed: {str(e)}")

    async def health_check(self) -> Dict[str, Any]:
        """Performs a comprehensive health check of the API service."""
        log_debug("Performing health check on Azure OpenAI service")
        health_status = {
            "status": "unknown",
            "latency": None,
            "token_usage": None,
            "error": None
        }

        try:
            start_time = asyncio.get_event_loop().time()
            
            # Test API functionality
            response = await self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=[{"role": "user", "content": "Health check"}],
                max_tokens=10
            )
            
            # Calculate latency
            latency = asyncio.get_event_loop().time() - start_time
            
            health_status.update({
                "status": "healthy",
                "latency": round(latency, 3),
                "token_usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            })
            
            log_info("Health check passed")
            
        except Exception as e:
            health_status.update({
                "status": "unhealthy",
                "error": str(e)
            })
            log_error(f"Health check failed: {e}")
            
        return health_status

    async def close(self):
        """Closes the API client and releases resources."""
        log_debug("Closing API client")
        try:
            # Close the client session
            await self.client.close()
            log_info("API client closed successfully")
        except Exception as e:
            log_error(f"Error closing API client: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def get_client_info(self) -> Dict[str, Any]:
        """Gets information about the API client configuration."""
        return {
            "endpoint": self.config.endpoint,
            "model": self.config.deployment_name,
            "api_version": self.config.api_version,
            "max_retries": self.config.max_retries,
            "timeout": self.config.request_timeout,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

    async def handle_rate_limits(self, retry_after: Optional[int] = None):
        """Handles rate limits by implementing exponential backoff."""
        if self.current_retry >= self.config.max_retries:
            raise TooManyRetriesError(
                f"Maximum retry attempts ({self.config.max_retries}) exceeded"
            )

        wait_time = retry_after or min(
            self.config.retry_delay * (2 ** self.current_retry),
            self.config.request_timeout
        )
        
        log_info(
            f"Rate limit encountered. Waiting {wait_time}s "
            f"(attempt {self.current_retry + 1}/{self.config.max_retries})"
        )

        self.current_retry += 1
        await asyncio.sleep(wait_time)

    def get_token_usage_stats(self) -> Dict[str, Union[int, float]]:
        """Returns current token usage statistics."""
        return self.token_manager.get_usage_stats()
