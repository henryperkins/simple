# api_interaction.py

import openai
import aiohttp
import json
import logging
import hashlib
import asyncio
import sentry_sdk
from config import Config
from monitoring import capture_openai_error
from typing import Dict, Any, Callable
from cache import ThreadSafeCache
from error_handling import ProcessingResult

# Initialize a thread-safe cache for prompt responses
prompt_cache = ThreadSafeCache(max_size_mb=Config.CACHE_MAX_SIZE_MB)

def get_cached_response(prompt_key: str) -> Optional[Dict[str, Any]]:
    """Retrieve a cached response if available.

    Args:
        prompt_key (str): The unique key for the prompt.

    Returns:
        Optional[Dict[str, Any]]: The cached response if available, otherwise None.
    """
    return prompt_cache.get(prompt_key)

def cache_response(prompt_key: str, response: Dict[str, Any]) -> None:
    """Cache the response for a given prompt key.

    Args:
        prompt_key (str): The unique key for the prompt.
        response (Dict[str, Any]): The response to cache.
    """
    prompt_cache.set(prompt_key, response)

class RateLimiter:
    """A token bucket rate limiter for controlling API request rates."""

    def __init__(self, tokens_per_second: float, bucket_size: int):
        """Initialize the rate limiter.

        Args:
            tokens_per_second (float): Rate at which tokens are added to the bucket.
            bucket_size (int): Maximum number of tokens in the bucket.
        """
        self.tokens_per_second = tokens_per_second
        self.bucket_size = bucket_size
        self.tokens = bucket_size
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token from the bucket, waiting if necessary."""
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(self.bucket_size, self.tokens + time_passed * self.tokens_per_second)
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.tokens_per_second
                await asyncio.sleep(wait_time)
                self.tokens = 1
            
            self.tokens -= 1
            self.last_update = now

class APIRetryStrategy:
    """A strategy for retrying API requests with exponential backoff."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """Initialize the retry strategy.

        Args:
            max_retries (int): Maximum number of retry attempts.
            base_delay (float): Initial delay between retries in seconds.
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.rate_limiter = RateLimiter(tokens_per_second=0.5, bucket_size=10)

    async def execute_with_retry(self, operation: Callable, *args, **kwargs) -> ProcessingResult:
        """Execute an operation with retry logic.

        Args:
            operation (Callable): The operation to execute.
            *args: Positional arguments for the operation.
            **kwargs: Keyword arguments for the operation.

        Returns:
            ProcessingResult: The result of the operation.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                await self.rate_limiter.acquire()
                result = await operation(*args, **kwargs)
                return ProcessingResult(success=True, data=result)
            except aiohttp.ClientResponseError as e:
                if e.status == 429:  # Rate limit
                    retry_after = float(e.headers.get('Retry-After', self.base_delay))
                    logging.warning(f"Rate limit hit. Retrying after {retry_after} seconds.")
                    await asyncio.sleep(retry_after)
                elif e.status >= 500:  # Server error
                    delay = self.base_delay * (2 ** retries)
                    logging.warning(f"Server error {e.status}. Retrying after {delay} seconds.")
                    await asyncio.sleep(delay)
                else:
                    logging.error(f"Client error {e.status}: {str(e)}")
                    return ProcessingResult(success=False, error=str(e))
                retries += 1
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                return ProcessingResult(success=False, error=str(e))
        
        logging.error("Max retries exceeded.")
        return ProcessingResult(success=False, error="Max retries exceeded")

async def make_openai_request(model_name: str, messages: list, functions: list, service: str) -> Dict[str, Any]:
    """Make a request to the OpenAI or Azure OpenAI API.

    Args:
        model_name (str): The name of the model to use.
        messages (list): The messages to send to the API.
        functions (list): The function schema for the API request.
        service (str): The service to use ('openai' or 'azure').

    Returns:
        Dict[str, Any]: The API response.

    Raises:
        Exception: If the API request fails.
    """
    try:
        with sentry_sdk.start_span(op="openai_request", description=f"{service} API call"):
            if service == "openai":
                openai.api_key = Config.OPENAI_API_KEY
                async def make_request():
                    response = openai.ChatCompletion.create(
                        model=model_name,
                        messages=messages,
                        functions=functions,
                        function_call="auto",
                        max_tokens=2000,
                        temperature=0.2
                    )
                    return response
            elif service == "azure":
                endpoint = f"{Config.AZURE_OPENAI_ENDPOINT}/openai/deployments/{Config.AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=2024-02-15-preview"
                headers = {"api-key": Config.AZURE_OPENAI_API_KEY}
                async def make_request():
                    async with aiohttp.ClientSession() as session:
                        async with session.post(endpoint, json={
                            "model": model_name,
                            "messages": messages,
                            "functions": functions,
                            "function_call": "auto",
                            "max_tokens": 2000,
                            "temperature": 0.2,
                        }, headers=headers) as response:
                            response_text = await response.text()
                            try:
                                return json.loads(response_text)
                            except json.JSONDecodeError as json_err:
                                logging.error(f"JSON decoding failed: {json_err}")
                                raise
            retry_strategy = APIRetryStrategy()
            return await retry_strategy.execute_with_retry(make_request)
    except Exception as e:
        capture_openai_error(e, {
            "model": model_name,
            "service": service,
            "message_count": len(messages)
        })
        raise

async def analyze_function_with_openai(function_details: Dict[str, Any], service: str) -> Dict[str, Any]:
    """Analyze a function using OpenAI's API and generate documentation.

    Args:
        function_details (Dict[str, Any]): Details of the function to analyze.
        service (str): The service to use ('openai' or 'azure').

    Returns:
        Dict[str, Any]: The analysis results, including summary, docstring, and changelog.

    Raises:
        Exception: If an error occurs during the analysis.
    """
    try:
        with sentry_sdk.start_span(op="function_analysis", description=function_details["name"]):
            function_hash = hashlib.sha256(function_details["code"].encode("utf-8")).hexdigest()

            cached_result = get_cached_response(function_hash)
            if cached_result:
                logging.info(f"Using cached result for function '{function_details['name']}'")
                return cached_result

            complexity_score = function_details.get("complexity_score", "Unknown")

            function_schema = [{
                "name": "analyze_function",
                "description": "Analyzes a Python function and provides documentation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "Brief description of function"},
                        "docstring": {"type": "string", "description": "Complete docstring"},
                    },
                    "required": ["summary", "docstring"]
                }
            }]

            messages = [
                {"role": "system", "content": "You are a Python documentation specialist."},
                {"role": "user", "content": (
                    f"Analyze this function:\n\n"
                    f"Name: {function_details['name']}\n"
                    f"Code:\n```python\n{function_details['code']}\n```\n"
                    f"Complexity Score: {complexity_score}"
                )}
            ]

            completion = await make_openai_request(
                model_name="gpt-4" if service == "openai" else "gpt-4-32k",
                messages=messages,
                functions=function_schema,
                service=service
            )

            result = {
                "name": function_details["name"],
                "complexity_score": complexity_score,
                "summary": completion.get("summary", "No summary available"),
                "docstring": completion.get("docstring", "No documentation available")
            }

            cache_response(function_hash, result)
            return result

    except Exception as e:
        logging.error(f"Error analyzing function {function_details['name']}: {e}")
        sentry_sdk.capture_exception(e)
        return {
            "name": function_details["name"],
            "complexity_score": None,
            "summary": f"Error during analysis: {str(e)}",
            "docstring": "Error: Documentation generation failed"
        }
