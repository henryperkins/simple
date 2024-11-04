<<<<<<< HEAD
=======
import openai
>>>>>>> 2d94ecaaf1aebee3c0cef377a3f4a9dfad24c7e6
import aiohttp
import json
import logging
import asyncio
<<<<<<< HEAD
import random
import hashlib
from typing import Dict, Any, List, Tuple
from config import Config
import sentry_sdk
from cache import get_cached_response, cache_response, initialize_cache
=======
import sentry_sdk
import time
from config import Config
from monitoring import capture_openai_error
from typing import Dict, Any, Callable, Optional
from cache import ThreadSafeCache
from error_handling import ProcessingResult
>>>>>>> 2d94ecaaf1aebee3c0cef377a3f4a9dfad24c7e6

def create_error_response(function_name: str, error_message: str) -> Dict[str, Any]:
    """Create an error response dictionary."""
    return {
        "name": function_name,
        "complexity_score": None,
        "summary": f"Error during analysis: {error_message}",
        "docstring": "Error: Documentation generation failed",
        "changelog": "No changelog available due to error."
    }

<<<<<<< HEAD
async def exponential_backoff_with_jitter(coro, max_retries: int = 5, initial_delay: float = 1.0, max_delay: float = 60.0) -> Any:
    """
    Retry a coroutine with exponential backoff and jitter.
=======
# Improved logging configuration
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

def get_cached_response(prompt_key: str) -> Optional[Dict[str, Any]]:
    """Retrieve a cached response if available.
>>>>>>> 2d94ecaaf1aebee3c0cef377a3f4a9dfad24c7e6

    Args:
        coro (coroutine): The coroutine to retry.
        max_retries (int): Maximum number of retries.
        initial_delay (float): Initial delay between retries.
        max_delay (float): Maximum delay between retries.

    Returns:
<<<<<<< HEAD
        Any: The result of the coroutine.
=======
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
                logging.error(f"Unexpected error: {str(e)}", exc_info=True)
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
>>>>>>> 2d94ecaaf1aebee3c0cef377a3f4a9dfad24c7e6

    Raises:
        Exception: If all retries fail.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return await coro()
        except Exception as e:
            if attempt == max_retries:
                logging.error(f"All {max_retries} retries failed.")
                raise
            jitter = random.uniform(0, delay * 0.1)
            total_delay = delay + jitter
            logging.warning(f"Attempt {attempt} failed with error: {e}. Retrying in {total_delay:.2f}s...")
            await asyncio.sleep(total_delay)
            delay = min(delay * 2, max_delay)
    raise Exception(f"Failed after {max_retries} retries")

async def make_openai_request(model_name: str, messages: list, functions: list, service: str) -> Dict[str, Any]:
    """Make an asynchronous request to the OpenAI or Azure OpenAI API with retries."""
    async def make_request() -> Dict[str, Any]:
        if service == "azure":
            endpoint = f"{Config.AZURE_OPENAI_ENDPOINT}/openai/deployments/{Config.AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={Config.AZURE_OPENAI_API_VERSION}"
            headers = {"api-key": Config.AZURE_OPENAI_API_KEY}
        else:
            endpoint = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }

        payload = {
            "model": model_name,
            "messages": messages,
            "functions": functions,
            "function_call": "auto",
            "max_tokens": 2000,
            "temperature": 0.2
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers, json=payload) as response:
                response_text = await response.text()
                logging.debug(f"Raw API response: {response_text}")
                try:
                    response_json = json.loads(response_text)
                except json.JSONDecodeError as json_err:
                    logging.error(f"JSON decoding failed: {json_err}")
                    sentry_sdk.capture_exception(json_err)
                    raise
                return response_json

    # Use exponential backoff with jitter for retries
    try:
        completion = await exponential_backoff_with_jitter(make_request)
        logging.debug(f"Full API response: {completion}")
        return completion
    except Exception as e:
        logging.error(f"API request failed: {e}")
        sentry_sdk.capture_exception(e)
        raise

async def analyze_function_with_openai(function_details: Dict[str, Any], service: str) -> Dict[str, Any]:
    """Analyze a function using OpenAI's API and generate documentation."""
    try:
        function_hash = hashlib.sha256(function_details["code"].encode("utf-8")).hexdigest()

        cached_result = get_cached_response(function_hash)
        if cached_result:
            logging.info(f"Using cached result for function '{function_details['name']}'")
            return cached_result

        system_prompt = (
            "You are a documentation specialist that analyzes Python functions and generates structured documentation. "
            "Always return responses in the following JSON format:\n"
            "{\n"
            "    'summary': '<concise function description>',\n"
            "    'docstring': '<Google-style docstring>',\n"
            "    'changelog': '<change history>'\n"
            "}"
        )
        user_prompt = (
            f"Analyze and document this function:\n\n"
            f"Function Name: {function_details['name']}\n"
            f"Parameters: {format_parameters(function_details['params'])}\n"
            f"Return Type: {function_details['return_type']}\n"
            f"Existing Docstring: {function_details['docstring'] or 'None'}\n\n"
            f"Source Code:\n"
            f"```python\n{function_details['code']}\n```\n\n"
            f"Requirements:\n"
            f"1. Generate a Google-style docstring\n"
            f"2. Include type hints if present\n"
            f"3. Provide a clear, concise summary\n"
            f"4. Include a changelog entry"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        function_schema = [
            {
                "name": "analyze_and_document_function",
                "description": "Analyzes a Python function and provides structured documentation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Concise description of function purpose and behavior",
                        },
                        "docstring": {
                            "type": "string",
                            "description": "Complete Google-style docstring",
                        },
                        "changelog": {
                            "type": "string",
                            "description": "Documentation change history or 'Initial documentation'",
                        },
                    },
                    "required": ["summary", "docstring", "changelog"],
                },
            }
        ]

        completion = await make_openai_request(
            model_name="gpt-4" if service == "openai" else "gpt-4-32k",
            messages=messages,
            functions=function_schema,
            service=service
        )

        if 'error' in completion:
            logging.error(f"API returned an error: {completion['error']}")
            return create_error_response(function_details["name"], error_message=completion['error']['message'])

        response_message = completion.get('choices', [{}])[0].get('message', {})
        if not response_message:
            raise KeyError("Missing 'choices' or 'message' in response")

        if 'function_call' in response_message:
            function_args = json.loads(response_message['function_call']['arguments'])
        elif 'content' in response_message:
            try:
                function_args = json.loads(response_message['content'])
            except json.JSONDecodeError:
                logging.error("Model did not return valid JSON in content.")
                function_args = {
                    "summary": response_message['content'].strip(),
                    "docstring": "No valid docstring generated.",
                    "changelog": "No changelog available."
                }

        result = {
            "name": function_details["name"],
            "complexity_score": function_details.get("complexity_score", "Unknown"),
            "summary": function_args.get("summary", "No summary available"),
            "docstring": function_args.get("docstring", "No documentation available"),
            "changelog": function_args.get("changelog", "No changelog available."),
        }

        cache_response(function_hash, result)
        return result

    except Exception as e:
        logging.error(f"Error analyzing function {function_details['name']}: {e}")
        sentry_sdk.capture_exception(e)
        return create_error_response(function_details["name"], error_message=str(e))

def format_parameters(params: List[Tuple[str, str]]) -> str:
    """Format function parameters for the prompt."""
    return ", ".join([f"{name}: {ptype}" for name, ptype in params])

if __name__ == "__main__":
    initialize_cache()