import aiohttp
import json
import logging
import asyncio
import random
import hashlib
from typing import Dict, Any, List, Tuple
from config import Config
import sentry_sdk
from cache import get_cached_response, cache_response, initialize_cache

def create_error_response(function_name: str, error_message: str) -> Dict[str, Any]:
    """Create an error response dictionary."""
    return {
        "name": function_name,
        "complexity_score": None,
        "summary": f"Error during analysis: {error_message}",
        "docstring": "Error: Documentation generation failed",
        "changelog": "No changelog available due to error."
    }

async def exponential_backoff_with_jitter(coro, max_retries: int = 5, initial_delay: float = 1.0, max_delay: float = 60.0) -> Any:
    """
    Retry a coroutine with exponential backoff and jitter.

    Args:
        coro (coroutine): The coroutine to retry.
        max_retries (int): Maximum number of retries.
        initial_delay (float): Initial delay between retries.
        max_delay (float): Maximum delay between retries.

    Returns:
        Any: The result of the coroutine.

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