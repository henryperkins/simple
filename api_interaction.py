# api_interaction.py
import openai
import aiohttp
import json
import logging
import hashlib
import asyncio
import sentry_sdk
from config import get_service_headers, get_azure_endpoint, OPENAI_API_KEY
from monitoring import capture_openai_error
from typing import Dict, Any
from cache import prompt_cache  # Absolute import
from file_processing import exponential_backoff_with_jitter

# Initialize a cache dictionary
prompt_cache = {}
    
def get_cached_response(prompt_key):
    """Retrieve a cached response if available.

    Args:
        prompt_key (str): The unique key for the prompt.

    Returns:
        dict or None: The cached response if available, otherwise None.
    """
    return prompt_cache.get(prompt_key)

def cache_response(prompt_key, response):
    """Cache the response for a given prompt key.

    Args:
        prompt_key (str): The unique key for the prompt.
        response (dict): The response to cache.
    """
    prompt_cache[prompt_key] = response

async def make_openai_request(model_name, messages, functions, service):
    """Make a request to the OpenAI or Azure OpenAI API.

    Args:
        model_name (str): The name of the model to use.
        messages (list): The messages to send to the API.
        functions (list): The function schema for the API request.
        service (str): The service to use ('openai' or 'azure').

    Returns:
        dict: The API response.

    Raises:
        Exception: If the API request fails.
    """
    try:
        with sentry_sdk.start_span(op="openai_request", description=f"{service} API call"):
            if service == "openai":
                openai.api_key = OPENAI_API_KEY
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
                endpoint = get_azure_endpoint()
                headers = get_service_headers(service)
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
            return await exponential_backoff_with_jitter(make_request)
    except Exception as e:
        capture_openai_error(e, {
            "model": model_name,
            "service": service,
            "message_count": len(messages)
        })
        raise

async def analyze_function_with_openai(function_details, service):
    """Analyze a function using OpenAI's API and generate documentation.

    Args:
        function_details (dict): Details of the function to analyze.
        service (str): The service to use ('openai' or 'azure').

    Returns:
        dict: The analysis results, including summary, docstring, and changelog.
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