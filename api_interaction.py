import aiohttp
import asyncio
import json
from typing import Any, Dict
from config import Config
import sentry_sdk
from logging_utils import setup_logger

# Initialize a logger specifically for this module
logger = setup_logger("api_interaction")

async def make_openai_request(model_name: str, messages: list, functions: list, service: str) -> Dict[str, Any]:
    """Make an asynchronous request to the OpenAI or Azure OpenAI API with retries."""
    headers = Config.get_service_headers(service)
    endpoint = Config.get_azure_endpoint() if service == "azure" else "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "functions": functions,
        "function_call": "auto"
    }
    
    logger.debug(f"Making API request to {endpoint} with payload: {json.dumps(payload, indent=2)}")
    
    retries = 3
    backoff = 2
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.debug(f"Received response: {json.dumps(result, indent=2)}")
                        return result
                    else:
                        error_msg = await response.text()
                        logger.error(f"API request failed with status {response.status}: {error_msg}")
                        sentry_sdk.capture_message(f"API request failed with status {response.status}: {error_msg}")
        except Exception as e:
            logger.error(f"Exception during API request: {e}")
            sentry_sdk.capture_exception(e)
        await asyncio.sleep(backoff ** attempt)
        logger.debug(f"Retrying API request, attempt {attempt + 1}")
    
    logger.error("Exceeded maximum retries for API request.")
    return {"error": "Failed to get a successful response from the API."}

async def analyze_function_with_openai(function_details: Dict[str, Any], service: str) -> Dict[str, Any]:
    """Analyze a function using OpenAI's API and generate documentation."""
    logger.info(f"Analyzing function: {function_details['name']}")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Provide a summary, docstring, and changelog for the following function:\n\n{function_details['code']}"}
    ]
    
    function_schema = {
        "name": "analyze_function",
        "description": "Analyze a Python function and provide a summary, docstring, and changelog.",
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
    
    try:
        response = await make_openai_request(
            model_name=Config.OPENAI_MODEL_NAME,
            messages=messages,
            functions=[function_schema],
            service=service
        )
        
        if 'error' in response:
            logger.error(f"API returned an error: {response['error']}")
            sentry_sdk.capture_message(f"API returned an error: {response['error']}")
            return {"name": function_details["name"], "complexity_score": "Unknown"}
        
        response_message = response.get('choices', [{}])[0].get('message', {})
        if not response_message:
            raise KeyError("Missing 'choices' or 'message' in response")
        
        if 'function_call' in response_message:
            function_args = json.loads(response_message['function_call']['arguments'])
        elif 'content' in response_message:
            try:
                function_args = json.loads(response_message['content'])
            except json.JSONDecodeError:
                logger.error("Model did not return valid JSON in content.")
                function_args = {
                    "summary": response_message['content'].strip(),
                    "docstring": "No valid docstring generated.",
                    "changelog": "No changelog available."
                }
        else:
            function_args = {
                "summary": "No summary available.",
                "docstring": "No docstring available.",
                "changelog": "No changelog available."
            }
        
        result = {
            "name": function_details["name"],
            "complexity_score": function_details.get("complexity_score", "Unknown"),
            "summary": function_args.get("summary", "No summary available."),
            "docstring": function_args.get("docstring", "No docstring available."),
            "changelog": function_args.get("changelog", "No changelog available.")
        }
        
        logger.info(f"Analysis complete for function: {function_details['name']}")
        logger.debug(f"Analysis result: {json.dumps(result, indent=2)}")
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing function {function_details['name']}: {e}")
        sentry_sdk.capture_exception(e)
        return {
            "name": function_details["name"],
            "complexity_score": None,
            "summary": f"Error during analysis: {str(e)}",
            "docstring": "Error: Documentation generation failed",
            "changelog": "Error: Changelog generation failed."
        }
