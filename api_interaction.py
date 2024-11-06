import aiohttp
import asyncio
import json
from typing import Any, Dict
from config import Config
import sentry_sdk
from logging_utils import setup_logger

# Initialize a logger specifically for this module
logger = setup_logger("api_interaction")

async def make_openai_request(
    model_name: str, messages: list, functions: list, service: str
) -> Dict[str, Any]:
    """
    Make an asynchronous request to the OpenAI or Azure OpenAI API with retries.

    Args:
        model_name (str): The name of the model to use.
        messages (list): The list of messages to send to the API.
        functions (list): The list of functions to include in the request.
        service (str): The service to use ('azure' or 'openai').

    Returns:
        Dict[str, Any]: The API response as a dictionary.
    """
    headers = Config.get_service_headers(service)
    endpoint = (
        Config.get_azure_endpoint()
        if service == "azure"
        else "https://api.openai.com/v1/chat/completions"
    )
    payload = {
        "model": model_name,
        "messages": messages,
        "functions": functions,
        "function_call": "auto",
    }

    logger.debug(f"Preparing API request to {endpoint}")

    retries = 3
    backoff = 2  # Exponential backoff factor
    for attempt in range(1, retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint, headers=headers, json=payload, timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.debug(f"Received response: {json.dumps(result, indent=2)}")
                        return result
                    else:
                        error_msg = await response.text()
                        logger.warning(
                            f"API request failed with status {response.status}: {error_msg}"
                        )
                        sentry_sdk.capture_message(
                            f"API request failed with status {response.status}: {error_msg}"
                        )
        except aiohttp.ClientError as e:
            logger.warning(f"HTTP error during API request: {e}")
            sentry_sdk.capture_exception(e)
        except asyncio.TimeoutError:
            logger.warning("API request timed out.")
            sentry_sdk.capture_message("API request timed out.")
        except Exception as e:
            logger.error(f"Unexpected exception during API request: {e}")
            sentry_sdk.capture_exception(e)

        # Implement exponential backoff with jitter
        sleep_time = backoff ** attempt
        logger.debug(f"Retrying API request in {sleep_time} seconds (Attempt {attempt}/{retries})")
        await asyncio.sleep(sleep_time)

    logger.error("Exceeded maximum retries for API request.")
    return {"error": "Failed to get a successful response from the API."}

async def analyze_function_with_openai(
    function_details: Dict[str, Any], service: str
) -> Dict[str, Any]:
    """
    Analyze a function using OpenAI's API and generate documentation.

    Args:
        function_details (Dict[str, Any]): Details of the function to analyze.
        service (str): The AI service to use ('azure' or 'openai').

    Returns:
        Dict[str, Any]: Analysis results including summary, docstring, and changelog.
    """
    function_name = function_details.get("name", "unknown")
    logger.info(f"Analyzing function: {function_name}")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates documentation.",
        },
        {
            "role": "user",
            "content": (
                f"Provide a summary, docstring, and changelog for the following function:\n\n"
                f"{function_details.get('code', '')}"
            ),
        },
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
            service=service,
        )

        if "error" in response:
            logger.error(f"API returned an error: {response['error']}")
            sentry_sdk.capture_message(f"API returned an error: {response['error']}")
            return {
                "name": function_name,
                "complexity_score": function_details.get("complexity_score", "Unknown"),
                "summary": "Error during analysis.",
                "docstring": "Error: Documentation generation failed.",
                "changelog": "Error: Changelog generation failed.",
            }

        response_message = response.get("choices", [{}])[0].get("message", {})
        if not response_message:
            error_msg = "Missing 'choices' or 'message' in API response."
            logger.error(error_msg)
            raise KeyError(error_msg)

        if "function_call" in response_message:
            function_args_str = response_message["function_call"].get("arguments", "{}")
            try:
                function_args = json.loads(function_args_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding error in function_call arguments: {e}")
                function_args = {}
        elif "content" in response_message:
            content = response_message["content"]
            try:
                function_args = json.loads(content)
            except json.JSONDecodeError:
                logger.warning("Model did not return valid JSON in content.")
                function_args = {
                    "summary": content.strip(),
                    "docstring": "No valid docstring generated.",
                    "changelog": "No changelog available.",
                }
        else:
            logger.warning("No function_call or content in response message.")
            function_args = {
                "summary": "No summary available.",
                "docstring": "No docstring available.",
                "changelog": "No changelog available.",
            }

        result = {
            "name": function_name,
            "complexity_score": function_details.get("complexity_score", "Unknown"),
            "summary": function_args.get("summary", "No summary available."),
            "docstring": function_args.get("docstring", "No docstring available."),
            "changelog": function_args.get("changelog", "No changelog available."),
        }

        logger.info(f"Analysis complete for function: {function_name}")
        logger.debug(f"Analysis result: {json.dumps(result, indent=2)}")

        return result

    except (KeyError, TypeError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing API response for function {function_name}: {e}")
        sentry_sdk.capture_exception(e)
        return {
            "name": function_name,
            "complexity_score": function_details.get("complexity_score", "Unknown"),
            "summary": "Error during analysis.",
            "docstring": "Error: Documentation generation failed.",
            "changelog": "Error: Changelog generation failed.",
        }
    except Exception as e:
        logger.error(f"Unexpected error analyzing function {function_name}: {e}")
        sentry_sdk.capture_exception(e)
        return {
            "name": function_name,
            "complexity_score": function_details.get("complexity_score", "Unknown"),
            "summary": f"Error during analysis: {str(e)}",
            "docstring": "Error: Documentation generation failed.",
            "changelog": "Error: Changelog generation failed.",
        }