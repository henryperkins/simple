import aiohttp
import asyncio
import json
import os
import sentry_sdk
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from core.logger import LoggerSetup
from extract.utils import validate_schema

# Initialize a logger specifically for this module
logger = LoggerSetup.get_logger("api_interaction")

# Load environment variables from .env file
load_dotenv()

# Determine which service to use
use_azure = os.getenv("USE_AZURE", "false").lower() == "true"

# Load environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
azure_api_key = os.getenv("AZURE_API_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")
azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")
azure_model_name = os.getenv("AZURE_MODEL_NAME", "gpt-4o-2024-08-06")
azure_api_version = os.getenv("AZURE_API_VERSION", "2024-08-01-preview")
sentry_dsn = os.getenv("SENTRY_DSN")

# Validate required environment variables based on the service
if use_azure:
    required_vars = {
        "AZURE_API_KEY": azure_api_key,
        "AZURE_ENDPOINT": azure_endpoint,
        "SENTRY_DSN": sentry_dsn
    }
else:
    required_vars = {
        "OPENAI_API_KEY": openai_api_key,
        "SENTRY_DSN": sentry_dsn
    }

for var_name, var_value in required_vars.items():
    if not var_value:
        logger.error(f"{var_name} is not set.")
        raise ValueError(f"{var_name} is not set.")
    else:
        logger.debug(f"{var_name} is set.")

def get_service_headers(service: str) -> dict:
    logger.debug(f"Getting headers for service: {service}")
    headers = {"Content-Type": "application/json"}
    if service == "azure":
        headers["api-key"] = azure_api_key
    elif service == "openai":
        headers["Authorization"] = f"Bearer {openai_api_key}"
    else:
        logger.error(f"Unsupported service: {service}")
        raise ValueError(f"Unsupported service: {service}")
    logger.debug(f"Headers set: {headers}")
    return headers

def get_azure_endpoint() -> str:
    logger.debug(f"Azure endpoint retrieved: {azure_endpoint}")
    return azure_endpoint

async def make_openai_request(
    messages: list, functions: list, service: str, model_name: Optional[str] = None
) -> Dict[str, Any]:
    logger.info(f"Preparing to make request to {service} service")
    headers = get_service_headers(service)
    
    if service == "azure":
        endpoint = f"{get_azure_endpoint()}/openai/deployments/{azure_deployment_name}/chat/completions?api-version={azure_api_version}"
        model_name = azure_model_name
    else:
        endpoint = "https://api.openai.com/v1/chat/completions"

    payload = {
        "model": model_name,
        "messages": messages,
        "functions": functions,
        "function_call": "auto",
    }

    logger.debug(f"Using endpoint: {endpoint}")
    logger.debug(f"Using headers: {headers}")
    logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

    retries = 3
    base_backoff = 2
    
    for attempt in tqdm(range(1, retries + 1), desc="API Request Progress"):
        logger.info(f"Attempt {attempt} of {retries}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint, headers=headers, json=payload, timeout=30
                ) as response:
                    logger.debug(f"Response status: {response.status}")
                    if response.status == 200:
                        result = await response.json()
                        logger.debug(f"Received response: {json.dumps(result, indent=2)}")
                        return result
                    else:
                        error_msg = await response.text()
                        logger.warning(
                            f"Attempt {attempt}: API request failed with status {response.status}: {error_msg}"
                        )
                        sentry_sdk.capture_message(
                            f"Attempt {attempt}: API request failed with status {response.status}: {error_msg}"
                        )
        except aiohttp.ClientError as e:
            logger.error(f"Attempt {attempt}: Client error during API request: {e}")
            sentry_sdk.capture_exception(e)
        except asyncio.TimeoutError:
            logger.error(f"Attempt {attempt}: API request timed out.")
            sentry_sdk.capture_message("API request timed out.")
        except Exception as e:
            logger.error(f"Attempt {attempt}: Unexpected exception during API request: {e}")
            sentry_sdk.capture_exception(e)

        sleep_time = base_backoff ** attempt
        logger.debug(f"Retrying API request in {sleep_time} seconds (Attempt {attempt}/{retries})")
        await asyncio.sleep(sleep_time)

    logger.error("Exceeded maximum retries for API request.")
    return {"error": "Failed to get a successful response from the API."}

async def analyze_function_with_openai(
    function_details: Dict[str, Any], service: str
) -> Dict[str, Any]:
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
                f"Provide a detailed analysis for the following function:\n\n"
                f"{function_details.get('code', '')}"
            ),
        },
    ]

    try:
        # Load the function schema from the JSON file
        function_schema_path = os.path.join(os.path.dirname(__file__), 'function_schema.json')
        logger.debug(f"Loading function schema from {function_schema_path}")
        with open(function_schema_path, 'r', encoding='utf-8') as schema_file:
            function_schema = json.load(schema_file)
        logger.debug("Function schema loaded successfully")

        response = await make_openai_request(
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

        choices = response.get("choices", [])
        if not choices:
            error_msg = "Missing 'choices' in API response."
            logger.error(error_msg)
            raise KeyError(error_msg)

        response_message = choices[0].get("message", {})
        if "function_call" in response_message:
            function_args_str = response_message["function_call"].get("arguments", "{}")
            try:
                function_args = json.loads(function_args_str)
                logger.debug(f"Parsed function arguments: {function_args}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding error in function_call arguments: {e}")
                raise

            return {
                "name": function_name,
                "complexity_score": function_details.get("complexity_score", "Unknown"),
                "summary": function_args.get("summary", ""),
                "docstring": function_args.get("docstring", ""),
                "changelog": function_args.get("changelog", ""),
                "classes": function_args.get("classes", []),
                "functions": function_args.get("functions", []),
                "file_content": function_args.get("file_content", [])
            }

        error_msg = "Missing 'function_call' in API response message."
        logger.error(error_msg)
        raise KeyError(error_msg)

    except (KeyError, TypeError, json.JSONDecodeError) as e:
        logger.error(f"Error processing API response: {e}")
        sentry_sdk.capture_exception(e)
        return {
            "name": function_name,
            "complexity_score": function_details.get("complexity_score", "Unknown"),
            "summary": "Error during analysis.",
            "docstring": "Error: Documentation generation failed.",
            "changelog": "Error: Changelog generation failed.",
        }

    except Exception as e:
        logger.error(f"Unexpected error during function analysis: {e}")
        sentry_sdk.capture_exception(e)
        return {
            "name": function_name,
            "complexity_score": function_details.get("complexity_score", "Unknown"),
            "summary": "Error during analysis.",
            "docstring": "Error: Documentation generation failed.",
            "changelog": "Error: Changelog generation failed.",
        }