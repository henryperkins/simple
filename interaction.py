# interaction.py

import asyncio
import json
from typing import Any, Dict, List

from core.logger import LoggerSetup
from api_client import APIClient
from documentation_analyzer import DocumentationAnalyzer
from response_parser import ClaudeResponseParser
from utils import validate_schema

logger = LoggerSetup.get_logger("main")

# Initialize default API client
default_api_client = APIClient()


async def analyze_function_with_openai(
    function_details: Dict[str, Any],
    service: str
) -> Dict[str, Any]:
    """
    Analyze function and generate documentation using specified service.

    Args:
        function_details: Dictionary containing function information
        service: Service to use ("azure", "openai", or "claude")

    Returns:
        Dictionary containing analysis results
    """
    function_name = function_details.get("name", "unknown")

    try:
        api_client = APIClient()
        analyzer = DocumentationAnalyzer(api_client)

        logger.info(f"Analyzing function: {function_name} using {service}")

		messages = [
		    {
		        "role": "system",
		        "content": "You are an expert code documentation generator."
		    },
		    {
		        "role": "user",
		        "content": f"""Analyze and document this function:
		```python
		{function_details.get('code', '')}
		```"""
		    }
		]

        response = await analyzer.make_api_request(messages, service)

        # Handle different response formats
        if service == "claude":
            content = response["completion"]
            parsed_response = ClaudeResponseParser.parse_function_analysis(content)
        else:
            tool_calls = response.get('choices', [{}])[0].get('message', {}).get('tool_calls', [])
            if tool_calls and tool_calls[0].get('function'):
                function_args = json.loads(tool_calls[0]['function']['arguments'])
                parsed_response = function_args
            else:
                logger.warning("No tool calls found in response")
                return ClaudeResponseParser.get_default_response()

        # Ensure changelog and classes are included
        parsed_response.setdefault("changelog", [])
        parsed_response.setdefault("classes", [])

        # Validate response against schema
        try:
            validate_schema(parsed_response)
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return ClaudeResponseParser.get_default_response()

        logger.info(f"Successfully analyzed function: {function_name}")
        return {
            "name": function_name,
            "complexity_score": function_details.get("complexity_score", "Unknown"),
            "summary": parsed_response.get("summary", ""),
            "docstring": parsed_response.get("docstring", ""),
            "params": parsed_response.get("params", []),
            "returns": parsed_response.get("returns", {"type": "None", "description": ""}),
            "examples": parsed_response.get("examples", []),
            "classes": parsed_response.get("classes", []),
            "changelog": parsed_response.get("changelog")
        }

    except Exception as e:
        logger.error(f"Error analyzing function {function_name}: {e}")
        return ClaudeResponseParser.get_default_response()


class AsyncAPIClient:
    """
    Asynchronous API client for batch processing.
    Useful for processing multiple functions concurrently.
    """

    def __init__(self, service: str):
        self.service = service
        self.api_client = APIClient()
        self.analyzer = DocumentationAnalyzer(self.api_client)
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

    async def process_batch(
        self,
        functions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of functions concurrently.

        Args:
            functions: List of function details to process

        Returns:
            List of documentation results
        """

        async def process_with_semaphore(func: Dict[str, Any]) -> Dict[str, Any]:
            async with self.semaphore:
                return await analyze_function_with_openai(func, self.service)

        tasks = [process_with_semaphore(func) for func in functions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
                processed_results.append(ClaudeResponseParser.get_default_response())
            else:
                processed_results.append(result)

        return processed_results


if __name__ == "__main__":
    # Example usage
    function_details = {
        "name": "example_function",
        "code": "def example_function(): pass"
    }
    results = asyncio.run(analyze_function_with_openai(function_details, service="openai"))
    print(results)