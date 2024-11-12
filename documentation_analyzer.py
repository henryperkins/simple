# documentation_analyzer.py

import json
from typing import Any, Dict, List, Optional
from core.logger import LoggerSetup
from api_client import APIClient
from response_parser import ClaudeResponseParser
from utils import validate_schema

logger = LoggerSetup.get_logger("documentation_analyzer")

class DocumentationAnalyzer:
    """Handles code analysis and documentation generation."""

    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        logger.info("Initializing DocumentationAnalyzer")

    async def make_api_request(
        self,
        messages: List[Dict[str, str]],
        service: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system_message: Optional[str] = None
    ) -> Any:
        """Make an API request to the specified service."""
        try:
            logger.info(f"Preparing to make API request to {service}")
            logger.debug(f"Messages: {messages}")

            if service == "openai":
                response = await self.api_client.openai_client.chat_completions.create(
                    model=self.api_client.openai_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_message=system_message
                )
            elif service == "azure":
                response = await self.api_client.azure_client.chat_completions.create(
                    deployment_id=self.api_client.azure_deployment,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_message=system_message
                )
            elif service == "claude":
                response = await self.api_client.anthropic_client.messages.create(
                    model="claude-3-opus-20240229",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format="json"
                )
            else:
                raise ValueError(f"Unknown service: {service}")

            logger.debug(f"Received response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error making API request: {e}")
            raise
    async def analyze_function(self, function_details: Dict[str, Any], service: str) -> Dict[str, Any]:
        """Analyze function and generate documentation using specified service."""
        function_name = function_details.get("name", "unknown")

        try:
            logger.info(f"Analyzing function: {function_name} using {service}")

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert code documentation generator."
                },
                {
                    "role": "user",
                    "content": f"Analyze and document this function:\n```python\n{function_details.get('code', '')}\n```"
                }
            ]

            response = await self.make_api_request(messages, service)

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

            parsed_response.setdefault("changelog", [])
            parsed_response.setdefault("classes", [])

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