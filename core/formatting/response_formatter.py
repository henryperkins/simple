"""Response formatter class."""

import json
from typing import Any, Dict, List, Optional, Union

from core.logger import LoggerSetup


class ResponseFormatter:
    """Formats responses from the AI model into a standardized structure."""

    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """
        Initialize the response formatter.

        :param correlation_id: Optional string for correlation purposes in logging.
        """
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}", correlation_id
        )
        self.correlation_id = correlation_id

    def format_summary_description_response(
        self, response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format a response that contains a summary or description by wrapping it into a
        standardized structure with choices.

        :param response: The raw response dict.
        :return: A dict representing the standardized response.
        """
        formatted = {
            "choices": [{"message": {"content": json.dumps(response)}}],
            "usage": response.get("usage", {}),
        }
        self.logger.debug(
            f"Formatted summary/description response: {formatted}",
            extra={"correlation_id": self.correlation_id},
        )
        return formatted

    def format_function_call_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a response that contains a function call into a standardized structure.

        :param response: The raw response dict containing a "function_call" key.
        :return: A dict with standardized structure focusing on the function call.
        """
        formatted_response = {
            "choices": [{"message": {"function_call": response["function_call"]}}],
            "usage": response.get("usage", {}),
        }
        self.logger.debug(
            f"Formatted function call response: {formatted_response}",
            extra={"correlation_id": self.correlation_id},
        )
        return formatted_response

    def format_tool_calls_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a response that contains tool calls into a standardized structure.

        :param response: The raw response dict containing "tool_calls".
        :return: A dict with standardized structure focusing on the tool calls.
        """
        formatted_response = {
            "choices": [{"message": {"tool_calls": response["tool_calls"]}}],
            "usage": response.get("usage", {}),
        }
        self.logger.debug(
            f"Formatted tool calls response: {formatted_response}",
            extra={"correlation_id": self.correlation_id},
        )
        return formatted_response

    def format_fallback_response(
        self, metadata: Dict[str, Any], error: str = ""
    ) -> Dict[str, Any]:
        """
        Create a fallback response structure when the incoming response is invalid or
        does not match expected formats.

        :param response: The raw invalid response dict.
        :param error: Optional error message describing the issue.
        :return: A standardized fallback response dict.
        """
        self.logger.warning(
            "Response format is invalid, creating fallback.",
            extra={"metadata": metadata, "correlation_id": self.correlation_id},
        )
        fallback_content: Dict[str, Any] = {
            "summary": "Invalid response format",
            "description": "The response did not match the expected structure.",
            "error": error,
            "args": [],
            "returns": {"type": "Any", "description": "No return description provided"},
            "raises": [],
            "complexity": 1,
        }

        fallback_response = {
            "choices": [{"message": {"content": json.dumps(fallback_content)}}],
            "usage": {},
        }

        self.logger.debug(
            f"Formatted fallback response: {fallback_response}",
            extra={"correlation_id": self.correlation_id},
        )
        return fallback_response

    def _standardize_response_format(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize response format to ensure proper structure."""
        try:
            # Case 1: Already in choices format
            if isinstance(response, dict) and "choices" in response:
                return response

            # Case 2: Raw content format 
            if isinstance(response, str):
                try:
                    # Try to parse as JSON first
                    content = json.loads(response)
                    if isinstance(content, dict):
                        return {
                            "choices": [{
                                "message": {
                                    "content": json.dumps(content)
                                }
                            }]
                        }
                except json.JSONDecodeError:
                    # If not JSON, wrap as plain text
                    return {
                        "choices": [{
                            "message": {
                                "content": response
                            }
                        }]
                    }

            # Case 3: Direct content format
            if isinstance(response, dict) and ("summary" in response or "description" in response):
                return {
                    "choices": [{
                        "message": {
                            "content": json.dumps({
                                "summary": response.get("summary", "No summary provided"),
                                "description": response.get("description", "No description provided"),
                                "args": response.get("args", []),
                                "returns": response.get("returns", {"type": "Any", "description": ""}),
                                "raises": response.get("raises", []),
                                "complexity": response.get("complexity", 1)
                            })
                        }
                    }]
                }

            # Case 4: Fallback for unknown format
            self.logger.warning(
                "Unknown response format, creating fallback",
                extra={"correlation_id": self.correlation_id}
            )
            return self.formatter.format_fallback_response(
                {},
                f"Unrecognized response format: {str(response)[:100]}..."
            )

        except Exception as e:
            self.logger.error(
                f"Error standardizing response format: {e}",
                exc_info=True,
                extra={"correlation_id": self.correlation_id}
            )
            return self.formatter.format_fallback_response({}, str(e))
