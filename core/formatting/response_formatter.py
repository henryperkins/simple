"""Response formatter class."""

import json
from typing import Any, Dict, Optional

from core.logger import LoggerSetup


class ResponseFormatter:
    """Formats responses from the AI model."""

    def __init__(self, correlation_id: Optional[str] = None):
        """Initialize the response formatter."""
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}", correlation_id
        )
        self.correlation_id = correlation_id

    def format_summary_description_response(
        self, response: dict[str, Any]
    ) -> dict[str, Any]:
        """Format response with summary or description."""
        return {
            "choices": [{"message": {"content": json.dumps(response)}}],
            "usage": response.get("usage", {}),
        }

    def format_function_call_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Format response with function call."""
        formatted_response = {
            "choices": [{"message": {"function_call": response["function_call"]}}],
            "usage": response.get("usage", {}),
        }
        self.logger.debug(
            f"Formatted function call response to: {formatted_response}",
            extra={"correlation_id": self.correlation_id},
        )
        return formatted_response

    def format_tool_calls_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Format response with tool calls."""
        formatted_response = {
            "choices": [{"message": {"tool_calls": response["tool_calls"]}}],
            "usage": response.get("usage", {}),
        }
        self.logger.debug(
            f"Formatted tool calls response to: {formatted_response}",
            extra={"correlation_id": self.correlation_id},
        )
        return formatted_response

    def format_fallback_response(
        self, response: dict[str, Any], error: str = ""
    ) -> dict[str, Any]:
        """Format a fallback response when the response format is invalid."""
        self.logger.warning(
            "Response format is invalid, creating fallback.",
            extra={"response": response, "correlation_id": self.correlation_id},
        )
        fallback_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "Invalid response format",
                                "description": "The response did not match the expected structure.",
                                "error": error,
                            }
                        )
                    }
                }
            ],
            "usage": {},
        }

        # Ensure 'returns' field exists with a default if missing
        for choice in fallback_response.get("choices", []):
            if "message" in choice and "content" in choice["message"]:
                try:
                    content = json.loads(choice["message"]["content"])
                    if isinstance(content, dict) and "returns" not in content:
                        content["returns"] = {
                            "type": "Any",
                            "description": "No return description provided",
                        }
                        choice["message"]["content"] = json.dumps(content)
                except (json.JSONDecodeError, TypeError) as e:
                    self.logger.error(
                        f"Error formatting fallback response: {e}",
                        extra={"correlation_id": self.correlation_id},
                    )

        self.logger.debug(
            f"Formatted generic response to: {fallback_response}",
            extra={"correlation_id": self.correlation_id},
        )
        return fallback_response

    def standardize_response_format(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize response format to use choices structure."""
        try:
            # Case 1: Already in choices format
            if isinstance(response, dict) and "choices" in response:
                return response

            # Case 2: Direct content format
            if isinstance(response, dict) and (
                "summary" in response or "description" in response
            ):
                # Wrap the content in choices format
                standardized = {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "summary": response.get("summary", ""),
                                        "description": response.get("description", ""),
                                        "args": response.get("args", []),
                                        "returns": response.get(
                                            "returns",
                                            {"type": "Any", "description": ""},
                                        ),
                                        "raises": response.get("raises", []),
                                        "complexity": response.get("complexity", 1),
                                        # Preserve any other fields
                                        **{
                                            k: v
                                            for k, v in response.items()
                                            if k
                                            not in [
                                                "summary",
                                                "description",
                                                "args",
                                                "returns",
                                                "raises",
                                                "complexity",
                                            ]
                                        },
                                    }
                                )
                            }
                        }
                    ],
                    "usage": response.get("usage", {}),
                }
                self.logger.debug(
                    f"Standardized direct format response: {standardized}",
                    extra={"correlation_id": self.correlation_id},
                )
                return standardized

            # Case 3: Fallback for unknown format
            self.logger.warning(
                "Unknown response format, creating fallback",
                extra={"correlation_id": self.correlation_id},
            )
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "Unknown response format",
                                    "description": str(response),
                                    "args": [],
                                    "returns": {"type": "Any", "description": ""},
                                    "raises": [],
                                    "complexity": 1,
                                }
                            )
                        }
                    }
                ],
                "usage": {},
            }

        except Exception as e:
            self.logger.error(
                f"Error standardizing response format: {e}",
                extra={"correlation_id": self.correlation_id},
                exc_info=True,
            )
            return self.format_fallback_response(response, str(e))
