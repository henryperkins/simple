"""
Response parsing service for handling Azure OpenAI API outputs.

This module provides comprehensive parsing and validation of Azure OpenAI API responses,
with support for structured outputs, function calling, and error handling according to
Azure best practices.
"""

import json
import time
from typing import Any, Dict, List

from core.logger import LoggerSetup
from core.types import ParsedResponse
from core.formatting.response_formatter import ResponseFormatter
from core.validation.docstring_validator import DocstringValidator


class ResponseParsingService:
    """
    Centralizes parsing and validation of Azure OpenAI API responses.

    This service handles the complex task of parsing, validating, and transforming
    responses from the Azure OpenAI API into structured data that can be used by
    the rest of the application. It includes comprehensive error handling and
    validation to ensure data quality and consistency.
    """

    def __init__(self, correlation_id: str | None = None) -> None:
        """Initialize the response parsing service with Azure-specific configurations."""
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}", correlation_id
        )
        self.formatter = ResponseFormatter(correlation_id)
        self.docstring_validator = DocstringValidator(correlation_id)
        self.correlation_id = correlation_id

        # Initialize parsing statistics
        self._parsing_stats = {
            "total_processed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "validation_failures": 0,
        }

        self.logger.info("ResponseParsingService initialized for Azure OpenAI")

    def _extract_content(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from various response formats."""
        try:
            self.logger.debug(
                f"Raw response content before extraction: {response}",
                extra={"correlation_id": self.correlation_id},
            )

            # Standardize format
            response = self.formatter.standardize_response_format(response)

            # Extract from choices
            message = response["choices"][0].get("message", {})
            if not message:
                return self.formatter.format_fallback_response(response)

            # Parse content
            if "content" in message:
                try:
                    content = json.loads(message["content"])
                    return self._ensure_required_fields(content)
                except json.JSONDecodeError:
                    return self.formatter.format_fallback_response(message["content"])
            elif "function_call" in message:
                return self._extract_content_from_function_call(
                    message["function_call"], "function_call"
                )
            elif "tool_calls" in message:
                return self._extract_content_from_tool_calls(
                    message["tool_calls"], "tool_calls"
                )

            return self.formatter.format_fallback_response(response)

        except Exception as e:
            self.logger.error(f"Error extracting content: {e}", exc_info=True)
            return self.formatter.format_fallback_response(response, str(e))

    def _ensure_required_fields(self, content: Dict[str, Any] | str) -> Dict[str, Any]:
        """Ensure required fields exist in the content."""
        if isinstance(content, str):
            return {
                "summary": "Content as string",
                "description": content.strip(),
                "args": [],
                "returns": {"type": "Any", "description": ""},
                "raises": [],
                "complexity": 1,
            }

        result = dict(content)
        defaults = {
            "summary": "No summary provided.",
            "description": "No description provided.",
            "args": [],
            "returns": {"type": "Any", "description": ""},
            "raises": [],
            "complexity": 1,
        }

        for key, default in defaults.items():
            if key not in result or not result[key]:
                self.logger.debug(
                    f"Setting default value for field: '{key}'",
                    extra={"correlation_id": self.correlation_id},
                )
                result[key] = default

        return result

    def _extract_content_from_function_call(
        self, function_data: Dict[str, Any], call_type: str
    ) -> Dict[str, Any]:
        """Extract content from a function call."""
        try:
            args_str = function_data.get("arguments", "{}")
            if not args_str:
                return {}
            args_dict = json.loads(args_str)
            if isinstance(args_dict, dict):
                # Validate against the function schema
                is_valid, errors = self._validate_content(args_dict, "function")
                if not is_valid:
                    self.logger.error(
                        f"Function call arguments validation failed: {errors}"
                    )
                    return self.formatter.format_fallback_response(
                        args_dict
                    )  # Return fallback content
                return args_dict
            return args_dict
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to decode {call_type} arguments: {function_data.get('arguments')} - Error: {e}",
                extra={"correlation_id": self.correlation_id},
            )
            return {}

    def _extract_content_from_tool_calls(
        self, tool_calls: List[Dict[str, Any]], call_type: str
    ) -> Dict[str, Any]:
        """Extract content from tool calls."""
        extracted_content = {}
        for tool_call in tool_calls:
            if tool_call.get("type") == "function":
                function_data = tool_call.get("function", {})
                if function_data:
                    try:
                        args_str = function_data.get("arguments", "{}")
                        if not args_str:
                            continue
                        args_dict = json.loads(args_str)
                        if isinstance(args_dict, dict):
                            # Validate against the function schema
                            is_valid, errors = self._validate_content(
                                args_dict, "function"
                            )
                            if not is_valid:
                                self.logger.error(
                                    f"Function call arguments validation failed: {errors}"
                                )
                                return self.formatter.format_fallback_response(
                                    args_dict
                                )  # Return fallback content
                            extracted_content.update(args_dict)
                    except json.JSONDecodeError as e:
                        self.logger.error(
                            f"Failed to decode {call_type} arguments: {function_data.get('arguments')} - Error: {e}",
                            extra={"correlation_id": self.correlation_id},
                        )
        return extracted_content

    def _extract_content_from_direct_content(
        self, content: str
    ) -> Dict[str, Any] | str:
        try:
            # Remove triple backticks and 'json' prefix if present
            if content.startswith("```json") and content.endswith("```"):
                content = content.strip("```").lstrip("json").strip()

            # Attempt to decode the JSON content
            content_dict = json.loads(content)
            if isinstance(content_dict, dict) and (
                "summary" in content_dict or "description" in content_dict
            ):
                return content_dict
            return content
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to decode content from: {content} - Error: {e}",
                extra={"correlation_id": self.correlation_id},
            )
            return {"content": content} if isinstance(content, str) else {}

    async def parse_response(
        self,
        response: Dict[str, Any],
        expected_format: str,
        validate_schema: bool = True,
    ) -> ParsedResponse:
        """Parse the response from the AI model."""
        start_time = time.time()
        errors = []
        metadata = {}

        self.logger.debug(
            f"Raw AI response before parsing: {response}",
            extra={"correlation_id": self.correlation_id},
        )
        try:
            if response is None:
                self.logger.error(
                    "Response is None, creating fallback",
                    extra={"correlation_id": self.correlation_id},
                )
                content = self.formatter.format_fallback_response(
                    error="Response is None"
                )
                errors.append("Response is None")
                return ParsedResponse(
                    content=content,
                    format_type=expected_format,
                    parsing_time=time.time() - start_time,
                    validation_success=False,
                    errors=errors,
                    metadata=metadata,
                )

            content = self._extract_content(response)

            if validate_schema:
                if expected_format == "docstring":
                    is_valid, schema_errors = (
                        self.docstring_validator.validate_docstring(content)
                    )
                    if not is_valid:
                        errors.extend(schema_errors)
                        return ParsedResponse(
                            content=content,
                            format_type=expected_format,
                            parsing_time=time.time() - start_time,
                            validation_success=False,
                            errors=errors,
                            metadata=metadata
                        )
                elif expected_format == "function":
                    is_valid, schema_errors = self._validate_content(
                        content, "function"
                    )
                    if not is_valid:
                        errors.extend(schema_errors)
                        return ParsedResponse(
                            content=content,
                            format_type=expected_format,
                            parsing_time=time.time() - start_time,
                            validation_success=False,
                            errors=errors,
                            metadata=metadata,
                            schema_errors=schema_errors,
                        )

            return ParsedResponse(
                content=content,
                format_type=expected_format,
                parsing_time=time.time() - start_time,
                validation_success=not errors,
                errors=errors,
                metadata=metadata
            )
        except Exception as e:
            self.logger.error(
                f"Unexpected error during parsing: {e}",
                exc_info=True,
                extra={"correlation_id": self.correlation_id},
            )
            errors.append(f"Unexpected error: {e}")
            content = self.formatter.format_fallback_response(response, str(e))
            return ParsedResponse(
                content=content,
                format_type=expected_format,
                parsing_time=time.time() - start_time,
                validation_success=False,
                errors=errors,
                metadata=metadata,
                schema_errors=[],
            )
