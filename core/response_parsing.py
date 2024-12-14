"""
Response parsing service with consistent error handling and validation.

This module provides functionality for parsing AI responses, validating 
them against specified schemas, and managing parsing statistics.
"""

import json
import os
from typing import Any, TypeVar, TypedDict, cast

from jsonschema import validate, ValidationError
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.docstring_processor import DocstringProcessor
from core.types import ParsedResponse

# Set up the base logger
base_logger = LoggerSetup.get_logger(__name__)

# Type variables for better type hinting
T = TypeVar('T')

class MessageDict(TypedDict, total=False):
    tool_calls: list[dict[str, Any]]
    function_call: dict[str, Any]
    content: str

class ChoiceDict(TypedDict):
    message: MessageDict

class ResponseDict(TypedDict, total=False):
    choices: list[ChoiceDict]
    usage: dict[str, int]

class ContentType(TypedDict, total=False):
    summary: str
    description: str
    args: list[dict[str, Any]]
    returns: dict[str, str]
    raises: list[dict[str, str]]
    complexity: int

class ResponseParsingService:
    """Centralized service for parsing and validating AI responses."""

    def __init__(self, correlation_id: str | None = None) -> None:
        """Initialize the response parsing service."""
        self.logger = CorrelationLoggerAdapter(base_logger)
        self.docstring_processor = DocstringProcessor()
        self.docstring_schema = self._load_schema("docstring_schema.json")
        self.function_schema = self._load_schema("function_tools_schema.json")
        self._parsing_stats = {
            "total_processed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "validation_failures": 0,
        }

    def _load_schema(self, schema_name: str) -> dict[str, Any]:
        """Load a JSON schema for validation."""
        try:
            schema_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "schemas", schema_name
            )
            with open(schema_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading schema {schema_name}: {e}")
            return {}

    async def parse_response(
        self,
        response: dict[str, Any],
        expected_format: str = "docstring",
        validate_schema: bool = True,
    ) -> ParsedResponse:
        """Parse the AI model response and return a ParsedResponse object."""
        try:
            # Handle direct response format
            if "summary" in response or "description" in response:
                content = response
                usage: dict[str, int] = {}
            else:
                # Extract content from the response
                content = self._extract_content(response)
                if content is None:
                    fallback = self._create_fallback_response()
                    return ParsedResponse(
                        content=fallback,
                        format_type=expected_format,
                        parsing_time=0.0,
                        validation_success=False,
                        errors=["Failed to extract content from response"],
                        metadata={},
                    )
                usage = cast(dict[str, int], response.get("usage", {}))

            # Ensure required fields exist
            content = self._ensure_required_fields(content)

            # Get parsing time safely
            total_ms = usage.get("total_ms", 0)
            parsing_time = float(total_ms)

            # Create a new dictionary with only the required fields
            filtered_content = {
                "summary": content.get("summary", ""),
                "description": content.get("description", ""),
                "args": content.get("args", []),
                "returns": content.get("returns", {"type": "Any", "description": ""}),
                "raises": content.get("raises", []),
                "complexity": content.get("complexity", 1),
            }

            # Validate the content if required
            is_valid = True
            validation_errors: list[str] = []
            if validate_schema:
                is_valid, validation_errors = self._validate_content(
                    filtered_content, expected_format
                )

            # Return the parsed response
            return ParsedResponse(
                content=filtered_content,
                format_type=expected_format,
                parsing_time=parsing_time,
                validation_success=is_valid,
                errors=validation_errors,
                metadata={},
            )

        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            fallback = self._create_fallback_response()
            return ParsedResponse(
                content=fallback,
                format_type=expected_format,
                parsing_time=0.0,
                validation_success=False,
                errors=[f"Failed to parse response: {e}"],
                metadata={},
            )

    def _extract_content(self, response: dict[str, Any]) -> dict[str, Any] | None:
        """Extract content from various response formats."""
        try:
            # Handle choices format
            if "choices" in response and response["choices"]:
                message = response["choices"][0].get("message", {})
                
                # Try function call
                if "function_call" in message:
                    try:
                        args_str = message["function_call"].get("arguments", "{}")
                        args_dict = json.loads(args_str)
                        # Return the parsed arguments directly if they match our expected format
                        if isinstance(args_dict, dict) and ("summary" in args_dict or "description" in args_dict):
                            return cast(dict[str, Any], args_dict)
                        # Otherwise wrap it in a summary field
                        return {"summary": json.dumps(args_dict)}
                    except json.JSONDecodeError:
                        self.logger.warning("Failed to parse function call arguments")
                
                # Try tool calls
                if "tool_calls" in message and message["tool_calls"]:
                    tool_call = message["tool_calls"][0]
                    if "function" in tool_call:
                        try:
                            args_str = tool_call["function"].get("arguments", "{}")
                            args_dict = json.loads(args_str)
                            # Return the parsed arguments directly if they match our expected format
                            if isinstance(args_dict, dict) and ("summary" in args_dict or "description" in args_dict):
                                return cast(dict[str, Any], args_dict)
                            # Otherwise wrap it in a summary field
                            return {"summary": json.dumps(args_dict)}
                        except json.JSONDecodeError:
                            self.logger.warning("Failed to parse tool call arguments")
                
                # Try direct content
                if "content" in message:
                    try:
                        # Try to parse content as JSON first
                        content_dict = json.loads(message["content"])
                        if isinstance(content_dict, dict) and ("summary" in content_dict or "description" in content_dict):
                            return cast(dict[str, Any], content_dict)
                    except json.JSONDecodeError:
                        # If not JSON, use as plain text summary
                        return {"summary": message["content"]}
            
            # Try direct response format
            if "summary" in response or "description" in response:
                return dict(response)
            
            return None
        except Exception as e:
            self.logger.error(f"Error extracting content: {e}")
            return None

    def _ensure_required_fields(self, content: dict[str, Any]) -> dict[str, Any]:
        """Ensure all required fields exist in the content."""
        # Create a new dictionary with the content
        result = dict(content) if hasattr(content, "items") else {"summary": str(content)}

        # Ensure required fields exist
        defaults: dict[str, Any] = {
            "summary": "",
            "description": "",
            "args": [],
            "returns": {"type": "Any", "description": ""},
            "raises": [],
            "complexity": 1
        }

        # Update with defaults for missing fields
        for key, default in defaults.items():
            if key not in result:
                result[key] = default

        return result

    def _create_fallback_response(self) -> dict[str, Any]:
        """Create a fallback response when parsing fails."""
        self.logger.info("Creating fallback response due to parsing failure")
        return {
            "summary": "Documentation generation failed",
            "description": "Unable to generate documentation due to parsing error",
            "args": [],
            "returns": {"type": "Any", "description": "Return value not documented"},
            "raises": [],
            "complexity": 1,
        }

    def _validate_content(
        self, content: dict[str, Any], format_type: str
    ) -> tuple[bool, list[str]]:
        """Validate the content against the appropriate schema."""
        validation_errors: list[str] = []
        try:
            if format_type == "docstring":
                if not self.docstring_schema:
                    validation_errors.append("Docstring schema not loaded")
                    return False, validation_errors
                
                schema = self.docstring_schema.get("schema", {})
                if not schema:
                    validation_errors.append("Invalid docstring schema structure")
                    return False, validation_errors
                
                validate(instance=content, schema=schema)
            elif format_type == "function":
                if not self.function_schema:
                    validation_errors.append("Function schema not loaded")
                    return False, validation_errors
                
                schema = self.function_schema.get("function", {}).get("parameters", {})
                if not schema:
                    validation_errors.append("Invalid function schema structure")
                    return False, validation_errors
                
                validate(instance=content, schema=schema)
            
            return True, validation_errors
        except ValidationError as e:
            validation_errors.append(str(e))
            return False, validation_errors
        except Exception as e:
            validation_errors.append(f"Unexpected validation error: {str(e)}")
            return False, validation_errors
