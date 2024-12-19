"""
Response parsing service for handling Azure OpenAI API outputs.

This module provides comprehensive parsing and validation of Azure OpenAI API responses,
with support for structured outputs, function calling, and error handling according to
Azure best practices.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TypedDict
import logging
from dataclasses import dataclass, field

from core.logger import LoggerSetup
from core.types import ParsedResponse
from core.formatting.response_formatter import ResponseFormatter
from core.validation.docstring_validator import DocstringValidator
from core.validation.schema_validator import SchemaValidator


@dataclass
class ParsedResponse:
    """Data class to hold the parsed response."""
    content: Any
    format_type: str
    parsing_time: float = 0.0
    validation_success: bool = True
    errors: List[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DocstringSchema(TypedDict):
    """Type definition for docstring schema."""
    summary: str
    description: str
    args: list[dict[str, Any]]
    returns: dict[str, str]
    raises: list[dict[str, str]]
    complexity: int
    metadata: dict[str, Any]

FALLBACK_SCHEMA = DocstringSchema(
    summary="No summary available",
    description="No description available",
    args=[],
    returns={"type": "Any", "description": "No return description provided"},
    raises=[],
    complexity=1,
    metadata={}
)


class ResponseParser:
    """Parses and validates AI responses."""

    def __init__(self, formatter: ResponseFormatter, docstring_validator: DocstringValidator, logger=logging.getLogger(__name__), correlation_id: Optional[str] = None):
        self.formatter = formatter
        self.logger = logger
        self.correlation_id = correlation_id
        self.docstring_validator = docstring_validator

    async def parse_response(
        self,
        response: Dict[str, Any],
        expected_format: str,
        validate_schema: bool = True,
        metadata: Optional[dict[str, Any]] = None
    ) -> ParsedResponse:
        """
        Parses and validates an AI response.

        Args:
            response (dict[str, Any]): The raw response from the AI model.
            expected_format (str): The expected format of the response (e.g., "docstring").
            validate_schema (bool, optional): Whether to validate the response against a schema. Defaults to True.
            metadata (dict[str, Any], optional): Additional metadata to include in the response. Defaults to None.

        Returns:
            ParsedResponse: A ParsedResponse object containing the parsed content, validation status, and any errors.
        """
        start_time = time.time()
        if metadata is None:
            metadata = {}
        self.logger.debug(f"Raw AI response before parsing: {response}",
                          extra={"correlation_id": self.correlation_id})

        try:
            # Validate basic response structure
            validated_response = self._validate_response_structure(response, expected_format, start_time, metadata)
            if validated_response.errors:
                return validated_response

            # Parse message content
            parsed_content = await self._parse_message_content(
                response["choices"][0]["message"].get("content", ""),
                expected_format,
                validate_schema,
                start_time,
                metadata
            )
            
            return self._create_response(
                content=parsed_content.content,
                format_type=expected_format,
                start_time=start_time,
                success=parsed_content.validation_success,
                errors=parsed_content.errors,
                metadata=metadata
            )

        except Exception as e:
            return self._handle_unexpected_error(e, expected_format, start_time, metadata)

    def _create_response(
        self,
        content: Any,
        format_type: str,
        start_time: float,
        success: bool,
        errors: List[str],
        metadata: dict[str, Any]
    ) -> ParsedResponse:
        """Creates a standardized ParsedResponse object."""
        return ParsedResponse(
            content=content,
            format_type=format_type,
            parsing_time=time.time() - start_time,
            validation_success=success,
            errors=errors,
            metadata=metadata
        )

    def _create_error_response(
        self,
        error: str,
        expected_format: str,
        start_time: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> ParsedResponse:
        """Creates a standardized error response."""
        return ParsedResponse(
            content=self.formatter.format_fallback_response(metadata if metadata else {}, error=error),
            format_type=expected_format,
            parsing_time=time.time() - start_time if start_time else 0.0,
            validation_success=False,
            errors=[error],
            metadata=metadata if metadata else {}
        )

    def _validate_response_structure(
        self, 
        response: Dict[str, Any],
        expected_format: str,
        start_time: float,
        metadata: dict[str, Any]
    ) -> ParsedResponse:
        """Validates the basic structure of the AI response."""
        if response is None:
            return self._create_error_response("Response is None", expected_format, start_time, metadata)

        if "choices" not in response or not response["choices"]:
            return self._create_error_response("No choices in response", expected_format, start_time, metadata)

        if not response["choices"][0].get("message"):
            return self._create_error_response("No message in response", expected_format, start_time, metadata)

        if "content" not in response["choices"][0]["message"]:
             return self._create_error_response("No content field in message", expected_format, start_time, metadata)

        return ParsedResponse(content={}, errors=[], format_type=expected_format, metadata=metadata)

    async def _parse_message_content(
        self,
        content: str,
        expected_format: str,
        validate_schema: bool,
        start_time: float,
        metadata: dict[str, Any]
    ) -> ParsedResponse:
        """Parses the message content based on the expected format."""
        if not content:
            return ParsedResponse(content={}, format_type=expected_format, validation_success=False, errors=["Content is empty"], metadata=metadata)

        try:
            if expected_format == "docstring":
                parsed_content = self._parse_docstring_content(content, validate_schema, start_time, metadata)
            elif expected_format == "text":
                parsed_content = ParsedResponse(content=content, format_type=expected_format, errors=[], metadata=metadata)
            else:
                return ParsedResponse(content={}, format_type=expected_format, validation_success=False, errors=[f"Unsupported format: {expected_format}"], metadata=metadata)

            return parsed_content
        except json.JSONDecodeError as e:
            return ParsedResponse(content={}, format_type=expected_format, validation_success=False, errors=[f"JSONDecodeError: {e}"], metadata=metadata)
        except Exception as e:
            return ParsedResponse(content={}, format_type=expected_format, validation_success=False, errors=[f"Unexpected error during parsing: {e}"], metadata=metadata)

    def _parse_docstring_content(self, content: str, validate_schema: bool, start_time: float, metadata: dict[str, Any]) -> ParsedResponse:
        """Parses docstring content and optionally validates it against a schema."""
        try:
            docstring_data = json.loads(content)
            if validate_schema:
                validation_result = self._validate_docstring_schema(docstring_data, start_time, metadata)
                if not validation_result.validation_success:
                    return validation_result
            return ParsedResponse(content=docstring_data, format_type="docstring", errors=[], metadata=metadata)
        except json.JSONDecodeError as e:
            self.logger.warning(
                    "Invalid JSON received from AI response, using fallback.",
                    extra={"error": str(e), "raw_response": content},
                )
            fallback_data = FALLBACK_SCHEMA.copy()
            fallback_data["metadata"] = metadata
            return ParsedResponse(
                content=fallback_data,
                format_type="docstring",
                parsing_time=time.time() - start_time,
                validation_success=False,
                errors=[f"Invalid JSON: {e}"],
                metadata=metadata,
            )

    def _validate_docstring_schema(self, content: dict[str, Any], start_time: float, metadata: dict[str, Any]) -> ParsedResponse:
        """Validates the parsed content against a predefined docstring schema."""
        # Use your docstring validator to validate the extracted dictionary
        is_valid, schema_errors = self.docstring_validator.validate_docstring(content)
        if not is_valid:
            return ParsedResponse(
                content=content,
                format_type="docstring",
                parsing_time=time.time() - start_time,
                validation_success=False,
                errors=schema_errors,
                metadata=metadata
            )
        return ParsedResponse(content=content, format_type="docstring", errors=[], metadata=metadata)

    def _handle_unexpected_error(
        self,
        error: Exception,
        expected_format: str,
        start_time: float,
        metadata: dict[str, Any]
    ) -> ParsedResponse:
        """Handles unexpected errors during parsing."""
        self.logger.error(
            f"Unexpected error during parsing: {error}",
            exc_info=True,
            extra={"correlation_id": self.correlation_id},
        )
        return self._create_error_response(
            error=f"Unexpected error: {error}",
            expected_format=expected_format,
            start_time=start_time,
            metadata=metadata
        )


class ResponseParsingService:
    """
    Centralizes parsing and validation of Azure OpenAI API responses.

    This service handles the complex task of parsing, validating, and transforming
    responses from the Azure OpenAI API into structured data that can be used by
    the rest of the application. It includes comprehensive error handling and
    validation to ensure data quality and consistency.
    """

    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """Initialize the response parsing service with Azure-specific configurations."""
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}", correlation_id
        )
        self.formatter = ResponseFormatter(correlation_id)
        self.docstring_validator = DocstringValidator(correlation_id)
        self.correlation_id = correlation_id
        self.response_parser = ResponseParser(formatter=self.formatter, docstring_validator=self.docstring_validator, logger=self.logger, correlation_id=self.correlation_id)


        self._parsing_stats: Dict[str, int] = {
            "total_processed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "validation_failures": 0,
        }

        self.logger.info("ResponseParsingService initialized for Azure OpenAI")

    def _validate_content(
        self, content: Dict[str, Any], schema_type: str
    ) -> Tuple[bool, List[str]]:
        """Validate content against a schema."""
        schema_paths = {
            "function": "core/schemas/function_tools_schema.json",
            "docstring": "core/schemas/docstring_schema.json",
        }

        schema_path = schema_paths.get(schema_type, None)
        if schema_path is None:
            return False, [f"Unknown schema type: {schema_type}"]

        try:
            schema_file = Path(schema_path)
            with schema_file.open("r", encoding="utf-8") as f:
                schema = json.load(f)

            schema_validator = SchemaValidator(
                logger_name=f"{__name__}.{self.__class__.__name__}",
                correlation_id=self.correlation_id,
            )
            return schema_validator.validate_schema(content, schema)
        except FileNotFoundError:
            return False, [f"Schema file not found: {schema_path}"]
        except json.JSONDecodeError as e:
            self.logger.warning("Invalid JSON received from AI response, using fallback.", extra={"error": str(e), "raw_response": content})
            return False, [f"Error decoding schema JSON: {e}"]
        except Exception as e:
            return False, [f"Unexpected error: {e}"]

    def _extract_content(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from various response formats."""
        self.logger.debug(
            f"Raw response content before extraction: {response}",
            extra={"correlation_id": self.correlation_id},
        )

        # Standardize format
        response = self.formatter.standardize_response_format(response)

        # Extract from choices
        if "choices" not in response or not response["choices"]:
            return self.formatter.format_fallback_response({}, "No choices found")

        message = response["choices"][0].get("message", {})
        if not message:
            return self.formatter.format_fallback_response({}, "No message found")

        # Parse content
        if "content" in message:
            try:
                content = json.loads(message["content"])
                return self._ensure_required_fields(content)
            except json.JSONDecodeError:
                # Content not valid JSON, fallback to raw content
                return self.formatter.format_fallback_response(
                    {}, "Invalid JSON in content"
                )
        elif "function_call" in message:
            return self._extract_content_from_function_call(
                message["function_call"], "function_call"
            )
        elif "tool_calls" in message:
            tool_calls = message["tool_calls"]
            if isinstance(tool_calls, list):
                return self._extract_content_from_tool_calls(tool_calls, "tool_calls")

        return self.formatter.format_fallback_response({}, "Unrecognized format")

    def _ensure_required_fields(
        self, content: Union[Dict[str, Any], str]
    ) -> Dict[str, Any]:
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
        defaults: Dict[str, Union[str, Dict[str, str], int, List[Any]]] = {
            "summary": "No summary provided. Ensure this field is populated.",
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
        args_str = function_data.get("arguments", "")
        if not args_str:
            return {}

        try:
            args_dict: Dict[str, Any] = json.loads(args_str)
            if isinstance(args_dict, dict):
                # Validate against the function schema
                is_valid, errors = self._validate_content(args_dict, "function")
                if not is_valid:
                    self.logger.error(
                        f"Function call arguments validation failed: {errors}"
                    )
                    return self.formatter.format_fallback_response(
                        {}, "Invalid function schema"
                    )
                return args_dict
            return args_dict
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to decode {call_type} arguments: {function_data.get('arguments')}"
                f" - Error: {e}",
                extra={"correlation_id": self.correlation_id},
            )
            return {}

    def _extract_content_from_tool_calls(
        self, tool_calls: List[Dict[str, Any]], call_type: str
    ) -> Dict[str, Any]:
        """Extract content from tool calls."""
        extracted_content: Dict[str, Any] = {}
        for tool_call in tool_calls:
            if tool_call.get("type") == "function":
                function_data = tool_call.get("function", {})
                if function_data:
                    args_str = function_data.get("arguments", "")
                    if not args_str:
                        continue
                    try:
                        args_dict: Dict[str, Any] = json.loads(args_str)
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
                                    {}, "Invalid tool function schema"
                                )
                            extracted_content.update(args_dict)
                    except json.JSONDecodeError as e:
                        self.logger.error(
                            f"Failed to decode {call_type} arguments: "
                            f"{function_data.get('arguments')} - Error: {e}",
                            extra={"correlation_id": self.correlation_id},
                        )
        return extracted_content

    def _extract_content_from_direct_content(self, content: str) -> Union[Dict[str, Any], str]:
        """Decode content from direct content."""
        cleaned_content = content.strip()
        if cleaned_content.startswith("```json") and cleaned_content.endswith("```"):
            cleaned_content = cleaned_content.strip("```").lstrip("json").strip()

        try:
            content_dict = json.loads(cleaned_content)
            if isinstance(content_dict, dict) and (
                "summary" in content_dict or "description" in content_dict
            ):
                return content_dict
            return cleaned_content
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to decode content from: {cleaned_content} - Error: {e}",
                extra={"correlation_id": self.correlation_id},
            )
            return {"content": cleaned_content}

    async def parse_response(
        self,
        response: Dict[str, Any],
        expected_format: str,
        validate_schema: bool = True,
        metadata: Optional[dict[str, Any]] = None
    ) -> ParsedResponse:
        """
        Parse and validate the AI response.
        This example assumes `expected_format == "docstring"` means we should validate
        the content against the docstring schema.
        """
        self._parsing_stats["total_processed"] += 1
        parsed_response = await self.response_parser.parse_response(response, expected_format, validate_schema, metadata)
        if parsed_response.validation_success:
            self._parsing_stats["successful_parses"] += 1
        else:
            self._parsing_stats["failed_parses"] += 1
            if not parsed_response.errors:
                self._parsing_stats["validation_failures"] += 1
        return parsed_response