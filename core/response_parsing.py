import json
import time
import logging
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TypedDict
import jsonschema
import os

from core.logger import LoggerSetup  # Assuming you have a logger setup

#####################################
# Dataclasses and TypedDict from base.py
#####################################

from dataclasses import dataclass, field, asdict


@dataclass
class ParsedResponse:
    """Response from parsing operations."""

    content: Dict[str, Any]
    format_type: str
    parsing_time: float
    validation_success: bool
    errors: List[str]
    metadata: Dict[str, Any]
    markdown: str = ""


#####################################
# Combined Class: ResponseParsingService
#####################################


class ResponseParsingService:
    """
    Unified service for formatting, validating, and parsing AI responses.
    """

    # Docstring schema typed definition (optional, for type hinting)
    class DocstringSchema(TypedDict):
        summary: str
        description: str
        args: list[dict[str, Any]]
        returns: dict[str, str]
        raises: list[dict[str, str]]
        complexity: int
        metadata: dict[str, Any]
        error: str
        error_type: str

    _FALLBACK_SCHEMA: DocstringSchema = {
        "summary": "No summary available",
        "description": "No description available",
        "args": [],
        "returns": {"type": "Any", "description": "No return description provided"},
        "raises": [],
        "complexity": 1,
        "metadata": {},
        "error": "",
        "error_type": "none",
    }

    def __init__(
        self, correlation_id: Optional[str] = None, schema_dir: Optional[str] = None
    ) -> None:
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}", correlation_id
        )
        self.correlation_id = correlation_id

        # --- Flexible Schema Directory Handling ---
        if schema_dir:
            self.schema_dir = Path(schema_dir)
        else:
            default_schema_dir = Path(__file__).resolve().parent.parent / "schemas"
            self.schema_dir = Path(os.environ.get("SCHEMA_DIR", default_schema_dir))

        # Ensure schema directory exists
        if not self.schema_dir.exists():
            self.logger.error(f"Schema directory does not exist: {self.schema_dir}")
            raise FileNotFoundError(f"Schema directory not found: {self.schema_dir}")

        # --- Load Schemas ---
        self.docstring_schema = self._load_schema("docstring_schema.json")
        self.function_schema = self._load_schema("function_tools_schema.json")

        # --- Initialize Parsing Statistics ---
        self._parsing_stats = {
            "total_processed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "validation_failures": 0,
        }
        
        # --- Initialize Schema Usage Metrics ---
        self.schema_usage_metrics = {
            "function": 0,
            "docstring": 0,
            "fallback": 0,
        }

        self.logger.info("ResponseParsingService initialized")

    #####################################
    # Internal Schema Validation Methods
    #####################################

    def _validate_schema(
        self, instance: Dict[str, Any], schema: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate an instance against a schema."""
        try:
            jsonschema.validate(instance=instance, schema=schema)
            return True, []
        except jsonschema.ValidationError as e:
            self.logger.error(
                f"Schema validation failed: {str(e)}",
                exc_info=True,
                extra={"correlation_id": self.correlation_id},
            )
            return False, [str(e)]
        except Exception as e:
            self.logger.error(
                f"Unexpected error during schema validation: {str(e)}",
                exc_info=True,
                extra={"correlation_id": self.correlation_id},
            )
            return False, [f"Unexpected validation error: {str(e)}"]

    def _load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load a schema by name from the schema directory."""
        schema_path = self.schema_dir / schema_name
        try:
            with schema_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError as e:
            self.logger.error(
                f"Schema file not found: {schema_name} - {e}",
                exc_info=True,
                extra={"correlation_id": self.correlation_id},
            )
            raise
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Error decoding JSON schema: {schema_name} - {e}",
                exc_info=True,
                extra={"correlation_id": self.correlation_id},
            )
            raise

    #####################################
    # Docstring Validation
    #####################################

    def _validate_docstring(
        self, docstring_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate docstring data against the docstring schema."""
        return self._validate_schema(docstring_data, self.docstring_schema)

    #####################################
    # Response Formatting Methods
    #####################################

    def _format_summary_description_response(
        self, response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format a summary/description response."""
        formatted = {
            "choices": [{"message": {"content": json.dumps(response)}}],
            "usage": response.get("usage", {}),
        }
        self.logger.debug(
            f"Formatted summary/description response: {formatted}",
            extra={"correlation_id": self.correlation_id},
        )
        return formatted

    def _format_function_call_response(
        self, response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format a function call response."""
        formatted_response = {
            "choices": [{"message": {"function_call": response["function_call"]}}],
            "usage": response.get("usage", {}),
        }
        self.logger.debug(
            f"Formatted function call response: {formatted_response}",
            extra={"correlation_id": self.correlation_id},
        )
        return formatted_response

    def _format_tool_calls_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Format a tool calls response."""
        formatted_response = {
            "choices": [{"message": {"tool_calls": response["tool_calls"]}}],
            "usage": response.get("usage", {}),
        }
        self.logger.debug(
            f"Formatted tool calls response: {formatted_response}",
            extra={"correlation_id": self.correlation_id},
        )
        return formatted_response

    def _format_fallback_response(
        self,
        metadata: Dict[str, Any],
        error: str = "",
        error_type: str = "format_error",
        format_type: str = "docstring",
    ) -> Dict[str, Any]:
        """
        Create a standardized fallback response structure.

        Args:
            metadata: Additional metadata to include in the fallback response.
            error: The error message describing why the fallback was triggered.
            error_type: The type of error (e.g., "format_error", "validation_error").
            format_type: The expected format type (e.g., "docstring", "function").

        Returns:
            A standardized fallback response dictionary.
        """
        # Log the fallback creation
        self.logger.warning(
            f"{error_type}: {error}. Creating fallback response.",
            extra={"metadata": metadata, "correlation_id": self.correlation_id},
        )

        # Add timestamp for debugging
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        # Create fallback content
        fallback_content: ResponseParsingService.DocstringSchema = {
            **self._FALLBACK_SCHEMA,
            "summary": f"Invalid {format_type} format",
            "description": f"The {format_type} response did not match the expected structure.",
            "error": error,
            "error_type": error_type,
            "metadata": {
                **metadata,
                "timestamp": timestamp,
                "correlation_id": self.correlation_id,
            },
        }

        # Populate usage field with default values or metrics
        fallback_usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": error,
        }

        # Construct the fallback response
        fallback_response = {
            "choices": [{"message": {"content": json.dumps(fallback_content)}}],
            "usage": fallback_usage,
        }

        # Log the formatted fallback response
        self.logger.debug(
            f"Formatted fallback response: {fallback_response}",
            extra={"correlation_id": self.correlation_id},
        )

        return fallback_response

    def _standardize_response_format(self, response: Any) -> Dict[str, Any]:
        """
        Standardize response format to ensure proper structure.

        Args:
            response: The raw response from the AI service.

        Returns:
            A standardized response dictionary with the "choices" structure.
        """
        try:
            # Case 1: Already in "choices" format
            if isinstance(response, dict) and "choices" in response:
                self.logger.debug("Response is already in 'choices' format.")
                return response

            # Case 2: Raw string content
            if isinstance(response, str):
                try:
                    # Attempt to parse as JSON
                    content = json.loads(response)
                    if isinstance(content, dict):
                        self.logger.debug("Raw string content parsed as JSON.")
                        return {
                            "choices": [{"message": {"content": json.dumps(content)}}]
                        }
                except json.JSONDecodeError:
                    self.logger.warning(
                        "Raw string content is not valid JSON. Wrapping as plain text.",
                        extra={"correlation_id": self.correlation_id},
                    )
                    return {"choices": [{"message": {"content": response}}]}

            # Case 3: Direct content format (dict with summary/description)
            if isinstance(response, dict) and (
                "summary" in response or "description" in response
            ):
                self.logger.debug("Response contains direct content with summary/description.")
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "summary": response.get(
                                            "summary", "No summary provided"
                                        ),
                                        "description": response.get(
                                            "description", "No description provided"
                                        ),
                                        "args": response.get("args", []),
                                        "returns": response.get(
                                            "returns",
                                            {"type": "Any", "description": ""},
                                        ),
                                        "raises": response.get("raises", []),
                                        "complexity": response.get("complexity", 1),
                                    }
                                )
                            }
                        }
                    ]
                }

            # Case 4: Unknown format
            self.logger.warning(
                "Unknown response format. Falling back to default.",
                extra={"correlation_id": self.correlation_id},
            )
            return self._format_fallback_response(
                metadata={"raw_response": str(response)[:100]},
                error="Unrecognized response format",
                error_type="format_error",
                format_type="unknown",
            )

        except Exception as e:
            # Handle unexpected errors
            self.logger.error(
                f"Error standardizing response format: {e}",
                exc_info=True,
                extra={"correlation_id": self.correlation_id},
            )
            return self._format_fallback_response(
                metadata={"raw_response": str(response)[:100]},
                error=str(e),
                error_type="standardization_error",
                format_type="unknown",
            )

    #####################################
    # Internal Validation and Content Extraction Methods
    #####################################

    def _select_schema(self, content: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Dynamically select the appropriate schema based on the content.

        Args:
            content: The response content to analyze.

        Returns:
            A tuple containing the selected schema and the schema type ("function", "docstring", or "fallback").
        """
        # Prioritize function schema if "parameters" or "examples" are present
        if "parameters" in content or "examples" in content:
            self.logger.info("Selected schema: function_tools_schema (priority: parameters/examples)")
            self.schema_usage_metrics["function"] += 1
            return self.function_schema, "function"

        # Use docstring schema if "summary" and "description" are present
        if "summary" in content and "description" in content:
            self.logger.info("Selected schema: docstring_schema (priority: summary/description)")
            self.schema_usage_metrics["docstring"] += 1
            return self.docstring_schema, "docstring"

        # Fallback if no schema matches
        self.logger.warning("No matching schema found. Using fallback schema.")
        self.schema_usage_metrics["fallback"] += 1
        return self._FALLBACK_SCHEMA, "fallback"

    def _validate_content(self, content: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate content against the dynamically selected schema.

        Args:
            content: The response content to validate.

        Returns:
            A tuple containing a boolean indicating validation success and a list of errors.
        """
        schema, schema_type = self._select_schema(content)

        if schema_type == "fallback":
            self.logger.warning("Using fallback schema. Skipping validation.")
            return True, []

        self.logger.info(f"Validating content against schema type: {schema_type}")
        return self._validate_schema(content, schema)

    def _validate_response_structure(
        self,
        response: Dict[str, Any],
        expected_format: str,
        start_time: float,
        metadata: Dict[str, Any],
    ) -> ParsedResponse:
        """Basic structure validation of the AI response."""
        if response is None:
            return self._create_error_response(
                "Response is None",
                expected_format,
                start_time,
                metadata,
                error_type="validation_error",
            )

        if "choices" not in response or not response["choices"]:
            return self._create_error_response(
                "No choices in response",
                expected_format,
                start_time,
                metadata,
                error_type="validation_error",
            )

        if not response["choices"][0].get("message"):
            return self._create_error_response(
                "No message in response",
                expected_format,
                start_time,
                metadata,
                error_type="validation_error",
            )

        if "content" not in response["choices"][0]["message"]:
            return self._create_error_response(
                "No content field in message",
                expected_format,
                start_time,
                metadata,
                error_type="validation_error",
            )

        return ParsedResponse(
            content={},
            format_type=expected_format,
            parsing_time=time.time() - start_time,
            validation_success=True,
            errors=[],
            metadata=metadata,
        )

    def _create_error_response(
        self,
        error: str,
        expected_format: str,
        start_time: Optional[float] = None,
        metadata: Optional[dict] = None,
        error_type: str = "format_error",
    ) -> ParsedResponse:
        """Create a standardized error response."""
        if metadata is None:
            metadata = {}
        return ParsedResponse(
            content=self._format_fallback_response(
                metadata,
                error=error,
                error_type=error_type,
                format_type=expected_format,
            ),
            format_type=expected_format,
            parsing_time=time.time() - start_time if start_time else 0.0,
            validation_success=False,
            errors=[error],
            metadata=metadata,
        )

    def _extract_content(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from the standardized response."""
        self.logger.debug(
            f"Raw response content before extraction: {response}",
            extra={"correlation_id": self.correlation_id},
        )

        # Standardize format
        response = self._standardize_response_format(response)

        # Extract from choices
        if "choices" not in response or not response["choices"]:
            return self._format_fallback_response(
                {},
                "No choices found",
                error_type="validation_error",
                format_type="extraction",
            )

        message = response["choices"][0].get("message", {})
        if not message:
            return self._format_fallback_response(
                {},
                "No message found",
                error_type="validation_error",
                format_type="extraction",
            )

        # Parse content
        if "content" in message:
            try:
                content = json.loads(message["content"])
                return self._ensure_required_fields(content)
            except json.JSONDecodeError:
                return self._format_fallback_response(
                    {},
                    "Invalid JSON in content",
                    error_type="parse_error",
                    format_type="extraction",
                )
        elif "function_call" in message:
            return self._extract_content_from_function_call(
                message["function_call"], "function_call"
            )
        elif "tool_calls" in message:
            tool_calls = message["tool_calls"]
            if isinstance(tool_calls, list):
                return self._extract_content_from_tool_calls(tool_calls, "tool_calls")

        return self._format_fallback_response(
            {},
            "Unrecognized format",
            error_type="format_error",
            format_type="extraction",
        )

    def _ensure_required_fields(
        self, content: Union[Dict[str, Any], str]
    ) -> Dict[str, Any]:
        """Ensure required fields exist in the docstring-like content."""
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
                        f"Function call arguments validation failed: {errors}",
                        extra={"correlation_id": self.correlation_id},
                    )
                    return self._format_fallback_response(
                        {},
                        "Invalid function schema",
                        error_type="validation_error",
                        format_type="function_call",
                    )
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
                                    f"Function call arguments validation failed: {errors}",
                                    extra={"correlation_id": self.correlation_id},
                                )
                                return self._format_fallback_response(
                                    {},
                                    "Invalid tool function schema",
                                    error_type="validation_error",
                                    format_type="tool_call",
                                )
                            extracted_content.update(args_dict)
                    except json.JSONDecodeError as e:
                        self.logger.error(
                            f"Failed to decode {call_type} arguments: "
                            f"{function_data.get('arguments')} - Error: {e}",
                            extra={"correlation_id": self.correlation_id},
                        )
        return extracted_content

    def _extract_content_from_direct_content(
        self, content: str
    ) -> Union[Dict[str, Any], str]:
        """Decode direct content."""
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

    def _create_response(
        self,
        content: Dict[str, Any],
        format_type: str,
        start_time: float,
        success: bool,
        errors: List[str],
        metadata: Dict[str, Any],
    ) -> ParsedResponse:
        """Create a standardized ParsedResponse object."""
        return ParsedResponse(
            content=content,
            format_type=format_type,
            parsing_time=time.time() - start_time,
            validation_success=success,
            errors=errors,
            metadata=metadata,
        )

    #####################################
    # Main Public Method for Parsing
    #####################################

    async def _parse_message_content(
        self,
        content: str,
        validate_schema: bool,
        start_time: float,
        metadata: dict[str, Any],
    ) -> ParsedResponse:
        """
        Parse and validate message content from AI response.

        Args:
            content: The raw content to parse.
            validate_schema: Whether to validate the parsed content against a schema.
            start_time: The start time of the parsing process.
            metadata: Additional metadata for logging and debugging.

        Returns:
            A ParsedResponse object with validation results.
        """
        try:
            parsed_content = json.loads(content)

            if validate_schema:
                is_valid, schema_errors = self._validate_content(parsed_content)
                if not is_valid:
                    return ParsedResponse(
                        content=parsed_content,
                        format_type="dynamic",
                        parsing_time=time.time() - start_time,
                        validation_success=False,
                        errors=schema_errors,
                        metadata=metadata,
                    )

            return ParsedResponse(
                content=parsed_content,
                format_type="dynamic",
                parsing_time=time.time() - start_time,
                validation_success=True,
                errors=[],
                metadata=metadata,
            )
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in message content: {e}")
            return ParsedResponse(
                content={},
                format_type="dynamic",
                parsing_time=time.time() - start_time,
                validation_success=False,
                errors=[f"Invalid JSON: {e}"],
                metadata=metadata,
            )
        except Exception as e:
            self.logger.error(f"Unexpected error parsing message content: {e}", exc_info=True)
            return ParsedResponse(
                content={},
                format_type="dynamic",
                parsing_time=time.time() - start_time,
                validation_success=False,
                errors=[f"Unexpected error: {e}"],
                metadata=metadata,
            )

    async def parse_response(
        self,
        response: Dict[str, Any],
        expected_format: str,
        validate_schema: bool = True,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ParsedResponse:
        """
        Parses and validates an AI response.
        """
        start_time = time.time()
        if metadata is None:
            metadata = {}
        self.logger.debug(
            f"Raw AI response before parsing: {response}",
            extra={"correlation_id": self.correlation_id},
        )

        try:
            # Validate basic response structure
            validated_response = self._validate_response_structure(
                response, expected_format, start_time, metadata
            )
            if validated_response.errors:
                return validated_response

            # Parse message content
            parsed_content = await self._parse_message_content(
                response["choices"][0]["message"].get("content", ""),
                expected_format,
                validate_schema,
                start_time,
                metadata,
            )

            return self._create_response(
                content=parsed_content.content,
                format_type=expected_format,
                start_time=start_time,
                success=parsed_content.validation_success,
                errors=parsed_content.errors,
                metadata=metadata,
            )

        except json.JSONDecodeError as e:
            self.logger.error(
                f"Invalid JSON received from AI response: {e}",
                extra={"raw_response": response, "correlation_id": self.correlation_id},
            )
            return self._create_error_response(
                f"Invalid JSON: {e}",
                expected_format,
                start_time,
                metadata,
                error_type="parse_error",
            )
        except Exception as e:
            self.logger.error(
                f"Error parsing response: {e}",
                exc_info=True,
                extra={"raw_response": response, "correlation_id": self.correlation_id},
            )
            return self._create_error_response(
                f"Unexpected error during parsing: {e}",
                expected_format,
                start_time,
                metadata,
                error_type="parse_error",
            )