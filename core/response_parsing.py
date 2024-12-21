import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import jsonschema
import os

from core.logger import LoggerSetup, CorrelationLoggerAdapter  # Import the logger
from core.types.base import ParsedResponse
from core.types.docstring import DocstringData
from dataclasses import dataclass, asdict
from core.console import print_info, print_error
from core.exceptions import ResponseParsingError
from utils import log_and_raise_error

class ResponseParsingService:
    """
    Unified service for formatting, validating, and parsing AI responses.
    """

    class DocstringSchema(TypedDict):
        summary: str
        description: str
        args: List[Dict[str, Any]]
        returns: Dict[str, str]
        raises: List[Dict[str, str]]
        complexity: int
        metadata: Dict[str, Any]
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
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(
                f"{__name__}.{self.__class__.__name__}",
            ),
            extra={"correlation_id": correlation_id},
        )
        self.correlation_id = correlation_id

        # Flexible Schema Directory Handling
        if schema_dir:
            self.schema_dir = Path(schema_dir)
        else:
            default_schema_dir = Path(__file__).resolve().parent / "schemas"
            self.schema_dir = Path(os.environ.get("SCHEMA_DIR", default_schema_dir))

        if not self.schema_dir.exists():
            log_and_raise_error(
                self.logger,
                FileNotFoundError(f"Schema directory not found: {self.schema_dir}"),
                ConfigurationError,
                "Schema directory not found",
                self.correlation_id,
                schema_dir=self.schema_dir,
            )

        # Load Schemas
        self.docstring_schema = self._load_schema("docstring_schema.json")
        self.function_schema = self._load_schema("function_tools_schema.json")

        self._parsing_stats = {
            "total_processed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "validation_failures": 0,
        }
        
        self.schema_usage_metrics = {
            "function": 0,
            "docstring": 0,
            "fallback": 0,
        }

        self.logger.info("ResponseParsingService initialized")

    def _validate_schema(
        self, instance: Dict[str, Any], schema: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate an instance against a schema."""
        try:
            jsonschema.validate(instance=instance, schema=schema)
            return True, []
        except jsonschema.ValidationError as e:
            log_and_raise_error(
                self.logger,
                e,
                ResponseParsingError,
                "Schema validation failed",
                self.correlation_id,
                instance=instance,
                schema=schema
            )
            return False, [str(e)]
        except Exception as e:
            log_and_raise_error(
                self.logger,
                e,
                ResponseParsingError,
                "Unexpected error during schema validation",
                self.correlation_id,
                instance=instance,
                schema=schema
            )
            return False, [f"Unexpected validation error: {str(e)}"]

    def _load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load a schema by name from the schema directory."""
        schema_path = self.schema_dir / schema_name
        try:
            with schema_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError as e:
            log_and_raise_error(
                self.logger,
                e,
                ConfigurationError,
                f"Schema file not found: {schema_name}",
                self.correlation_id,
                schema_path=schema_path,
            )
            raise
        except json.JSONDecodeError as e:
            log_and_raise_error(
                self.logger,
                e,
                ConfigurationError,
                f"Error decoding JSON schema: {schema_name}",
                self.correlation_id,
                schema_path=schema_path,
            )
            raise

    def _format_fallback_response(
        self,
        metadata: Dict[str, Any],
        error: str = "",
        error_type: str = "format_error",
        format_type: str = "docstring",
    ) -> Dict[str, Any]:
        """
        Create a standardized fallback response structure.
        """
        self.logger.warning(
            f"{error_type}: {error}. Creating fallback response.",
            extra={"metadata": metadata},
        )

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

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

        fallback_usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": error,
        }

        fallback_response = {
            "choices": [{"message": {"content": json.dumps(fallback_content)}}],
            "usage": fallback_usage,
        }

        self.logger.debug(
            f"Formatted fallback response: {fallback_response}",
        )

        return fallback_response

    def _select_schema(self, content: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Dynamically select the appropriate schema based on the content.
        """
        if "parameters" in content or "examples" in content:
            self.logger.info("Selected schema: function_tools_schema (priority: parameters/examples)")
            self.schema_usage_metrics["function"] += 1
            return self.function_schema, "function"

        if "summary" in content and "description" in content:
            self.logger.info("Selected schema: docstring_schema (priority: summary/description)")
            self.schema_usage_metrics["docstring"] += 1
            return self.docstring_schema, "docstring"

        self.logger.warning("No matching schema found. Using fallback schema.")
        self.schema_usage_metrics["fallback"] += 1
        return self._FALLBACK_SCHEMA, "fallback"

    def _validate_content(self, content: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate content against the dynamically selected schema.
        """
        schema, schema_type = self._select_schema(content)

        if schema_type == "fallback":
            self.logger.warning("Using fallback schema. Skipping validation.")
            return True, []

        self.logger.info(f"Validating content against schema type: {schema_type}")
        return self._validate_schema(content, schema)

    def _ensure_required_fields(
        self, content: Any
    ) -> Dict[str, Any]:
        """Ensure required fields exist in docstring-like content."""
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
                )
                result[key] = default

        return result

    def _create_error_response(
        self,
        error: str,
        expected_format: str,
        start_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_type: str = "format_error",
    ) -> ParsedResponse:
        """Creates a ParsedResponse indicating an error."""
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

    def _validate_response_structure(
        self,
        response: Optional[Dict[str, Any]],
        expected_format: str,
        start_time: float,
        metadata: Dict[str, Any],
    ) -> ParsedResponse:
        """
        Basic structure validation of the AI response.
        """
        if response is None:
            return self._create_error_response(
                "Response is None",
                expected_format,
                start_time,
                metadata,
                error_type="validation_error",
            )

        if isinstance(response, str) and not response.strip():
            return self._create_error_response(
                "Empty response from AI service",
                expected_format,
                start_time,
                metadata,
                error_type="validation_error",
            )

        if not isinstance(response, dict):
            return self._create_error_response(
                f"Unexpected response type: {type(response)}",
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
            content=response["choices"][0]["message"]["content"],
            format_type=expected_format,
            parsing_time=time.time() - start_time,
            validation_success=True,
            errors=[],
            metadata=metadata,
        )

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

    async def _parse_message_content(
        self,
        content: str,
        expected_format: str,
        validate_schema: bool,
        start_time: float,
        metadata: Dict[str, Any],
    ) -> ParsedResponse:
        """Parse and validate message content from AI response."""
        try:
            if not content or not content.strip():
                return self._create_response(
                    content={},
                    format_type=expected_format,
                    start_time=start_time,
                    success=False,
                    errors=["Empty response content"],
                    metadata=metadata,
                )

            # Attempt to remove markdown code block if present
            if content.startswith("```json") and content.endswith("```"):
                content = content[len("```json"):-len("```")].strip()
                self.logger.debug("Detected and removed markdown code block from content.")

            try:
                parsed_content = json.loads(content)
                if not isinstance(parsed_content, dict):
                    return self._create_response(
                        content={},
                        format_type=expected_format,
                        start_time=start_time,
                        success=False,
                        errors=["Response content must be a JSON object"],
                        metadata=metadata,
                    )
            except json.JSONDecodeError as e:
                return self._create_response(
                    content={},
                    format_type=expected_format,
                    start_time=start_time,
                    success=False,
                    errors=[f"Invalid JSON: {e}"],
                    metadata=metadata,
                )

            if validate_schema:
                is_valid, validation_errors = self._validate_content(parsed_content)
                if not is_valid:
                    return self._create_response(
                        content=parsed_content,
                        format_type=expected_format,
                        start_time=start_time,
                        success=False,
                        errors=validation_errors,
                        metadata=metadata,
                    )

            parsed_content = self._ensure_required_fields(parsed_content)

            return self._create_response(
                content=parsed_content,
                format_type=expected_format,
                start_time=start_time,
                success=True,
                errors=[],
                metadata=metadata,
            )

        except Exception as e:
            log_and_raise_error(
                self.logger,
                e,
                ResponseParsingError,
                "Unexpected error parsing message content",
                self.correlation_id,
            )
            return self._create_response(
                content={},
                format_type=expected_format,
                start_time=start_time,
                success=False,
                errors=[f"Unexpected error: {e}"],
                metadata=metadata,
            )
        
    async def parse_response(
        self,
        response: Dict[str, Any],
        expected_format: str,
        validate_schema: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ParsedResponse:
        """Parses and validates an AI response."""
        start_time = time.time()
        if metadata is None:
            metadata = {}

        try:
            validated_response = self._validate_response_structure(
                response, expected_format, start_time, metadata
            )
            if validated_response.errors:
                return validated_response

            content = response["choices"][0]["message"].get("content", "")
            
            return await self._parse_message_content(
                content=content,
                expected_format=expected_format,
                validate_schema=validate_schema,
                start_time=start_time,
                metadata=metadata,
            )

        except json.JSONDecodeError as e:
            log_and_raise_error(
                self.logger,
                e,
                ResponseParsingError,
                "Invalid JSON received from AI response",
                self.correlation_id,
            )
            return self._create_error_response(
                f"Invalid JSON: {e}",
                expected_format,
                start_time,
                metadata,
                error_type="parse_error",
            )
        except Exception as e:
            log_and_raise_error(
                self.logger,
                e,
                ResponseParsingError,
                "Error parsing response",
                self.correlation_id,
            )
            return self._create_error_response(
                f"Unexpected error during parsing: {e}",
                expected_format,
                start_time,
                metadata,
                error_type="parse_error",
            )
