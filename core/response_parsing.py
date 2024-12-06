"""
Response parsing service with consistent error handling and validation.
"""

# Standard library imports
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

# Third-party imports
from jsonschema import validate, ValidationError

# Local imports
from core.logger import LoggerSetup
from core.schema_loader import load_schema
from core.docstring_processor import DocstringProcessor
from core.types import ParsedResponse
from exceptions import ValidationError as CustomValidationError

logger = LoggerSetup.get_logger(__name__)

class ResponseParsingService:
    """Centralized service for parsing and validating AI responses."""

    def __init__(self) -> None:
        """Initialize the response parsing service."""
        self.logger = LoggerSetup.get_logger(__name__)
        self.docstring_processor = DocstringProcessor()
        self.docstring_schema = load_schema("docstring_schema")
        self.function_schema = load_schema("function_tools_schema")
        self._parsing_stats = {
            "total_processed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "validation_failures": 0,
        }

    async def parse_response(
        self,
        response: str,
        expected_format: str = "json",
        validate_schema: bool = True
    ) -> ParsedResponse:
        """
        Parse and validate an AI response.

        Args:
            response: Raw response string to parse
            expected_format: Expected format ('json', 'markdown', 'docstring')
            validate_schema: Whether to validate against schema

        Returns:
            ParsedResponse: Structured response data with metadata

        Raises:
            CustomValidationError: If validation fails
        """
        start_time = datetime.now()
        errors = []
        parsed_content = None

        self._parsing_stats["total_processed"] += 1

        try:
            if expected_format == "json":
                parsed_content = await self._parse_json_response(response)
            elif expected_format == "markdown":
                parsed_content = await self._parse_markdown_response(response)
            elif expected_format == "docstring":
                parsed_content = await self._parse_docstring_response(response)
            else:
                raise ValueError(f"Unsupported format: {expected_format}")

            validation_success = False
            if parsed_content and validate_schema:
                validation_success = await self._validate_response(
                    parsed_content, expected_format
                )
                if not validation_success:
                    errors.append("Schema validation failed")
                    self._parsing_stats["validation_failures"] += 1
                    parsed_content = self._create_fallback_response()

            if parsed_content:
                self._parsing_stats["successful_parses"] += 1
            else:
                self._parsing_stats["failed_parses"] += 1
                parsed_content = self._create_fallback_response()

            processing_time = (datetime.now() - start_time).total_seconds()

            return ParsedResponse(
                content=parsed_content,
                format_type=expected_format,
                parsing_time=processing_time,
                validation_success=validation_success,
                errors=errors,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "response_size": len(response),
                },
            )

        except Exception as e:
            error_message = f"Response parsing failed: {e}"
            self.logger.error(error_message, exc_info=True)
            errors.append(error_message)
            self._parsing_stats["failed_parses"] += 1
            raise CustomValidationError(error_message) from e

    async def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse a JSON response, handling code blocks and cleaning.

        Args:
            response: The response string to parse

        Returns:
            Parsed JSON content or None if parsing fails
        """
        try:
            response = response.strip()

            # Extract JSON from code blocks if present
            if "```json" in response and "```" in response:
                start = response.find("```json") + 7
                end = response.rfind("```")
                if start > 7 and end > start:
                    response = response[start:end].strip()

            # Parse JSON into Python dictionary
            parsed_content = json.loads(response)

            # Ensure required fields
            required_fields = {"summary", "description", "args", "returns", "raises"}
            for field in required_fields:
                if field not in parsed_content:
                    if field in {"args", "raises"}:
                        parsed_content[field] = []
                    elif field == "returns":
                        parsed_content[field] = {
                            "type": "Any",
                            "description": "",
                        }
                    else:
                        parsed_content[field] = ""

            # Validate field types
            if not isinstance(parsed_content["args"], list):
                parsed_content["args"] = []
            if not isinstance(parsed_content["raises"], list):
                parsed_content["raises"] = []
            if not isinstance(parsed_content["returns"], dict):
                parsed_content["returns"] = {
                    "type": "Any",
                    "description": "",
                }

            return parsed_content

        except json.JSONDecodeError as e:
            error_message = f"Failed to parse JSON response: {e}"
            self.logger.error(error_message)
            return None
        except Exception as e:
            error_message = f"Unexpected error during JSON response parsing: {e}"
            self.logger.error(error_message)
            return None

    async def _validate_response(
        self, content: Dict[str, Any], format_type: str
    ) -> bool:
        """
        Validate response against appropriate schema.

        Args:
            content: Content to validate
            format_type: Type of format to validate against

        Returns:
            True if validation succeeds, False otherwise
        """
        try:
            if format_type == "docstring":
                schema = self.docstring_schema["schema"]
                validate(instance=content, schema=schema)
            elif format_type == "function":
                validate(instance=content, schema=self.function_schema["schema"])
            return True
        except ValidationError as e:
            self.logger.error(f"Schema validation failed: {e.message}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during schema validation: {e}")
            return False

    def _create_fallback_response(self) -> Dict[str, Any]:
        """
        Create a fallback response when parsing fails.

        Returns:
            Default response structure
        """
        return {
            "summary": "AI-generated documentation not available",
            "description": "Documentation could not be generated by AI service",
            "args": [],
            "returns": {"type": "Any", "description": "Return value not documented"},
            "raises": [],
            "complexity": 1,
        }

    async def _parse_docstring_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse a docstring response, handling common formatting issues.

        Args:
            response: The response string to parse

        Returns:
            Parsed docstring content or None if parsing fails
        """
        try:
            response = response.strip()
            parsed_content = self.docstring_processor.parse(response)
            return parsed_content.__dict__ if parsed_content else None
        except Exception as e:
            self.logger.error(f"Failed to parse docstring response: {e}", exc_info=True)
            return None
