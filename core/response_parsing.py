"""
Response parsing service with consistent error handling and validation.

This module provides functionality for parsing AI responses, validating 
them against specified schemas, and managing parsing statistics.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
from jsonschema import validate, ValidationError
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.docstring_processor import DocstringProcessor

from core.types import ParsedResponse
from core.exceptions import ResponseParsingError

# Set up the base logger
base_logger = LoggerSetup.get_logger(__name__)


class ResponseParsingService:
    """Centralized service for parsing and validating AI responses.

    Attributes:
        docstring_processor (DocstringProcessor): Processes and validates docstring content.
        docstring_schema (Dict[str, Any]): Schema for validating docstring content.
        function_schema (Dict[str, Any]): Schema for validating function structures.
        _parsing_stats (Dict[str, int]): Tracks statistics about parsing processes.
    """

    def __init__(self, correlation_id: Optional[str] = None) -> None:
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

    def _load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load a JSON schema for validation.

        Args:
            schema_name: Name of the schema file to load

        Returns:
            Dictionary containing the loaded schema
        """
        try:
            schema_path = os.path.join(os.path.dirname(
                os.path.dirname(__file__)), 'schemas', schema_name)
            with open(schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading schema {schema_name}: {e}")
            return {}

    async def _parse_docstring_response(self, response: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Parse a docstring response, handling both string and dictionary inputs.

        Args:
            response (Union[str, Dict[str, Any]]): The response to parse.

        Returns:
            Optional[Dict[str, Any]]: The parsed response content, if successful.
        """
        try:
            if isinstance(response, dict):
                self.logger.debug("Processing response as a dictionary")
                parsed_content = self.docstring_processor.parse(response)
                return parsed_content.__dict__ if parsed_content else None
            elif isinstance(response, str):
                self.logger.debug("Processing response as a string")
                response = response.strip()
                if response.startswith('{') and response.endswith('}'):
                    try:
                        parsed_dict = json.loads(response)
                        parsed_content = self.docstring_processor.parse(
                            parsed_dict)
                        return parsed_content.__dict__ if parsed_content else None
                    except json.JSONDecodeError as json_error:
                        self.logger.warning(
                            "JSON decoding failed: %s", json_error)
                parsed_content = self.docstring_processor.parse(response)
                return parsed_content.__dict__ if parsed_content else None
            self.logger.error(f"Unsupported response type: {type(response)}")
            return None
        except Exception as e:
            self.logger.error(
                f"Failed to parse docstring response: {e}", exc_info=True)
            return None

    async def parse_response(
        self,
        response: Dict[str, Any],
        expected_format: str = "docstring",
        validate_schema: bool = True,
    ) -> "ParsedResponse":
        """
        Parses the AI model response and returns a ParsedResponse object.

        Args:
            response: The raw response from the AI model.
            expected_format: The expected format of the content.
            validate_schema: Whether to validate the content against a schema.

        Returns:
            ParsedResponse: The parsed response containing content and metadata.
        """
        try:
            # Check if 'choices' exists and is a non-empty list
            if "choices" not in response or not isinstance(response["choices"], list) or not response["choices"]:
                self.logger.error("Missing or empty 'choices' in response.")
                # Return a fallback ParsedResponse instead of raising an exception
                fallback_content = self._create_fallback_response()
                return ParsedResponse(
                    content=fallback_content,
                    format_type=expected_format,
                    parsing_time=0.0,
                    validation_success=False,
                    errors=["Missing or empty 'choices' in response."],
                    metadata={},
                )

            # Extract the content from the response
            content_str = response["choices"][0]["message"]["function_call"]["arguments"]
            content = json.loads(content_str)

            parsing_time = response.get("usage", {}).get("processing_ms", 0)

            # Validate the content if required
            is_valid = True
            validation_errors = []
            if validate_schema:
                is_valid, validation_errors = self._validate_content(content, expected_format)

            # Return the parsed response
            return ParsedResponse(
                content=content,
                format_type=expected_format,
                parsing_time=parsing_time,
                validation_success=is_valid,
                errors=validation_errors,
                metadata={},
            )
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            # Optionally, return a fallback ParsedResponse here as well
            fallback_content = self._create_fallback_response()
            return ParsedResponse(
                content=fallback_content,
                format_type=expected_format,
                parsing_time=0.0,
                validation_success=False,
                errors=[f"Failed to parse response: {e}"],
                metadata={},
            )

    async def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse a JSON response, handling code blocks and cleaning.

        Args:
            response (str): The JSON response to parse.

        Returns:
            Optional[Dict[str, Any]]: The parsed content, if successful.
        """
        try:
            response = response.strip()
            if "```json" in response and "```" in response:
                start = response.find("```json") + 7
                end = response.rfind("```")
                if start > 7 and end > start:
                    response = response[start:end].strip()

            parsed_content = json.loads(response)

            required_fields = {"summary", "description",
                               "args", "returns", "raises"}
            for field in required_fields:
                if field not in parsed_content:
                    if field in {"args", "raises"}:
                        parsed_content[field] = []
                    elif field == "returns":
                        parsed_content[field] = {
                            "type": "Any", "description": ""}
                    else:
                        parsed_content[field] = ""

            if not isinstance(parsed_content["args"], list):
                parsed_content["args"] = []
            if not isinstance(parsed_content["raises"], list):
                parsed_content["raises"] = []
            if not isinstance(parsed_content["returns"], dict):
                parsed_content["returns"] = {"type": "Any", "description": ""}

            self.logger.debug("JSON response parsed successfully")
            return parsed_content

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error during JSON response parsing: {e}", exc_info=True)
            return None

    async def _validate_response(self, content: Dict[str, Any], format_type: str) -> bool:
        """Validate response against appropriate schema.

        Args:
            content (Dict[str, Any]): The content to validate.
            format_type (str): The format type that dictates which schema to use for validation.

        Returns:
            bool: True if validation is successful, otherwise False.
        """
        try:
            if format_type == "docstring":
                if not self.docstring_schema:
                    self.logger.error("Docstring schema not loaded")
                    return False
                validate(instance=content,
                         schema=self.docstring_schema["schema"])
            elif format_type == "function":
                if not self.function_schema:
                    self.logger.error("Function schema not loaded")
                    return False
                validate(instance=content,
                         schema=self.function_schema["schema"])
            self.logger.debug("Schema validation successful")
            return True
        except ValidationError as e:
            self.logger.error(
                f"Schema validation failed: {e.message}", exc_info=True)
            return False
        except Exception as e:
            self.logger.error(
                f"Unexpected error during schema validation: {e}", exc_info=True)
            return False

    def _create_fallback_response(self) -> Dict[str, Any]:
        """Create a fallback response when parsing fails.

        Returns:
            Dict[str, Any]: A default response indicating documentation generation failure.
        """
        self.logger.info("Creating fallback response due to parsing failure")
        return {
            "summary": "AI-generated documentation not available",
            "description": "Documentation could not be generated by AI service",
            "args": [],
            "returns": {"type": "Any", "description": "Return value not documented"},
            "raises": [],
            "complexity": 1,
        }

    async def _parse_markdown_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse a markdown response, handling common formatting issues.

        Args:
            response (str): The markdown response to parse.

        Returns:
            Optional[Dict[str, Any]]: The parsed content, if successful.
        """
        try:
            response = response.strip()
            parsed_content = self._extract_markdown_sections(response)
            self.logger.debug("Markdown response parsed successfully")
            return parsed_content if parsed_content else None
        except Exception as e:
            self.logger.error(
                f"Failed to parse markdown response: {e}", exc_info=True)
            return None

    def _extract_markdown_sections(self, response: str) -> Dict[str, str]:
        """Extract sections from a markdown response.

        Args:
            response (str): The markdown response to parse.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted sections.
        """
        sections = {}
        current_section = None
        current_content: List[str] = []

        for line in response.splitlines():
            if line.startswith("#"):
                if current_section:
                    sections[current_section] = "\n".join(
                        current_content).strip()
                current_section = line.strip("# ").strip()
                current_content = []
            else:
                current_content.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        self.logger.debug(
            f"Extracted markdown sections: {list(sections.keys())}")
        return sections

    def _validate_content(self, content: Dict[str, Any], format_type: str) -> Tuple[bool, List[str]]:
        """
        Validate the content against the appropriate schema.

        Args:
            content: The content to validate
            format_type: The type of format to validate against

        Returns:
            Tuple containing validation success status and list of validation errors
        """
        try:
            validation_errors = []
            
            if format_type == "docstring":
                if not self.docstring_schema:
                    validation_errors.append("Docstring schema not loaded")
                    return False, validation_errors
                validate(instance=content, schema=self.docstring_schema)
            elif format_type == "function":
                if not self.function_schema:
                    validation_errors.append("Function schema not loaded")
                    return False, validation_errors
                validate(instance=content, schema=self.function_schema)
                
            return True, validation_errors
            
        except ValidationError as e:
            validation_errors.append(str(e))
            return False, validation_errors
        except Exception as e:
            validation_errors.append(f"Unexpected validation error: {str(e)}")
            return False, validation_errors
