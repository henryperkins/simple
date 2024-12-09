"""
Response parsing service with consistent error handling and validation.

This module provides functionality for parsing AI responses, validating 
them against specified schemas, and managing parsing statistics.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from jsonschema import validate, ValidationError
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.docstring_processor import DocstringProcessor

from core.types import ParsedResponse

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
        self.logger = CorrelationLoggerAdapter(base_logger, correlation_id)  # Use correlation logger adapter
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
            schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'schemas', schema_name)
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
                        parsed_content = self.docstring_processor.parse(parsed_dict)
                        return parsed_content.__dict__ if parsed_content else None
                    except json.JSONDecodeError as json_error:
                        self.logger.warning("JSON decoding failed: %s", json_error)
                parsed_content = self.docstring_processor.parse(response)
                return parsed_content.__dict__ if parsed_content else None
            self.logger.error(f"Unsupported response type: {type(response)}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to parse docstring response: {e}", exc_info=True)
            return None

    async def parse_response(self, response: Union[str, Dict[str, Any]], expected_format: str = "json", validate_schema: bool = True) -> ParsedResponse:
        """Parse and validate an AI response.

        Args:
            response (Union[str, Dict[str, Any]]): The AI response to parse.
            expected_format (str): The expected format of the response (e.g., "json", "markdown", "docstring").
            validate_schema (bool): Whether to validate the parsed response against a schema.

        Returns:
            ParsedResponse: An object containing the parsed content and metadata about the parsing process.
        """
        start_time = datetime.now()
        errors: List[str] = []
        parsed_content = None

        self._parsing_stats["total_processed"] += 1
        self.logger.info(f"Parsing response, expected format: {expected_format}")

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
                validation_success = await self._validate_response(parsed_content, expected_format)
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
            self.logger.debug(f"Parsing completed in {processing_time:.6f} seconds")

            return ParsedResponse(
                content=parsed_content,
                format_type=expected_format,
                parsing_time=processing_time,
                validation_success=validation_success,
                errors=errors,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "response_size": len(str(response)),
                },
            )

        except Exception as e:
            error_message = f"Response parsing failed: {e}"
            self.logger.error(error_message, exc_info=True)
            errors.append(error_message)
            self._parsing_stats["failed_parses"] += 1
            
            return ParsedResponse(
                content=self._create_fallback_response(),
                format_type=expected_format,
                parsing_time=(datetime.now() - start_time).total_seconds(),
                validation_success=False,
                errors=errors,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "response_size": len(str(response)),
                    "error": str(e)
                },
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

            required_fields = {"summary", "description", "args", "returns", "raises"}
            for field in required_fields:
                if field not in parsed_content:
                    if field in {"args", "raises"}:
                        parsed_content[field] = []
                    elif field == "returns":
                        parsed_content[field] = {"type": "Any", "description": ""}
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
            self.logger.error(f"Unexpected error during JSON response parsing: {e}", exc_info=True)
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
                validate(instance=content, schema=self.docstring_schema["schema"])
            elif format_type == "function":
                if not self.function_schema:
                    self.logger.error("Function schema not loaded")
                    return False
                validate(instance=content, schema=self.function_schema["schema"])
            self.logger.debug("Schema validation successful")
            return True
        except ValidationError as e:
            self.logger.error(f"Schema validation failed: {e.message}", exc_info=True)
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during schema validation: {e}", exc_info=True)
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
            self.logger.error(f"Failed to parse markdown response: {e}", exc_info=True)
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
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = line.strip("# ").strip()
                current_content = []
            else:
                current_content.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        self.logger.debug(f"Extracted markdown sections: {list(sections.keys())}")
        return sections
