"""
Response parsing service with consistent error handling and validation.

This module provides functionality for parsing AI responses, validating 
them against specified schemas, and managing parsing statistics.
"""

import json
import os
from typing import Any, TypeVar, TypedDict, cast
from pathlib import Path

from jsonschema import validate, ValidationError
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.docstring_processor import DocstringProcessor
from core.markdown_generator import MarkdownGenerator
from core.types import ParsedResponse, DocumentationData
from core.types.base import DocstringData, DocstringSchema
from core.exceptions import ValidationError as CustomValidationError

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
        self.markdown_generator = MarkdownGenerator(correlation_id)
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
        """Parse the AI model response with strict validation."""
        try:
            content = self._extract_content(response)
            if not content:
                raise CustomValidationError("Failed to extract content from response")

            # If content is a string, try to parse it as JSON
            if isinstance(content, str):
                try:
                    content_dict = json.loads(content)
                    if isinstance(content_dict, dict):
                        content = content_dict
                except json.JSONDecodeError:
                    # Keep content as string if it's not valid JSON
                    pass

            # Validate content structure
            if validate_schema:
                try:
                    if isinstance(content, dict):
                        DocstringSchema(**content)
                    else:
                        # For string content, we'll validate after parsing
                        pass
                except ValueError as e:
                    fallback = self._create_fallback_response()
                    return ParsedResponse(
                        content=fallback,
                        markdown=self._generate_markdown(fallback),
                        format_type=expected_format,
                        parsing_time=0.0,
                        validation_success=False,
                        errors=[str(e)],
                        metadata={}
                    )

            # Return the parsed response with the appropriate content
            return ParsedResponse(
                content=content,
                markdown=self._generate_markdown(content if isinstance(content, dict) else {"summary": content}),
                format_type=expected_format,
                parsing_time=0.0,
                validation_success=True,
                errors=[],
                metadata={}
            )

        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            raise CustomValidationError(f"Failed to parse response: {e}")

    def _generate_markdown(self, content: dict[str, Any]) -> str:
        """Convert parsed content to markdown format."""
        try:
            # Create a minimal DocumentationData object for markdown generation
            doc_data = DocumentationData(
                module_name="",
                module_path=Path("."),  # Use current directory as default
                module_summary=content.get("summary", ""),
                source_code="",
                code_metadata={},
                ai_content=content,  # The AI-generated content we want to format
                docstring_data=DocstringData(**content)
            )
            return self.markdown_generator.generate(doc_data)
        except Exception as e:
            self.logger.error(f"Error generating markdown: {e}")
            return f"Error generating markdown documentation: {str(e)}"

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
