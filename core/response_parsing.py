"""
Response parsing service for handling Azure OpenAI API outputs.

This module provides comprehensive parsing and validation of Azure OpenAI API responses,
with support for structured outputs, function calling, and error handling according to
Azure best practices.
"""

import json
import time
from typing import Any, Dict, List
from pathlib import Path

from jsonschema import validate, ValidationError

from core.logger import LoggerSetup
from core.docstring_processor import DocstringProcessor
from core.markdown_generator import MarkdownGenerator
from core.metrics_collector import MetricsCollector
from core.types import ParsedResponse, DocumentationData, DocstringData
from core.types.base import DocstringSchema
from core.exceptions import DocumentationError


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
        self.docstring_processor = DocstringProcessor()
        self.markdown_generator = MarkdownGenerator(correlation_id)
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)
        self.correlation_id = correlation_id
        
        # Initialize schema validation
        try:
            self.docstring_schema = self._load_schema("docstring_schema.json")
            self.function_schema = self._load_schema("function_tools_schema.json")
        except Exception as e:
            self.logger.error(f"Schema initialization failed: {e}", exc_info=True)
            raise

        # Initialize parsing statistics
        self._parsing_stats = {
            "total_processed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "validation_failures": 0,
        }
        
        self.logger.info("ResponseParsingService initialized for Azure OpenAI")

    def _load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load a JSON schema for validation."""
        schema_path = Path(__file__).resolve().parent.parent / "schemas" / schema_name
        try:
            with schema_path.open("r") as f:
                return json.load(f)
        except FileNotFoundError as e:
            self.logger.error(f"Schema file not found: '{schema_name}' - {e}", exc_info=True)
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON schema: '{schema_name}' - {e}", exc_info=True)
            raise

    def _generate_markdown(self, content: Dict[str, Any]) -> str:
        """Convert parsed content to markdown format."""
        if not content.get("source_code"):
            self.logger.error("Source code missing from content")
            raise DocumentationError("source_code is required")

        doc_data = DocumentationData(
            module_name=content.get("module_name", ""),
            module_path=Path(content.get("file_path", ".")),
            module_summary=content.get("summary", ""),
            source_code=content["source_code"],
            docstring_data=self._create_docstring_data(content),
            ai_content=content,
            code_metadata={**content.get("code_metadata", {}), "source_code": content["source_code"]},
        )
        return self.markdown_generator.generate(doc_data)

    def _create_docstring_data(self, content: Dict[str, Any]) -> DocstringData:
        """Create DocstringData from content dict."""
        content_copy = content.copy()
        content_copy.pop("source_code", None)
        return DocstringData(
            summary=str(content_copy.get("summary", "")),
            description=str(content_copy.get("description", "")),
            args=content_copy.get("args", []),
            returns=content_copy.get("returns", {"type": "Any", "description": ""}),
            raises=content_copy.get("raises", []),
            complexity=int(content_copy.get("complexity", 1)),
        )

    def _validate_content(self, content: Dict[str, Any], format_type: str) -> tuple[bool, List[str]]:
        """Validate the content against the appropriate schema."""
        validation_errors: List[str] = []
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

            if not validation_errors:
                self.metrics_collector.collect_validation_metrics(success=True)
            else:
                self.metrics_collector.collect_validation_metrics(success=False)
            return True, validation_errors
        except ValidationError as e:
            validation_errors.append(str(e))
            self.logger.error(
                f"Validation error: {e}",
                extra={
                    "correlation_id": self.correlation_id,
                    "content": content,
                    "format_type": format_type,
                },
            )
            return False, validation_errors
        except Exception as e:
            validation_errors.append(f"Unexpected validation error: {str(e)}")
            self.logger.error(
                f"Unexpected validation error: {e}",
                exc_info=True,
                extra={
                    "correlation_id": self.correlation_id,
                    "content": content,
                    "format_type": format_type,
                },
            )
            return False, validation_errors

    def _create_fallback_response(self, response: Dict[str, Any] | str | None = None, error: str = "") -> Dict[str, Any]:
        """Create a fallback response when parsing fails."""
        self.logger.warning("Creating fallback response due to parsing failure", extra={"correlation_id": self.correlation_id})

        fallback = {
            "summary": "This module provides functionality for code analysis and documentation generation.",
            "description": (
                "The module is part of a larger system aimed at analyzing code, "
                "extracting key elements such as classes and functions, and generating "
                "comprehensive documentation. It ensures maintainability and readability "
                "by adhering to structured documentation standards."
            ),
            "args": [],
            "returns": {"type": "Any", "description": "No return value documented."},
            "raises": [],
            "complexity": 1,
            "source_code": "",
            "code_metadata": {"source_code": ""},
            "error": error
        }

        try:
            from core.dependency_injection import Injector
            context = Injector.get("extraction_context")
            if context and hasattr(context, "get_source_code"):
                source_code = context.get_source_code() or ""
                fallback["source_code"] = source_code
                fallback["code_metadata"]["source_code"] = source_code
        except Exception as e:
            self.logger.warning(f"Could not get source code from context: {e}", extra={"correlation_id": self.correlation_id})

        if isinstance(response, str):
            fallback["description"] = response.strip()
        elif isinstance(response, dict):
            if "choices" in response:
                message = response.get("choices", [{}])[0].get("message", {})
                if "content" in message and message["content"] is not None:
                    try:
                        content = json.loads(message["content"])
                        fallback.update(content)
                    except json.JSONDecodeError:
                        fallback["description"] = str(message["content"])
            elif "summary" in response and "description" in response:
                fallback.update({
                    "summary": response.get("summary", fallback["summary"]),
                    "description": response.get("description", fallback["description"]),
                })

        return fallback
    
    def _standardize_response_format(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize response format to use choices structure."""
        try:
            # Case 1: Already in choices format
            if isinstance(response, dict) and "choices" in response:
                return response

            # Handle missing or empty fields
            defaults = {
                "summary": "No summary provided.",
                "description": "No description provided.",
                "args": [],
                "returns": {"type": "Any", "description": "No return value documented."},
                "raises": [],
                "complexity": 1,
            }
            for key, default in defaults.items():
                if key not in response or not response[key]:
                    self.logger.debug(f"Setting default value for missing field: '{key}'")
                    response[key] = default

            # Normalize field formats
            if "summary" in response and isinstance(response["summary"], str):
                response["summary"] = response["summary"].strip()

            if "description" in response and isinstance(response["description"], str):
                response["description"] = response["description"].strip()

            return response

            # Case 2: Direct content format
            if isinstance(response, dict) and ("summary" in response or "description" in response):
                # Wrap the content in choices format
                standardized = {
                    "choices": [{
                        "message": {
                            "content": json.dumps({
                                "summary": response.get("summary", ""),
                                "description": response.get("description", ""),
                                "args": response.get("args", []),
                                "returns": response.get("returns", {"type": "Any", "description": ""}),
                                "raises": response.get("raises", []),
                                "complexity": response.get("complexity", 1),
                                # Preserve any other fields
                                **{k: v for k, v in response.items() if k not in ["summary", "description", "args", "returns", "raises", "complexity"]}
                            })
                        }
                    }],
                    "usage": response.get("usage", {})
                }
                self.logger.debug(
                    f"Standardized direct format response: {standardized}",
                    extra={"correlation_id": self.correlation_id}
                )
                return standardized

            # Case 3: Fallback for unknown format  
            self.logger.warning(
                "Unknown response format, creating fallback",
                extra={"correlation_id": self.correlation_id}
            )
            return {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "summary": "Unknown response format",
                            "description": str(response),
                            "args": [],
                            "returns": {"type": "Any", "description": ""},
                            "raises": [],
                            "complexity": 1
                        })
                    }
                }],
                "usage": {}
            }

        except Exception as e:
            self.logger.error(
                f"Error standardizing response format: {e}",
                extra={"correlation_id": self.correlation_id},
                exc_info=True
            )
            return self._create_fallback_response(response)

    def _extract_content(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from various response formats."""
        try:
            self.logger.debug(
                f"Raw response content before extraction: {response}",
                extra={
                    "correlation_id": self.correlation_id,
                    "response_type": type(response).__name__,
                },
            )

            # Standardize format
            response = self._standardize_response_format(response)

            # Extract from choices
            message = response["choices"][0].get("message", {})
            if not message:
                return self._create_fallback_response()

            # Parse content
            if "content" in message:
                try:
                    content = json.loads(message["content"])
                    return self._ensure_required_fields(content) 
                except json.JSONDecodeError:
                    return self._create_fallback_response(message["content"])
            elif "function_call" in message:
                return self._extract_content_from_function_call(
                    message["function_call"], "function_call"
                )
                
            return self._create_fallback_response()

        except Exception as e:
            self.logger.error(
                f"Error extracting content: {e}",
                exc_info=True,
                extra={
                    "correlation_id": self.correlation_id,
                    "response_snapshot": response,
                },
            )
            return self._create_fallback_response()

    def _ensure_required_fields(self, content: Dict[str, Any] | str) -> Dict[str, Any]:
        """Ensure required fields exist in the content."""
        if isinstance(content, str):
            return {
                "summary": "Content as string",
                "description": content.strip(),
                "args": [],
                "returns": {"type": "Any", "description": ""},
                "raises": [],
                "complexity": 1
            }

        result = dict(content)
        defaults = {
            "summary": "No summary provided.",
            "description": "No description provided.", 
            "args": [],
            "returns": {"type": "Any", "description": ""},
            "raises": [],
            "complexity": 1
        }

        for key, default in defaults.items():
            if key not in result or not result[key]:
                self.logger.debug(
                    f"Setting default value for field: '{key}'",
                    extra={"correlation_id": self.correlation_id},
                )
                result[key] = default

        return result

    def _extract_content_from_function_call(self, function_data: Dict[str, Any], call_type: str) -> Dict[str, Any]:
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
                    self.logger.error(f"Function call arguments validation failed: {errors}")
                    return self._create_fallback_response(args_dict)  # Return fallback content
                return args_dict
            return args_dict
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to decode {call_type} arguments: {function_data.get('arguments')} - Error: {e}",
                extra={"correlation_id": self.correlation_id}
            )
            return {}

    def _extract_content_from_direct_content(self, content: str) -> Dict[str, Any] | str:
        try:
            # Remove triple backticks and 'json' prefix if present
            if content.startswith("```json") and content.endswith("```"):
                content = content.strip("```").lstrip("json").strip()

            # Attempt to decode the JSON content
            content_dict = json.loads(content)
            if isinstance(content_dict, dict) and ("summary" in content_dict or "description" in content_dict):
                return content_dict
            return content
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode content from: {content} - Error: {e}", extra={"correlation_id": self.correlation_id})
            return {"content": content} if isinstance(content, str) else {}

    def _preprocess_response(self, response: dict) -> dict:
        """Preprocess the AI response to fix minor inconsistencies."""
        if isinstance(response.get("complexity"), str):
            # Map string values to integers
            complexity_map = {"low": 1, "medium": 2, "high": 3}
            response["complexity"] = complexity_map.get(response["complexity"].lower(), 1)
        return response

    async def parse_response(self, response: Dict[str, Any], expected_format: str, validate_schema: bool = True) -> ParsedResponse:
        """Parse the response from the AI model."""
        response = self._preprocess_response(response)
        """Parse the response from the AI model."""
        start_time = time.time()
        errors = []
        metadata = {}

        self.logger.debug(
            f"Raw AI response before parsing: {response}",
            extra={
                "correlation_id": self.correlation_id,
                "response_length": len(str(response)),
            },
        )
        try:
            if response is None:
                self.logger.error("Response is None, creating fallback", extra={"correlation_id": self.correlation_id})
                content = self._create_fallback_response()
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
                    try:
                        DocstringSchema.parse_obj(content)
                    except Exception as e:
                        self.logger.error(f"Schema validation failed: {e}", exc_info=True, extra={"correlation_id": self.correlation_id})
                        errors.append(str(e))
                        return ParsedResponse(
                            content=content,
                            format_type=expected_format,
                            parsing_time=time.time() - start_time,
                            validation_success=False,
                            errors=errors,
                            metadata=metadata,
                        )

            return ParsedResponse(
                content=content,
                format_type=expected_format,
                parsing_time=time.time() - start_time,
                validation_success=not errors,
                errors=errors,
                metadata=metadata,
            )
        except Exception as e:
            self.logger.error(
                f"Unexpected error during parsing: {e}",
                exc_info=True,
                extra={
                    "correlation_id": self.correlation_id,
                    "response_snapshot": response,
                },
            )
            errors.append(f"Unexpected error: {e}")
            content = self._create_fallback_response(response, str(e))
            return ParsedResponse(
                content=content,
                format_type=expected_format,
                parsing_time=time.time() - start_time,
                validation_success=False,
                errors=errors,
                metadata=metadata,
            )
