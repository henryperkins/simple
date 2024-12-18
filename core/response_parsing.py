"""Response parsing service for handling AI model outputs."""

import json
import time
from typing import Any, Dict, List, TypedDict
from pathlib import Path

from jsonschema import validate, ValidationError
from core.logger import LoggerSetup
from core.docstring_processor import DocstringProcessor
from core.markdown_generator import MarkdownGenerator
from core.metrics_collector import MetricsCollector
from core.types import ParsedResponse, DocumentationData, DocstringData
from core.types.base import DocstringSchema
from core.exceptions import ValidationError as CustomValidationError, DocumentationError


class MessageDict(TypedDict, total=False):
    tool_calls: List[Dict[str, Any]]
    function_call: Dict[str, Any]
    content: str


class ChoiceDict(TypedDict):
    message: MessageDict


class ResponseDict(TypedDict, total=False):
    choices: List[ChoiceDict]
    usage: Dict[str, int]


class ContentType(TypedDict, total=False):
    summary: str
    description: str
    args: List[Dict[str, Any]]
    returns: Dict[str, str]
    raises: List[Dict[str, str]]
    complexity: int


class ResponseParsingService:
    """Centralized service for parsing and validating AI responses."""

    def __init__(self, correlation_id: str | None = None) -> None:
        """Initialize the response parsing service."""
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}", correlation_id
        )
        self.docstring_processor = DocstringProcessor()
        self.markdown_generator = MarkdownGenerator(correlation_id)
        self.docstring_schema = self._load_schema("docstring_schema.json")
        self.function_schema = self._load_schema("function_tools_schema.json")
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)
        self.correlation_id = correlation_id
        self._parsing_stats = {
            "total_processed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "validation_failures": 0,
        }
        self.logger.info("ResponseParsingService initialized.")

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
            code_metadata={**content.get("code_metadata", {}), "source_code": content["source_code"]}
        )
        return self.markdown_generator.generate(doc_data)

    def _create_docstring_data(self, content: Dict[str, Any]) -> DocstringData:
        """Create DocstringData from content dict."""
        content_copy = content.copy()
        content_copy.pop('source_code', None)
        return DocstringData(
            summary=str(content_copy.get("summary", "")),
            description=str(content_copy.get("description", "")),
            args=content_copy.get("args", []),
            returns=content_copy.get("returns", {"type": "Any", "description": ""}),
            raises=content_copy.get("raises", []),
            complexity=int(content_copy.get("complexity", 1))
        )

    def _ensure_required_fields(self, content: Dict[str, Any] | str) -> Dict[str, Any]:
        """Ensure all required fields exist in the content."""
        result = {"summary": content} if isinstance(content, str) else dict(content)

        defaults: Dict[str, Any] = {
            "summary": "",
            "description": "",
            "args": [],
            "returns": {"type": "Any", "description": ""},
            "raises": [],
            "complexity": 1
        }

        for key, default in defaults.items():
            if key not in result:
                self.logger.debug(
                    f"Setting default value for field: '{key}', because it was missing from the content",
                    extra={"correlation_id": self.correlation_id},
                )
                result[key] = default

        return result

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
            return True, []
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
            self.logger.error(f"Validation error: {e}", extra={"correlation_id": self.correlation_id})
            return False, validation_errors
        except Exception as e:
            validation_errors.append(f"Unexpected validation error: {str(e)}")
            self.logger.error(f"Unexpected validation error: {e}", exc_info=True, extra={"correlation_id": self.correlation_id})
            return False, validation_errors

    def _create_fallback_response(self, response: Dict[str, Any] | str | None = None) -> Dict[str, Any]:
        """Create a fallback response when parsing fails."""
        self.logger.warning("Creating fallback response due to parsing failure", extra={"correlation_id": self.correlation_id})

        fallback = {
            "summary": "Documentation generation failed",
            "description": "Unable to generate documentation due to parsing error",
            "args": [],
            "returns": {"type": "Any", "description": "No return value documented."},
            "raises": [],
            "complexity": 1,
            "source_code": "",
            "code_metadata": {"source_code": ""}
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
            lines = response.strip().split('\n')
            if lines and lines[0].startswith("# Module Summary"):
                fallback["summary"] = lines[0].lstrip("# Module Summary").strip()
                fallback["description"] = '\n'.join(lines[1:]).strip()
        elif isinstance(response, dict) and "choices" in response:
            message = response.get("choices", [{}])[0].get("message", {})
            if "content" in message and message["content"] is not None:
                fallback["description"] = str(message["content"])

        return fallback

    def _extract_content(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from various response formats."""
        try:
            content = {}
            source_code = response.get("source_code")

            if "choices" in response and response["choices"]:
                message = response["choices"][0].get("message", {})
                content = self._extract_content_from_message(message)

            if not content:
                if "summary" in response and "description" in response:
                    content = response
                else:
                    self.logger.warning("Response format is invalid, creating fallback.")
                    content = self._create_fallback_response(response)

            content = self._ensure_required_fields(content)

            if source_code:
                content["source_code"] = source_code
                content.setdefault("code_metadata", {})["source_code"] = source_code

            return content

        except Exception as e:
            self.logger.error(f"Error extracting content: {e}", exc_info=True)
            return {}

    def _extract_content_from_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from a message."""
        try:
            if "function_call" in message:
                return self._extract_content_from_function_call(message["function_call"], "function_call")
            if "tool_calls" in message and message["tool_calls"]:
                tool_call = message["tool_calls"][0]
                if "function" in tool_call:
                    return self._extract_content_from_function_call(tool_call["function"], "tool_call")
            if "content" in message:
                return self._extract_content_from_direct_content(message["content"])
            return {}
        except Exception as e:
            self.logger.error(f"Error extracting content from message: {e}", extra={"correlation_id": self.correlation_id})
            return {}

    def _extract_content_from_function_call(self, function_data: Dict[str, Any], call_type: str) -> Dict[str, Any]:
        """Extract content from a function call."""
        try:
            args_str = function_data.get("arguments", "{}")
            if not args_str:
                return {}
            args_dict = json.loads(args_str)
            if isinstance(args_dict, dict) and ("summary" in args_dict or "description" in args_dict):
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
            content_dict = json.loads(content)
            if isinstance(content_dict, dict) and ("summary" in content_dict or "description" in content_dict):
                return content_dict
            return content
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode content from: {content} - Error: {e}", extra={"correlation_id": self.correlation_id})
            return {"content": content} if isinstance(content, str) else {}

    async def parse_response(self, response: Dict[str, Any], expected_format: str, validate_schema: bool = True) -> ParsedResponse:
        """Parse the response from the AI model."""
        start_time = time.time()
        errors = []
        metadata = {}

        self.logger.debug(f"Raw AI response before parsing: {response}", extra={"correlation_id": self.correlation_id})
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

            self.logger.debug(f"Raw AI response: {response}")
            if not isinstance(response, dict) or "choices" not in response:
                self.logger.error(f"Invalid response format: {response}. Expected a dictionary with a 'choices' key or valid summary/description.", extra={"correlation_id": self.correlation_id})
                if "summary" in response and "description" in response:
                    content = response
                else:
                    content = self._create_fallback_response(response)
                errors.append("Response is not a dict or missing 'choices'")
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
                        validated_content = DocstringSchema.parse_obj(content)
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
            self.logger.error(f"Unexpected error during parsing: {e}", exc_info=True, extra={"correlation_id": self.correlation_id})
            errors.append(f"Unexpected error: {e}")
            content = self._create_fallback_response(response)
            return ParsedResponse(
                content=content,
                format_type=expected_format,
                parsing_time=time.time() - start_time,
                validation_success=False,
                errors=errors,
                metadata=metadata,
            )
