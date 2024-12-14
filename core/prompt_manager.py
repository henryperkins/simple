"""Manages prompt generation and formatting for AI interactions."""

from typing import Any, TypedDict, cast, Literal
from pathlib import Path
import json
from jinja2 import Environment, FileSystemLoader

from core.types.base import ExtractedClass, ExtractedFunction, DocstringData, Injector
from utils import handle_error
from core.logger import CorrelationLoggerAdapter


class MetricsDict(TypedDict, total=False):
    """Type definition for metrics dictionary."""
    cyclomatic_complexity: int | Literal["Unknown"]


class AttributeDict(TypedDict, total=False):
    """Type definition for attribute dictionary."""
    name: str


class DocstringDict(TypedDict, total=False):
    """Type definition for docstring dictionary."""
    summary: str
    description: str


class PromptManager:
    """Manages the generation and formatting of prompts for AI interactions."""

    def __init__(self, correlation_id: str | None = None) -> None:
        """Initialize the PromptManager.

        Args:
            correlation_id: Optional correlation ID for tracking related operations.
        """
        self.correlation_id = correlation_id
        self.logger = CorrelationLoggerAdapter(
            Injector.get("logger"), extra={"correlation_id": correlation_id}
        )
        self.docstring_processor = Injector.get("docstring_processor")

        # Load the function schema from a file
        schema_path = Path(__file__).parent / "function_tools_schema.json"
        try:
            with open(schema_path, "r") as f:
                self._function_schema = json.load(f)
                self.logger.info(f"Function schema loaded from {schema_path}")
        except FileNotFoundError:
            self.logger.error(
                "Function schema file not found", extra={"path": str(schema_path)}
            )
            raise
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to parse JSON in function schema file {schema_path}: {e}",
                exc_info=True
            )
            raise
        # Setup Jinja2 environment
        template_dir = Path(__file__).parent
        self.env = Environment(loader=FileSystemLoader(template_dir))
        try:
            self.env.get_template("documentation_prompt.txt")
            self.logger.info("Template 'documentation_prompt.txt' loaded successfully.")
            self.env.get_template("code_analysis_prompt.txt")
            self.logger.info("Template 'code_analysis_prompt.txt' loaded successfully.")
        except Exception as e:
             self.logger.error(f"Error loading template file : {e}", exc_info=True)
             raise

    @handle_error
    async def create_documentation_prompt(
        self,
        module_name: str = "",
        file_path: str = "",
        source_code: str = "",
        classes: list[ExtractedClass] | None = None,
        functions: list[ExtractedFunction] | None = None,
    ) -> str:
        """Create a comprehensive prompt for documentation generation.

        Args:
            module_name: Name of the module.
            file_path: Path to the source file.
            source_code: The source code to document.
            classes: List of extracted class information.
            functions: List of extracted function information.

        Returns:
            Formatted prompt string for the AI model.

        Raises:
            ValueError: If required information is missing for prompt generation.
        """
        self.logger.debug(
            "Creating documentation prompt",
            extra={"module_name": module_name, "file_path": file_path},
        )

        if not module_name or not file_path or not source_code:
            raise ValueError(
                "Module name, file path, and source code are required for prompt generation."
            )

        template = self.env.get_template("documentation_prompt.txt")
        prompt = template.render(
            module_name=module_name,
            file_path=file_path,
            source_code=source_code,
            classes=classes,
            functions=functions,
            _format_class_info=self._format_class_info,
            _format_function_info=self._format_function_info,
        )
        prompt += "\n\nPlease respond with a JSON object that matches the schema defined in the function parameters."

        self.logger.debug("Documentation prompt created successfully")
        return prompt

    @handle_error
    def create_code_analysis_prompt(self, code: str) -> str:
        """Create a prompt for code quality analysis.

        Args:
            code: Source code to analyze.

        Returns:
            Formatted prompt for code analysis.

        Raises:
            ValueError: If the code is empty or None.
        """
        self.logger.debug("Creating code analysis prompt")

        if not code:
            raise ValueError("Source code is required for prompt generation.")

        template = self.env.get_template("code_analysis_prompt.txt")
        prompt = template.render(code=code)

        self.logger.debug("Code analysis prompt created successfully")
        return prompt

    @handle_error
    def _format_function_info(self, func: ExtractedFunction) -> str:
        """Format function information for prompt.

        Args:
            func: The extracted function information.

        Returns:
            Formatted function string for the prompt.

        Raises:
            ValueError: If the function name is missing.
        """
        self.logger.debug(f"Formatting function info for: {func.name}")

        if not func.name:
            raise ValueError(
                "Function name is required to format function information."
            )

        args_str = ", ".join(
            f"{arg.name}: {arg.type or 'Any'}"
            + (f" = {arg.default_value}" if arg.default_value else "")
            for arg in func.args
        )

        # Use the injected docstring_processor to create a DocstringData instance
        docstring_info = (
            self.docstring_processor.parse(func.docstring)
            if func.docstring
            else DocstringData(summary="No summary available", description="No description available")
        )
        returns_info = func.returns or {"type": "Any", "description": ""}

        # Get summary with proper type handling
        summary = "No summary available"
        if isinstance(docstring_info, DocstringData):
            summary = docstring_info.summary
        elif isinstance(docstring_info, dict):
            docstring_dict = cast(DocstringDict, docstring_info)
            summary = docstring_dict.get("summary", "No summary available")

        metrics = cast(MetricsDict, func.metrics or {})
        complexity = str(metrics.get("cyclomatic_complexity", "Unknown"))

        formatted_info = (
            f"Function: {func.name}\n"
            f"Arguments: ({args_str})\n"
            f"Returns: {returns_info['type']}\n"
            f"Existing Docstring: {summary}\n"
            f"Decorators: {', '.join(str(d) for d in (func.decorators or []))}\n"
            f"Is Async: {'Yes' if func.is_async else 'No'}\n"
            f"Complexity Score: {complexity}\n"
        )

        self.logger.debug(f"Function info formatted for: {func.name}")
        return formatted_info

    @handle_error
    def _format_class_info(self, cls: ExtractedClass) -> str:
        """Format class information for prompt.

        Args:
            cls: The extracted class information.

        Returns:
            Formatted class string for the prompt.

        Raises:
            ValueError: If the class name is missing.
        """
        self.logger.debug(f"Formatting class info for: {cls.name}")

        if not cls.name:
            raise ValueError("Class name is required to format class information.")

        methods_str = "\n    ".join(
            f"- {m.name}({', '.join(str(a.name) for a in m.args)})" for m in cls.methods
        )

        # Use synchronous parse with fallback to empty DocstringData
        try:
            docstring_info = (
                self.docstring_processor.parse(cls.docstring)
                if cls.docstring
                else DocstringData(summary="No summary available", description="No description available")
            )
        except Exception as e:
            self.logger.warning(f"Failed to parse docstring for {cls.name}: {e}")
            docstring_info = DocstringData(
                summary="Failed to parse docstring",
                description="Failed to parse docstring description"
            )

        # Get summary with proper type handling
        summary = "No summary available"
        if isinstance(docstring_info, DocstringData):
            summary = docstring_info.summary
        elif isinstance(docstring_info, dict):
            docstring_dict = cast(DocstringDict, docstring_info)
            summary = docstring_dict.get("summary", "No summary available")

        attributes = [cast(AttributeDict, a) for a in cls.attributes]
        instance_attrs = [cast(AttributeDict, a) for a in cls.instance_attributes]
        metrics = cast(MetricsDict, cls.metrics or {})
        complexity = str(metrics.get("cyclomatic_complexity", "Unknown"))

        formatted_info = (
            f"Class: {cls.name}\n"
            f"Base Classes: {', '.join(str(b) for b in (cls.bases or []))}\n"
            f"Existing Docstring: {summary}\n"
            f"Methods:\n    {methods_str}\n"
            f"Attributes: {', '.join(str(a.get('name', '')) for a in attributes)}\n"
            f"Instance Attributes: {', '.join(str(a.get('name', '')) for a in instance_attrs)}\n"
            f"Decorators: {', '.join(str(d) for d in (cls.decorators or []))}\n"
            f"Is Exception: {'Yes' if cls.is_exception else 'No'}\n"
            f"Complexity Score: {complexity}\n"
        )

        self.logger.debug(f"Class info formatted for: {cls.name}")
        return formatted_info

    @handle_error
    def get_function_schema(self, schema: dict[str, Any] | None = None) -> dict[str, Any]:
        """Get the function schema for structured output.

        Returns:
            Function schema dictionary.

        Raises:
            ValueError: If the schema is not properly formatted.
        """
        self.logger.debug("Retrieving function schema")

        if schema:
            return {
                "name": "generate_docstring",
                "description": "Generates structured documentation from source code.",
                "parameters": schema
            }

        if not self._function_schema:
            raise ValueError("Function schema is not properly defined.")

        return {
            "name": "generate_docstring",
            "description": "Generates structured documentation from source code.",
            "parameters": self._function_schema["function"]["parameters"]
        }
    
    @handle_error
    def get_prompt_with_schema(self, prompt: str, schema: dict[str, Any]) -> str:
        """
        Adds function calling instructions to a prompt.

        Args:
            prompt: The base prompt.
            schema: The schema to use for function calling.

        Returns:
            The prompt with function calling instructions.
        """
        self.logger.debug("Adding function calling instructions to prompt")
        return f"{prompt}\n\nPlease respond with a JSON object that matches the schema defined in the function parameters."
