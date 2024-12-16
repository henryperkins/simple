"""Manages prompt generation and formatting for AI interactions."""

from typing import TypedDict, cast, Literal, Optional, Any
from pathlib import Path
import json
from jinja2 import Environment, FileSystemLoader
import time

from core.types.base import ExtractedClass, ExtractedFunction
from core.logger import CorrelationLoggerAdapter, LoggerSetup
from core.metrics_collector import MetricsCollector
from core.console import print_info, print_success, print_error
from core.dependency_injection import Injector


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

    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """Initialize the PromptManager."""
        self.correlation_id = correlation_id
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__), extra={"correlation_id": correlation_id}
        )
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)
        self.token_manager = Injector.get("token_manager")

        # Load templates using Jinja2
        template_dir = Path(__file__).parent
        self.env = Environment(loader=FileSystemLoader(template_dir))
        print_info("PromptManager initialized.")
        
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

    async def create_documentation_prompt(
        self,
        module_name: str = "",
        file_path: str = "",
        source_code: str = "",
        classes: Optional[list[ExtractedClass]] = None,
        functions: Optional[list[ExtractedFunction]] = None,
    ) -> str:
        """
        Create a comprehensive prompt for documentation generation.
        Args:
            module_name (str): The name of the module for which documentation is being generated.
            file_path (str): The file path of the module.
            source_code (str): The source code of the module.
            classes (Optional[list[ExtractedClass]]): A list of extracted classes from the module.
            functions (Optional[list[ExtractedFunction]]): A list of extracted functions from the module.
        Returns:
            str: The generated documentation prompt.
        Raises:
            ValueError: If module_name, file_path, or source_code are not provided.
            Exception: If an error occurs during prompt generation.
        """
        """Create a comprehensive prompt for documentation generation."""
        start_time = time.time()
        try:
            if not module_name or not file_path or not source_code:
                raise ValueError("Module name, file path, and source code are required for prompt generation.")

            print_info("Generating documentation prompt.")
            template = self.env.get_template("documentation_prompt.txt")
            prompt = template.render(
                module_name=module_name,
                file_path=file_path,
                source_code=source_code,
                classes=classes or [],
                functions=functions or [],
                _format_class_info=self._format_class_info,
                _format_function_info=self._format_function_info,
            )

            # Estimate tokens
            prompt_tokens = self.token_manager._estimate_tokens(prompt)
            print_info(f"Generated prompt with {prompt_tokens} tokens.")

            # Track prompt generation
            await self.metrics_collector.track_operation(
                operation_type="prompt_generation",
                success=True,
                duration=time.time() - start_time,
                metadata={"prompt_tokens": prompt_tokens, "template": "documentation_prompt.txt"},
            )

            processing_time = time.time() - start_time
            print_success(f"Prompt generation completed in {processing_time:.2f}s.")
            return prompt
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error generating prompt: {e}", exc_info=True)
            await self.metrics_collector.track_operation(
                operation_type="prompt_generation",
                success=False,
                duration=processing_time,
                metadata={"error": str(e)},
            )
            print_error(f"Prompt generation failed: {e}")
            raise

    def _format_function_info(self, func: ExtractedFunction) -> str:
        """Format function information for prompt.

        Args:
            func: The extracted function information.

        Returns:
            Formatted function string for the prompt.

        Raises:
            ValueError: If the function name is missing.
        """
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
            func.docstring
            if func.docstring
            else {"summary": "No summary available", "description": "No description available"}
        )
        returns_info = func.returns or {"type": "Any", "description": ""}

        # Get summary with proper type handling
        summary = "No summary available"
        if isinstance(docstring_info, dict):
            docstring_dict = cast(DocstringDict, docstring_info)
            summary = docstring_dict.get("summary", "No summary available")
        else:
            summary = docstring_info

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

        return formatted_info

    def _format_class_info(self, cls: ExtractedClass) -> str:
        """Format class information for prompt.

        Args:
            cls: The extracted class information.

        Returns:
            Formatted class string for the prompt.

        Raises:
            ValueError: If the class name is missing.
        """
        if not cls.name:
            raise ValueError("Class name is required to format class information.")

        methods_str = "\n    ".join(
            f"- {m.name}({', '.join(str(a.name) for a in m.args)})" for m in cls.methods
        )

        # Use synchronous parse with fallback to empty DocstringData
        docstring_info = (
            cls.docstring
            if cls.docstring
            else {"summary": "No summary available", "description": "No description available"}
        )

        # Get summary with proper type handling
        summary = "No summary available"
        if isinstance(docstring_info, dict):
            docstring_dict = cast(DocstringDict, docstring_info)
            summary = docstring_dict.get("summary", "No summary available")
        else:
            summary = docstring_info

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

        return formatted_info

    def get_function_schema(self, schema: Optional[dict[str, Any]] = None) -> dict[str, Any]:
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

        if not hasattr(self, "_function_schema") or not self._function_schema:
            raise ValueError("Function schema is not properly defined.")

        return {
            "name": "generate_docstring",
            "description": "Generates structured documentation from source code.",
            "parameters": self._function_schema["function"]["parameters"]
        }

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
