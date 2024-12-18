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
        # Delayed import to avoid circular dependency
        from core.dependency_injection import Injector

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
        schema_path = Path(__file__).resolve().parent.parent / "schemas" / "function_tools_schema.json"
        if not schema_path.exists():
            self.logger.error(f"Function schema file not found at {schema_path}")
            raise FileNotFoundError(f"Function schema file is missing. Expected at: {schema_path}")
        try:
            with schema_path.open("r", encoding="utf-8") as f:
                self._function_schema = json.load(f)
                self.logger.info(f"Function schema loaded successfully from {schema_path}")
        except FileNotFoundError:
            self.logger.error(
                "Function schema file not found", extra={"path": str(schema_path)}
            )
            raise
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to parse JSON in function schema file {schema_path}: {e}",
                exc_info=True,
            )
            raise

    async def create_documentation_prompt(
        self,
        module_name: str,
        file_path: str,
        source_code: str,
        classes: list[ExtractedClass] | None = None,
        functions: list[ExtractedFunction] | None = None,
    ) -> str:
        """Create a comprehensive prompt for documentation generation."""
        print_info("Generating documentation prompt.")
        start_time = time.time()  # Add start_time here

        # Enhanced system context with more specific instructions
        system_context = """You are a technical documentation expert. Analyze the provided Python source code and generate 
comprehensive documentation that includes:
1. A clear and concise module summary that describes the module's primary purpose
2. Detailed descriptions of all components focusing on their roles and relationships
3. Accurate type information and parameter descriptions
4. Clear examples where appropriate
5. Implementation notes for complex components
6. Architecture and design pattern explanations where relevant

When describing components:
- Focus on explaining the "why" and "how" beyond just the "what"
- Highlight key design decisions and architectural patterns
- Note relationships between different components
- Identify potential usage patterns and best practices
- Document any important implementation details or constraints

        # Enhanced code context with metadata
        code_context = f"""Module: {module_name}
Path: {file_path}
Language: Python

Please analyze the following code focusing on:
- Core functionality and purpose
- Key components and their relationships
- Implementation patterns and design choices
- Usage patterns and constraints
- Performance considerations where relevant
"""

        # Format the components list for better context
        components_list = []
        if classes:
            components_list.extend(
                [
                    "\nClasses:",
                    *[
                        f"- {cls.name}: {cls.docstring_info.summary if cls.docstring_info else 'No description'}"
                        for cls in classes
                    ],
                ]
            )
        if functions:
            components_list.extend(
                [
                    "\nFunctions:",
                    *[
                        f"- {func.name}: {func.get_docstring_info().summary if func.get_docstring_info() else 'No description'}"
                        for func in functions
                    ],
                ]
            )

        # Build the complete prompt with explicit instructions
        prompt = f"""{system_context}

{code_context}

Component Overview:
{chr(10).join(components_list)}

Source Code:
```
{source_code}
```

Expected Response Format:
Please respond with a JSON object in the following format:
{{
    "summary": "A brief summary of the module's purpose.",
    "description": "A detailed explanation of the module's functionality and design.",
    "args": [
        {{
            "name": "argument_name",
            "type": "argument_type",
            "description": "A brief description of the argument."
        }}
    ],
    "returns": {{
        "type": "return_type",
        "description": "A brief description of the return value."
    }},
    "raises": [
        {{
            "exception": "exception_name",
            "description": "A brief description of the exception."
        }}
    ],
    "complexity": 1
}}

Important Notes:
- Ensure the response strictly adheres to the JSON format above.
- Include all relevant details about the module, classes, and functions.
- Avoid adding any extraneous information outside the JSON structure.
"""

        # Estimate tokens
        prompt_tokens = self.token_manager._estimate_tokens(prompt)
        print_info(f"Generated prompt with {prompt_tokens} tokens.")

        # Track prompt generation
        await self.metrics_collector.track_operation(
            operation_type="prompt_generation",
            success=True,
            duration=time.time() - start_time,
            metadata={
                "prompt_tokens": prompt_tokens,
                "template": "documentation_prompt.txt",
            },
        )

        processing_time = time.time() - start_time
        print_success(f"Prompt generation completed in {processing_time:.2f}s.")
        return prompt

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
            else {
                "summary": "No summary available",
                "description": "No description available",
            }
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
            else {
                "summary": "No summary available",
                "description": "No description available",
            }
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

    def get_function_schema(
        self, schema: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
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
                "parameters": schema,
            }

        if not hasattr(self, "_function_schema") or not self._function_schema:
            raise ValueError("Function schema is not properly defined.")

        return self._function_schema["function"]

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
