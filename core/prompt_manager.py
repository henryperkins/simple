"""Manages prompt generation and formatting for AI interactions."""

from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from core.types.base import ExtractedClass, ExtractedFunction, DocstringData, Injector
from utils import handle_error
from core.logger import CorrelationLoggerAdapter


class PromptManager:
    """Manages the generation and formatting of prompts for AI interactions."""

    def __init__(self, correlation_id: Optional[str] = None) -> None:
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
        schema_path = Path(__file__).parent / "function_schema.json"
        try:
            with open(schema_path, "r") as f:
                self._function_schema = json.load(f)
        except FileNotFoundError:
            self.logger.error(
                "Function schema file not found", extra={"path": str(schema_path)}
            )
            raise
        except json.JSONDecodeError:
            self.logger.error(
                "Failed to parse JSON in function schema file",
                extra={"path": str(schema_path)},
            )
            raise

    @handle_error
    async def create_documentation_prompt(
        self,
        module_name: str = "",
        file_path: str = "",
        source_code: str = "",
        classes: Optional[List[ExtractedClass]] = None,
        functions: Optional[List[ExtractedFunction]] = None,
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

        prompt = (
            f"Objective: Generate comprehensive Google-style documentation for the following Python module.\n\n"
            f"Context: This module is part of a larger system aimed at providing AI-driven solutions. "
            f"Consider the target audience as developers who will use this documentation to understand and maintain the code. "
            f"Ensure the documentation is detailed enough to facilitate onboarding and maintenance.\n\n"
            f"Module Name: {module_name}\n"
            f"File Path: {file_path}\n\n"
            "Code Structure:\n\n"
        )

        if classes:
            prompt += "Classes:\n"
            for cls in classes:
                class_info = await self._format_class_info(cls)
                prompt += class_info
            prompt += "\n"

        if functions:
            prompt += "Functions:\n"
            for func in functions:
                func_info = await self._format_function_info(func)
                prompt += func_info
            prompt += "\n"

        prompt += (
            "Source Code:\n"
            f"{source_code}\n\n"
            "Analyze the code and generate comprehensive Google-style documentation. "
            "Include a brief summary, detailed description, arguments, return values, and possible exceptions. "
            "Ensure all descriptions are clear and technically accurate. "
            "If any information is missing or cannot be determined, explicitly state that it is not available."
        )

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

        prompt = (
            "Objective: Analyze the following code for quality and provide specific improvements.\n\n"
            "Context: This code is part of a critical system component where performance and reliability are paramount. "
            "Consider historical issues such as performance bottlenecks and error handling failures. "
            "The analysis should help in identifying potential risks and areas for optimization.\n\n"
            f"Code:\n{code}\n\n"
            "Consider the following aspects:\n"
            "1. Code complexity and readability\n"
            "2. Best practices and design patterns\n"
            "3. Error handling and edge cases\n"
            "4. Performance considerations\n"
            "5. Documentation completeness\n\n"
            "Examples of good practices include:\n"
            "- Clear variable naming that enhances readability.\n"
            "- Efficient algorithms that improve performance.\n"
            "Avoid:\n"
            "- Deep nesting that complicates understanding.\n"
            "- Lack of error handling that could lead to failures.\n\n"
            "Provide specific examples of improvements where applicable, and suggest alternative approaches or refactorings. "
            "If any information is missing or cannot be determined, explicitly state that it is not available."
        )

        self.logger.debug("Code analysis prompt created successfully")
        return prompt

    @handle_error
    async def _format_function_info(self, func: ExtractedFunction) -> str:
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
            else DocstringData(summary="")
        )
        returns_info = func.returns or {"type": "Any", "description": ""}

        formatted_info = (
            f"Function: {func.name}\n"
            f"Arguments: ({args_str})\n"
            f"Returns: {returns_info['type']}\n"
            f"Existing Docstring: {docstring_info.summary}\n"
            f"Decorators: {', '.join(func.decorators) if func.decorators else 'None'}\n"
            f"Is Async: {'Yes' if func.is_async else 'No'}\n"
            f"Complexity Score: {func.metrics.cyclomatic_complexity if func.metrics else 'Unknown'}\n"
        )

        self.logger.debug(f"Function info formatted for: {func.name}")
        return formatted_info

    @handle_error
    async def _format_class_info(self, cls: ExtractedClass) -> str:
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
            raise ValueValueError("Class name is required to format class information.")

        methods_str = "\n    ".join(
            f"- {m.name}({', '.join(a.name for a in m.args)})" for m in cls.methods
        )

        # Use synchronous parse with fallback to empty DocstringData
        try:
            docstring_info = (
                self.docstring_processor.parse(cls.docstring)
                if cls.docstring
                else DocstringData(summary="")
            )
        except Exception as e:
            self.logger.warning(f"Failed to parse docstring for {cls.name}: {e}")
            docstring_info = DocstringData(summary="Failed to parse docstring")

        formatted_info = (
            f"Class: {cls.name}\n"
            f"Base Classes: {', '.join(cls.bases) if cls.bases else 'None'}\n"
            f"Existing Docstring: {docstring_info.summary}\n"
            f"Methods:\n    {methods_str}\n"
            f"Attributes: {', '.join(a['name'] for a in cls.attributes)}\n"
            f"Instance Attributes: {', '.join(a['name'] for a in cls.instance_attributes)}\n"
            f"Decorators: {', '.join(cls.decorators) if cls.decorators else 'None'}\n"
            f"Is Exception: {'Yes' if cls.is_exception else 'No'}\n"
            f"Complexity Score: {cls.metrics.cyclomatic_complexity if cls.metrics else 'Unknown'}\n"
        )

        self.logger.debug(f"Class info formatted for: {cls.name}")
        return formatted_info

    @handle_error
    def get_function_schema(self) -> Dict[str, Any]:
        """Get the function schema for structured output.

        Returns:
            Function schema dictionary.

        Raises:
            ValueError: If the schema is not properly formatted.
        """
        self.logger.debug("Retrieving function schema")

        if not self._function_schema:
            raise ValueError("Function schema is not properly defined.")

        return {
            "name": "generate_docstring",
            "description": "Generates structured documentation from source code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "description": {"type": "string"},
                    "args": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "description": {"type": "string"},
                            },
                        },
                    },
                    "returns": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                    "raises": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "exception": {"type": "string"},
                                "description": {"type": "string"},
                            },
                        },
                    },
                    "complexity": {"type": "integer"},
                },
                "required": ["summary", "description"],
            },
        }
