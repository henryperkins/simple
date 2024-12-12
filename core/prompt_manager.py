"""Manages prompt generation and formatting for AI interactions."""

from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from core.types.base import ExtractedClass, ExtractedFunction
from core.logger import LoggerSetup, CorrelationLoggerAdapter

class PromptManager:
    """Manages the generation and formatting of prompts for AI interactions."""

    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """Initialize the PromptManager.

        Args:
            correlation_id: Optional correlation ID for tracking related operations
        """
        self.correlation_id = correlation_id
        self.logger = CorrelationLoggerAdapter(LoggerSetup.get_logger(__name__))

        # Define the function schema for structured output
        self.function_schema = {
            "name": "generate_docstring",
            "description": "Generate Google-style documentation for code",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A brief one-line summary of what the code does",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed explanation of the functionality and purpose",
                    },
                    "args": {
                        "type": "array",
                        "description": "List of arguments for the method or function",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The name of the argument",
                                },
                                "type": {
                                    "type": "string",
                                    "description": "The data type of the argument",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "A brief description of the argument",
                                },
                            },
                            "required": ["name", "type", "description"],
                        },
                    },
                    "returns": {
                        "type": "object",
                        "description": "Details about the return value",
                        "properties": {
                            "type": {
                                "type": "string",
                                "description": "The data type of the return value",
                            },
                            "description": {
                                "type": "string",
                                "description": "A brief description of the return value",
                            },
                        },
                        "required": ["type", "description"],
                    },
                    "raises": {
                        "type": "array",
                        "description": "List of exceptions that may be raised",
                        "items": {
                            "type": "object",
                            "properties": {
                                "exception": {
                                    "type": "string",
                                    "description": "The name of the exception that may be raised",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "A brief description of when this exception is raised",
                                },
                            },
                            "required": ["exception", "description"],
                        },
                    },
                    "complexity": {
                        "type": "integer",
                        "description": "McCabe complexity score",
                    },
                },
                "required": [
                    "summary",
                    "description",
                    "args",
                    "returns",
                    "raises",
                    "complexity",
                ],
            },
        }

    def create_documentation_prompt(
        self,
        module_name: str,
        file_path: str,
        source_code: str,
        classes: Optional[List[ExtractedClass]] = None,
        functions: Optional[List[ExtractedFunction]] = None
    ) -> str:
        """Create a comprehensive prompt for documentation generation.

        Args:
            module_name: Name of the module
            file_path: Path to the source file
            source_code: The source code to document
            classes: List of extracted class information
            functions: List of extracted function information

        Returns:
            Formatted prompt string for the AI model
        """
        self.logger.debug("Creating documentation prompt", extra={
            'module_name': module_name,
            'file_path': file_path
        })

        prompt = (
            f"Objective: Generate comprehensive Google-style documentation for the following Python module.\n\n"
            f"Context: This module is part of a larger system aimed at providing AI-driven solutions. "
            f"Consider the target audience as developers who will use this documentation to understand and maintain the code. "
            f"Ensure the documentation is detailed enough to facilitate onboarding and maintenance.\n\n"
            f"Module Name: {module_name}\n"
            f"File Path: {file_path}\n\n"
            "Code Structure:\n\n"
            "Examples of desired documentation include:\n"
            "- Clear summaries that succinctly describe the purpose of each component.\n"
            "- Detailed descriptions that explain the functionality and usage.\n"
            "- Well-defined argument lists with types and descriptions.\n"
            "Avoid:\n"
            "- Vague descriptions that do not add value.\n"
            "- Incomplete argument details that could lead to misunderstandings.\n\n"
            "Classes and Functions:\n"
            "Provide detailed documentation for each class and function, including their purpose, usage, and any important details.\n"
        )

        # Add class information
        if classes:
            prompt += "Classes:\n"
            for cls in classes:
                prompt += self._format_class_info(cls)
            prompt += "\n"

        # Add function information
        if functions:
            prompt += "Functions:\n"
            for func in functions:
                prompt += self._format_function_info(func)
            prompt += "\n"

        # Add source code
        prompt += (
            "Source Code:\n"
            f"{source_code}\n\n"
            "Analyze the code and generate comprehensive Google-style documentation. "
            "Include a brief summary, detailed description, arguments, return values, and possible exceptions. "
            "Ensure all descriptions are clear and technically accurate."
        )

        self.logger.debug("Documentation prompt created successfully")
        return prompt

    def create_code_analysis_prompt(self, code: str) -> str:
        """Create a prompt for code quality analysis.

        Args:
            code: Source code to analyze

        Returns:
            Formatted prompt for code analysis
        """
        self.logger.debug("Creating code analysis prompt")

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
            "Provide specific examples of improvements where applicable, and suggest alternative approaches or refactorings."
        )

        self.logger.debug("Code analysis prompt created successfully")
        return prompt

    def _format_function_info(self, func: ExtractedFunction) -> str:
        """Format function information for prompt.

        Args:
            func: The extracted function information

        Returns:
            Formatted function string
        """
        self.logger.debug(f"Formatting function info for: {func.name}")

        args_str = ", ".join(
            f"{arg.name}: {arg.type or 'Any'}"
            + (f" = {arg.default_value}" if arg.default_value else "")
            for arg in func.args
        )

        # Safely access docstring_info and returns
        docstring_info = func.docstring_info or DocstringData()
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

    def _format_class_info(self, cls: ExtractedClass) -> str:
        """Format class information for prompt.

        Args:
            cls: The extracted class information

        Returns:
            Formatted class string
        """
        self.logger.debug(f"Formatting class info for: {cls.name}")

        methods_str = "\n    ".join(
            f"- {m.name}({', '.join(a.name for a in m.args)})"
            for m in cls.methods
        )

        # Safely access docstring_info
        docstring_info = cls.docstring_info or {}

        formatted_info = (
            f"Class: {cls.name}\n"
            f"Base Classes: {', '.join(cls.bases) if cls.bases else 'None'}\n"
            f"Existing Docstring: {docstring_info.get('summary', 'None')}\n"
            f"Methods:\n    {methods_str}\n"
            f"Attributes: {', '.join(a['name'] for a in cls.attributes)}\n"
            f"Instance Attributes: {', '.join(a['name'] for a in cls.instance_attributes)}\n"
            f"Decorators: {', '.join(cls.decorators) if cls.decorators else 'None'}\n"
            f"Is Exception: {'Yes' if cls.is_exception else 'No'}\n"
            f"Complexity Score: {cls.metrics.cyclomatic_complexity if cls.metrics else 'Unknown'}\n"
        )

        self.logger.debug(f"Class info formatted for: {cls.name}")
        return formatted_info

    def get_function_schema(self) -> Dict[str, Any]:
        """Get the function schema for structured output.

        Returns:
            Function schema dictionary
        """
        self.logger.debug("Retrieving function schema")
        return self.function_schema
