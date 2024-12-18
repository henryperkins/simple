"""Manages prompt generation and formatting for AI interactions."""

from typing import Any, List
from core.types.base import ExtractedClass, ExtractedFunction

class PromptManager:
    """Manages the generation and formatting of prompts for AI interactions."""

    def __init__(self, correlation_id: str | None = None) -> None:
        """Initialize the PromptManager."""
        self.correlation_id = correlation_id

    def create_prompt(self, context: dict) -> str:
        """Create a prompt based on the provided context."""
        return f"Prompt for context: {context}"

    async def create_documentation_prompt(
        self,
        module_name: str,
        file_path: str,
        source_code: str,
        classes: List[ExtractedClass],
        functions: List[ExtractedFunction],
    ) -> str:
        """Generate a documentation prompt for the given module.

        Args:
            module_name: The name of the module.
            file_path: The file path of the module.
            source_code: The source code of the module.
            classes: A list of extracted classes.
            functions: A list of extracted functions.

        Returns:
            A formatted documentation prompt string.
        """
        class_info = "\n".join([f"Class: {cls.name}" for cls in classes])
        function_info = "\n".join([f"Function: {func.name}" for func in functions])

        prompt = (
            f"Module Name: {module_name}\n"
            f"File Path: {file_path}\n"
            f"Source Code:\n{source_code}\n\n"
            f"Classes:\n{class_info}\n\n"
            f"Functions:\n{function_info}\n"
        )
        return prompt
