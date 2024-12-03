from core.types import DocstringData, DocumentationContext
from core.docstring_processor import DocstringProcessor
from core.logger import LoggerSetup
from core.markdown_generator import MarkdownGenerator
from core.response_parsing import ResponseParsingService
from exceptions import DocumentationError


logger = LoggerSetup.get_logger(__name__)


class DocStringManager:
    """Manages the generation of documentation, integrating AI-generated content."""

    def __init__(self, context: DocumentationContext, ai_handler, response_parser: ResponseParsingService):
        """
        Initialize the DocStringManager.

        Args:
            context (DocumentationContext): The context containing source code and metadata.
            ai_handler (AIInteractionHandler): Handler for AI interactions.
            response_parser (ResponseParsingService): Service for parsing AI responses.
        """
        self.context = context
        self.ai_handler = ai_handler
        self.response_parser = response_parser
        self.docstring_processor = DocstringProcessor()
        self.logger = LoggerSetup.get_logger(__name__)  # Add a logger instance

    async def generate_documentation(self) -> str:
        """Generates the complete markdown documentation."""
        try:
            if not self.context.ai_generated:
                raise DocumentationError("AI content not generated.")

            markdown_generator = MarkdownGenerator()
            markdown_context = {  # Construct the context for MarkdownGenerator
                "module_name": self.context.metadata.get("module_name", "Unknown Module"),
                "file_path": self.context.metadata.get("file_path", "Unknown File"),
                "description": self.context.ai_generated.get(
                    "description", "No description provided."
                ),
                "classes": self.context.classes,
                "functions": self.context.functions,
                "constants": self.context.constants,
                "changes": self.context.changes,
                "source_code": self.context.source_code,
                "ai_documentation": self.context.ai_generated,
            }
            documentation = markdown_generator.generate(markdown_context)  # Call generate with the context
            self.logger.debug("Documentation generated successfully.")
            return documentation

        except Exception as e:
            self.logger.error(f"Failed to generate documentation: {e}")
            raise DocumentationError(str(e))

    async def update_docstring(self, existing: str, new_content: str) -> str:
        """
        Update an existing docstring with new content.

        Args:
            existing (str): The existing docstring.
            new_content (str): The new content to merge.

        Returns:
            str: The updated docstring.
        """
        try:
            # Parse the existing and new docstring content
            existing_data = self.docstring_processor.parse(existing)
            new_data = self.docstring_processor.parse(new_content)

            # Merge data, preferring new content but fallback to existing where empty
            merged = DocstringData(
                summary=new_data.summary or existing_data.summary,
                description=new_data.description or existing_data.description,
                args=new_data.args or existing_data.args,
                returns=new_data.returns or existing_data.returns,
                raises=new_data.raises or existing_data.raises,
                complexity=new_data.complexity or existing_data.complexity
            )

            # Format the merged docstring
            updated_docstring = self.docstring_processor.format(merged)
            self.logger.debug("Docstring updated successfully.")
            return updated_docstring

        except Exception as e:
            self.logger.error(f"Error updating docstring: {e}")
            raise DocumentationError(f"Failed to update docstring: {e}")

    async def __aenter__(self) -> 'DocStringManager':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass  # Cleanup if needed
