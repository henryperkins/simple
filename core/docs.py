"""
Documentation generation orchestration module.
Coordinates documentation processes between various components.
"""

from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime

from core.logger import LoggerSetup
from core.types import DocumentationContext, DocumentationData
from core.markdown_generator import MarkdownGenerator
from exceptions import DocumentationError

logger = LoggerSetup.get_logger(__name__)

class DocumentationOrchestrator:
    """Orchestrates the documentation generation process."""

    def __init__(self, ai_handler, docstring_processor, markdown_generator: Optional[MarkdownGenerator] = None):
        """
        Initialize the documentation orchestrator.

        Args:
            ai_handler: Handler for AI interactions
            docstring_processor: Processor for docstrings
            markdown_generator: Generator for markdown output
        """
        self.logger = LoggerSetup.get_logger(__name__)
        self.ai_handler = ai_handler
        self.docstring_processor = docstring_processor
        self.markdown_generator = markdown_generator or MarkdownGenerator()

    async def generate_documentation(self, context: DocumentationContext) -> Tuple[str, str]:
        """
        Generate documentation for the given context.

        Args:
            context: Documentation context containing source code and metadata

        Returns:
            Tuple[str, str]: Updated source code and generated documentation

        Raises:
            DocumentationError: If documentation generation fails
        """
        try:
            self.logger.info("Starting documentation generation process")

            # Process code through AI handler
            doc_data = await self.ai_handler.process_code(
                context.metadata,
                cache_key=context.get_cache_key()
            )
            
            if not doc_data:
                self.logger.error("No documentation data returned from AI handler")
                raise DocumentationError("Failed to generate documentation data")

            # Validate the docstring data
            valid, errors = self.docstring_processor.validate(doc_data.docstring_data)
            if not valid:
                self.logger.warning(f"Docstring validation errors: {errors}")

            # Generate markdown using complete context
            markdown_context = {
                "module_name": doc_data.module_info.get("name", "Unknown Module"),
                "file_path": doc_data.module_info.get("file_path", "Unknown Path"),
                "description": doc_data.docstring_data.description or "No description available",
                "classes": context.classes or [],
                "functions": context.functions or [],
                "constants": context.constants or [],
                "source_code": doc_data.source_code if context.include_source else None,
                "ai_documentation": doc_data.ai_content or "No AI-generated content",
                "metrics": doc_data.metrics or {}
            }

            documentation = self.markdown_generator.generate(markdown_context)
            self.logger.info("Documentation generation completed successfully")
            return doc_data.source_code, documentation

        except DocumentationError as de:
            self.logger.error(f"DocumentationError encountered: {de}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during documentation generation: {e}")
            raise DocumentationError(f"Failed to generate documentation due to an unexpected error: {e}")

    async def generate_module_documentation(self, file_path: Path, output_dir: Path) -> None:
        """
        Generate documentation for a single module.

        Args:
            file_path: Path to the Python file
            output_dir: Output directory for documentation

        Raises:
            DocumentationError: If module documentation generation fails
        """
        try:
            self.logger.info(f"Generating documentation for {file_path}")
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / file_path.with_suffix('.md').name

            # Read source code
            source_code = file_path.read_text(encoding='utf-8')

            # Create documentation context
            context = DocumentationContext(
                source_code=source_code,
                module_path=file_path,
                include_source=True,
                metadata={
                    "file_path": str(file_path),
                    "module_name": file_path.stem,
                    "creation_time": datetime.now().isoformat()
                }
            )

            # Generate documentation
            _, documentation = await self.generate_documentation(context)

            # Write output
            output_path.write_text(documentation, encoding='utf-8')
            self.logger.info(f"Documentation written to {output_path}")

        except Exception as e:
            self.logger.error(f"Module documentation generation failed: {e}")
            raise DocumentationError(f"Failed to generate module documentation: {e}")

    async def __aenter__(self) -> 'DocumentationOrchestrator':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        pass