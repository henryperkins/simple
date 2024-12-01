"""
Documentation generation orchestration module.

This module provides functionality to orchestrate the documentation generation process,
including extracting code elements, processing docstrings, and generating markdown documentation.

Usage Example:
    ```python
    from core.docs import DocStringManager
    from core.types import DocumentationContext

    context = DocumentationContext(source_code="def example(): pass")
    manager = DocStringManager(context)
    documentation = await manager.generate_documentation()
    print(documentation)
    ```
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from core.logger import LoggerSetup
from core.code_extraction import CodeExtractor
from core.docstring_processor import DocstringProcessor
from core.markdown_generator import MarkdownGenerator
from exceptions import DocumentationError
from core.types import DocumentationContext, DocstringData


class DocStringManager:
    """Orchestrates the documentation generation process."""

    def __init__(
        self,
        context: DocumentationContext,
        docstring_processor: Optional[DocstringProcessor] = None,
        markdown_generator: Optional[MarkdownGenerator] = None
    ):
        """
        Initialize the documentation manager.

        Args:
            context (DocumentationContext): The context for documentation generation.
            docstring_processor (Optional[DocstringProcessor]): Optional docstring processor.
            markdown_generator (Optional[MarkdownGenerator]): Optional markdown generator.
        """
        self.logger = LoggerSetup.get_logger(__name__)
        self.context = context
        self.docstring_processor = docstring_processor or DocstringProcessor()
        self.markdown_generator = markdown_generator or MarkdownGenerator()
        self.code_extractor = CodeExtractor()

    async def generate_documentation(self) -> str:
        """
        Generate complete documentation.

        Returns:
            str: The generated documentation in markdown format.

        Raises:
            DocumentationError: If documentation generation fails.
        """
        try:
            # Extract code elements
            extraction_result = self.code_extractor.extract_code(self.context.source_code)
            if not extraction_result:
                raise DocumentationError("Code extraction failed")

            # Create documentation context
            doc_context = {
                'module_name': self.context.module_path.stem if self.context.module_path else "Unknown",
                'file_path': str(self.context.module_path) if self.context.module_path else "",
                'description': extraction_result.module_docstring or "No description available.",
                'classes': extraction_result.classes,
                'functions': extraction_result.functions,
                'constants': extraction_result.constants,
                'metrics': extraction_result.metrics,
                'source_code': self.context.source_code if self.context.include_source else None,
                'ai_docs': self.context.ai_generated
            }

            # Generate markdown using the markdown generator
            return self.markdown_generator.generate(doc_context)

        except Exception as e:
            self.logger.error(f"Documentation generation failed: {e}")
            raise DocumentationError(f"Failed to generate documentation: {e}")

    async def process_file(
        self,
        file_path: Path,
        output_dir: Optional[Path] = None
    ) -> Tuple[str, str]:
        """
        Process a single file and generate documentation.

        Args:
            file_path (Path): The path to the file to process.
            output_dir (Optional[Path]): Optional output directory to save the documentation.

        Returns:
            Tuple[str, str]: The source code and generated documentation.

        Raises:
            DocumentationError: If file processing fails.
        """
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Read source code
            source_code = file_path.read_text(encoding='utf-8')

            # Create documentation context
            context = DocumentationContext(
                source_code=source_code,
                module_path=file_path,
                include_source=True
            )

            # Generate documentation
            documentation = await self.generate_documentation()

            # Save to output directory if specified
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{file_path.stem}.md"
                output_file.write_text(documentation)

            return context.source_code, documentation

        except Exception as e:
            self.logger.error(f"File processing failed: {e}")
            raise DocumentationError(f"Failed to process file: {e}")
