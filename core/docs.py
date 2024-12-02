"""
Documentation generation orchestration module.

This module provides functionality to orchestrate the documentation generation
process, including extracting code elements, processing docstrings, and
generating markdown documentation.

Usage Example:
    ```python
    from core.docs import DocStringManager
    from core.types import DocumentationContext, ExtractedFunction, ExtractedClass

    context = DocumentationContext(source_code="def example(): pass")
    manager = DocStringManager(context)
    documentation = await manager.generate_documentation()
    print(documentation)
    ```
"""
import ast
import json
from pathlib import Path
from typing import Optional, Any, Tuple, Dict, List
from datetime import datetime
from core.logger import LoggerSetup
from core.code_extraction import CodeExtractor, ExtractedFunction, ExtractedClass
from core.docstring_processor import DocstringProcessor
from core.markdown_generator import MarkdownGenerator
from exceptions import DocumentationError
from core.types import DocumentationContext


class DocStringManager:
    """Orchestrates the documentation generation process."""
    
    def __init__(
        self,
        context: DocumentationContext,
        ai_handler: AIHandler,  # Required parameter
        docstring_processor: Optional[DocstringProcessor] = None,
        markdown_generator: Optional[MarkdownGenerator] = None
    ):
        self.logger = LoggerSetup.get_logger(__name__)
        self.context = context
        self.ai_handler = ai_handler
        self.docstring_processor = docstring_processor or DocstringProcessor()
        self.markdown_generator = markdown_generator or MarkdownGenerator()
        self.code_extractor = CodeExtractor()

    async def generate_documentation(self) -> str:
        """Generate complete documentation."""
        try:
            # Extract code elements
            extraction_result = self.code_extractor.extract_code(self.context.source_code)
            if not extraction_result:
                raise DocumentationError("Code extraction failed")

            # Process AI-generated documentation if available
            ai_docs = {}
            if self.context.ai_generated:
                try:
                    if isinstance(self.context.ai_generated, str):
                        ai_docs = json.loads(self.context.ai_generated)
                    elif isinstance(self.context.ai_generated, dict):
                        ai_docs = self.context.ai_generated
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse AI-generated documentation")

            # Create documentation context
            doc_context = {
                'module_name': self.context.module_path.stem if self.context.module_path else "Unknown",
                'file_path': str(self.context.module_path) if self.context.module_path else "",
                'description': extraction_result.module_docstring or "No description available.",
                'classes': extraction_result.classes,
                'functions': extraction_result.functions,
                'constants': extraction_result.constants,
                'metrics': extraction_result.metrics,
                'source_code': self.context.source_code if self.context.include_source else None
            }

            # Generate markdown using the markdown generator
            return self.markdown_generator.generate(doc_context)

        except Exception as e:
            self.logger.error(f"Documentation generation failed: {e}")
            raise DocumentationError(f"Failed to generate documentation: {e}")
        
    def _format_constants(self, constants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                'name': str(const.get('name', '')),  # Use safe .get() and convert to string
                'type': str(const.get('type', '')),
                'value': str(const.get('value', ''))
            } for const in constants
        ]

    def _format_classes(self, classes: List[ExtractedClass]) -> List[Dict[str, Any]]:
        return [
            {
                'name': cls.name,
                'docstring': cls.docstring,
                'methods': [
                    {   # Format methods correctly
                        'name': m.name,
                        'docstring': m.docstring,
                        'args': m.args,
                        'return_type': m.return_type,
                        'metrics': m.metrics,
                        'source': m.source
                    } for m in cls.methods
                ],
                'bases': cls.bases,
                'metrics': cls.metrics,
                'source': cls.source
            } for cls in classes
        ]

    def _format_functions(self, functions: List[ExtractedFunction]) -> List[Dict[str, Any]]:
        return [
            {
                'name': func.name,
                'docstring': func.docstring,
                'args': func.args,
                'return_type': func.return_type,
                'metrics': func.metrics,
                'source': func.source,
            } for func in functions
        ]
    
    async def process_file(
        self,
        file_path: Path,
        output_dir: Optional[Path] = None
    ) -> Tuple[str, str]:
        """
        Process a single file and generate documentation.

        Args:
            file_path (Path): The path to the file to process.
            output_dir (Optional[Path]): Optional output directory to save the
                documentation.

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