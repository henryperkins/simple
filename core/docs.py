"""
Documentation generation orchestration module.

This module provides functionality to orchestrate the documentation generation
process, including extracting code elements, processing docstrings, and
generating markdown documentation.

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
import json
from pathlib import Path
from typing import Optional, Any, Tuple, Dict, List
from datetime import datetime
from core.logger import LoggerSetup
from core.code_extraction import CodeExtractor
from core.docstring_processor import DocstringProcessor
from core.markdown_generator import MarkdownGenerator
from exceptions import DocumentationError
from core.types import DocumentationContext


class DocStringManager:
    """Orchestrates the documentation generation process."""
    
    def __init__(
        self,
        context: DocumentationContext,
        ai_handler: Any,  # Required parameter
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
        """
        Generate complete documentation.

        Returns:
            str: The generated documentation in markdown format.

        Raises:
            DocumentationError: If documentation generation fails.
        """
        self.logger.debug(f"Metadata received: {self.context.metadata}")
        try:
            # Extract code elements
            extraction_result = self.code_extractor.extract_code(
                self.context.source_code
            )
            if not extraction_result:
                raise DocumentationError("Code extraction failed")

            # Format constants properly
            formatted_constants = []
            if extraction_result.constants:
                for const in extraction_result.constants:
                    if isinstance(const, dict) and all(key in const for key in ['name', 'type', 'value']):
                        formatted_constants.append(const)
                    else:
                        try:
                            if isinstance(const, tuple):
                                name, value, const_type = const
                            else:
                                name = const
                                value = const
                                const_type = type(const).__name__

                            formatted_constants.append({
                                'name': str(name),
                                'type': str(const_type),
                                'value': str(value)
                            })
                        except Exception as e:
                            self.logger.warning(f"Skipping malformed constant: {const} - {str(e)}")

            # Format classes
            formatted_classes = []
            if extraction_result.classes:
                for cls in extraction_result.classes:
                    if isinstance(cls, dict):
                        formatted_classes.append(cls)
                    else:
                        try:
                            formatted_classes.append({
                                'name': cls.name,
                                'docstring': cls.docstring,
                                'methods': cls.methods,
                                'bases': cls.bases if hasattr(cls, 'bases') else [],
                                'metrics': cls.metrics if hasattr(cls, 'metrics') else {},
                                'source': cls.source if hasattr(cls, 'source') else None
                            })
                        except Exception as e:
                            self.logger.warning(f"Skipping malformed class: {cls} - {str(e)}")

            # Format functions
            formatted_functions = []
            if extraction_result.functions:
                for func in extraction_result.functions:
                    if isinstance(func, dict):
                        formatted_functions.append(func)
                    else:
                        try:
                            formatted_functions.append({
                                'name': func.name,
                                'docstring': func.docstring,
                                'args': func.args,
                                'return_type': func.return_type if hasattr(func, 'return_type') else None,
                                'metrics': func.metrics if hasattr(func, 'metrics') else {},
                                'source': func.source if hasattr(func, 'source') else None
                            })
                        except Exception as e:
                            self.logger.warning(f"Skipping malformed function: {func} - {str(e)}")

            # Create documentation context
            doc_context = {
                'module_name': (
                    self.context.module_path.stem
                    if self.context.module_path else "Unknown"
                ),
                'file_path': (
                    str(self.context.module_path)
                    if self.context.module_path else ""
                ),
                'description': (
                    extraction_result.module_docstring
                    or "No description available."
                ),
                'classes': formatted_classes,
                'functions': formatted_functions,
                'constants': formatted_constants,
                'metrics': extraction_result.metrics or {},
                'source_code': (
                    self.context.source_code
                    if self.context.include_source else None
                ),
                'imports': extraction_result.imports if hasattr(extraction_result, 'imports') else {},
                'ai_docs': (
                    self.context.metadata.get('ai_generated')
                    if hasattr(self.context, 'metadata') else None
                )
            }

            # Add any additional metadata from context
            if hasattr(self.context, 'metadata') and isinstance(self.context.metadata, dict):
                for key, value in self.context.metadata.items():
                    if key not in doc_context:
                        doc_context[key] = value

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