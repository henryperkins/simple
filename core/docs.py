"""
Documentation generation orchestrator.

Coordinates the process of generating documentation from source code files,
using AI services and managing the overall documentation workflow.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.extraction.code_extractor import CodeExtractor
from core.markdown_generator import MarkdownGenerator
from core.ai_service import AIService
from core.cache import Cache
from core.types.base import Injector
from core.types.base import (
    DocstringData,
    DocumentationContext,
    DocumentationData,
    ExtractionContext,
    ExtractedClass,
    ExtractedFunction,
    ProcessingResult
)
from core.exceptions import DocumentationError
from utils import ensure_directory, read_file_safe
from core.console import (
    print_info,
    print_error,
    print_warning,
    display_metrics,
    create_progress
)

class DocumentationOrchestrator:
    """
    Orchestrates the process of generating documentation from source code.

    This class manages the end-to-end process of analyzing source code,
    generating documentation using AI, and producing formatted output.
    """

    def __init__(self, ai_service: Optional[AIService] = None, correlation_id: Optional[str] = None) -> None:
        """
        Initialize the DocumentationOrchestrator.

        Args:
            ai_service: Service for AI interactions. Created if not provided.
            correlation_id: Optional correlation ID for tracking related operations
        """
        self.correlation_id = correlation_id or str(uuid.uuid4())
        print_info(f"Initializing DocumentationOrchestrator")
        self.logger = CorrelationLoggerAdapter(LoggerSetup.get_logger(__name__))
        self.ai_service = ai_service or Injector.get('ai_service')
        self.code_extractor = CodeExtractor()
        self.markdown_generator = Injector.get('markdown_generator')

    async def generate_documentation(self, context: DocumentationContext) -> Tuple[str, str]:
        """
        Generate documentation for the given source code.

        Args:
            context: Information about the source code and its environment.

        Returns:
            Updated source code and generated markdown documentation.

        Raises:
            DocumentationError: If documentation generation fails.
        """
        try:
            print_info(f"Starting documentation generation process with correlation ID: {self.correlation_id}")

            if not context.source_code or not context.source_code.strip():
                raise DocumentationError("Source code is empty or missing")

            with create_progress() as progress:
                task = progress.add_task("Generating documentation", total=100)
                progress.update(task, advance=20, description="Extracting code...")

                extraction_context = self._create_extraction_context(context)
                extraction_result = await self.code_extractor.extract_code(
                    context.source_code, extraction_context
                )
                context.classes = [
                    self._create_extracted_class(cls) for cls in extraction_result.classes
                ]
                context.functions = [
                    self._create_extracted_function(func) for func in extraction_result.functions
                ]

                processing_result = await self.ai_service.generate_documentation(context)

                documentation_data = self._create_documentation_data(
                    context, processing_result, extraction_result
                )

                markdown_doc = self.markdown_generator.generate(documentation_data)
                progress.update(task, advance=80, description="Generating markdown...")

                self._validate_documentation_data(documentation_data)

                print_info(f"Documentation generation completed successfully with correlation ID: {self.correlation_id}")
                return context.source_code, markdown_doc

        except Exception as e:
            print_error(f"Error: {e} in documentation_generation for module_path: {context.module_path} with correlation ID: {self.correlation_id}")
            raise DocumentationError(f"Failed to generate documentation: {e}") from e

    def _create_extraction_context(self, context: DocumentationContext) -> ExtractionContext:
        """Creates an extraction context from the documentation context."""
        return ExtractionContext(
            module_name=context.metadata.get("module_name", context.module_path.stem),
            source_code=context.source_code,
            base_path=context.module_path,
            metrics_enabled=True,
            include_private=False,
            include_magic=False,
            include_nested=True,
            include_source=True
        )

    def _create_extracted_class(self, cls_data: Dict[str, Any]) -> ExtractedClass:
        """Creates an ExtractedClass instance from extracted data."""
        return ExtractedClass(
            name=cls_data.name,
            lineno=cls_data.lineno,
            source=cls_data.source,
            docstring=cls_data.docstring,
            metrics=cls_data.metrics,
            dependencies=cls_data.dependencies,
            decorators=cls_data.decorators,
            complexity_warnings=cls_data.complexity_warnings,
            methods=cls_data.methods,
            attributes=cls_data.attributes,
            instance_attributes=cls_data.instance_attributes,
            bases=cls_data.bases,
            metaclass=cls_data.metaclass,
            is_exception=cls_data.is_exception,
            docstring_info=cls_data.docstring_info
        )

    def _create_extracted_function(self, func_data: Dict[str, Any]) -> ExtractedFunction:
        """Creates an ExtractedFunction instance from extracted data."""
        return ExtractedFunction(
            name=func_data.name,
            lineno=func_data.lineno,
            source=func_data.source,
            docstring=func_data.docstring,
            metrics=func_data.metrics,
            dependencies=func_data.dependencies,
            decorators=func_data.decorators,
            complexity_warnings=func_data.complexity_warnings,
            args=func_data.args,
            returns=func_data.returns,
            raises=func_data.raises,
            body_summary=func_data.body_summary,
            docstring_info=func_data.docstring_info,
            is_async=func_data.is_async,
            is_method=func_data.is_method,
            parent_class=func_data.parent_class
        )

    def _create_documentation_data(self, context: DocumentationContext, processing_result: ProcessingResult, extraction_result: ExtractionResult) -> DocumentationData:
        """Creates a DocumentationData instance from processing and extraction results."""
        docstring_data = DocstringData(
            summary=processing_result.content.get("summary", ""),
            description=processing_result.content.get("description", ""),
            args=processing_result.content.get("args", []),
            returns=processing_result.content.get("returns", {"type": "None", "description": ""}),
            raises=processing_result.content.get("raises", []),
            complexity=extraction_result.maintainability_index or 1
        )

        return DocumentationData(
            module_name=str(context.metadata.get("module_name", "")),
            module_path=context.module_path,
            module_summary=str(processing_result.content.get("summary", "")),
            source_code=context.source_code,
            docstring_data=docstring_data,
            ai_content=processing_result.content,
            code_metadata={
                "classes": [cls.to_dict() for cls in context.classes] if context.classes else [],
                "functions": [func.to_dict() for func in context.functions] if context.functions else [],
                "constants": context.constants or [],
                "maintainability_index": extraction_result.maintainability_index,
                "dependencies": extraction_result.dependencies
            },
            glossary={},
            changes=[],
            complexity_scores={},
            metrics={},
            validation_status=False,
            validation_errors=[]
        )

    def _validate_documentation_data(self, documentation_data: DocumentationData) -> None:
        """Validates the generated documentation data."""
        if not self.markdown_generator.has_complete_information(documentation_data):
            self.logger.warning("Documentation generated with missing information", extra={
                'correlation_id': self.correlation_id})

    async def generate_module_documentation(self, file_path: Path, output_dir: Path, source_code: Optional[str] = None) -> None:
        """
        Generate documentation for a single module.

        Args:
            file_path: Path to the module file.
            output_dir: Directory where documentation will be output.
            source_code: The source code to use (optional).

        Raises:
            DocumentationError: If documentation generation fails.
        """
        try:
            self.logger.info(f"Generating documentation for {file_path} with correlation ID: {self.correlation_id}, module name: {context.metadata.get('module_name', 'Unknown')}")
            output_dir = ensure_directory(output_dir)
            output_path = output_dir / file_path.with_suffix(".md").name

            source_code = source_code or read_file_safe(file_path)

            context = DocumentationContext(
                source_code=source_code,
                module_path=file_path,
                include_source=True,
                metadata={
                    "file_path": str(file_path),
                    "module_name": file_path.stem,
                    "creation_time": datetime.now().isoformat(),
                }
            )

            updated_code, markdown_doc = await self.generate_documentation(context)

            # Write output files
            output_path.write_text(markdown_doc, encoding="utf-8")
            file_path.write_text(updated_code, encoding="utf-8")

            self.logger.info(f"Documentation written to {output_path} with correlation ID: {self.correlation_id}")

        except DocumentationError as de:
            error_msg = f"Module documentation generation failed for {file_path}: {de}"
            self.logger.error(f"{error_msg} with correlation ID: {self.correlation_id}")
            raise DocumentationError(error_msg) from de
        except Exception as e:
            error_msg = f"Unexpected error generating documentation for {file_path}: {e}"
            self.logger.error(f"{error_msg} with correlation ID: {self.correlation_id}")
            raise DocumentationError(error_msg) from e

    async def generate_batch_documentation(
        self,
        file_paths: List[Path],
        output_dir: Path
    ) -> Dict[Path, bool]:
        """
        Generate documentation for multiple files.

        Args:
            file_paths: List of file paths to process.
            output_dir: Output directory for documentation.

        Returns:
            Dictionary mapping file paths to success status.
        """
        results = {}
        for file_path in file_paths:
            try:
                await self.generate_module_documentation(file_path, output_dir)
                results[file_path] = True
            except DocumentationError as e:
                self.logger.error(f"Failed to generate docs for {file_path}: {e} with correlation ID: {self.correlation_id}")
                results[file_path] = False
            except Exception as e:
                self.logger.error(f"Unexpected error for {file_path}: {e} with correlation ID: {self.correlation_id}")
                results[file_path] = False
        return results

    async def __aenter__(self) -> "DocumentationOrchestrator":
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager."""
        if self.ai_service:
            await self.ai_service.close()
