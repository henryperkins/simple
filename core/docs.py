"""
Documentation generation orchestrator.

Coordinates the process of generating documentation from source code files,
using AI services and managing the overall documentation workflow.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.extraction.code_extractor import CodeExtractor
from core.markdown_generator import MarkdownGenerator
from core.ai_service import AIService
from core.cache import Cache
from core.config import Config
from core.logger import LoggerSetup, CorrelationLoggerAdapter, log_error, log_info
from core.metrics_collector import MetricsCollector
from core.types.base import (
    DocstringData, 
    DocumentationContext, 
    DocumentationData,
    ExtractionContext, 
    ProcessingResult,
    ExtractedClass,
    ExtractedFunction
)
from core.exceptions import DocumentationError
from utils import ensure_directory, handle_extraction_error, read_file_safe
import uuid

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
        self.config = Config()
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__),
            correlation_id=self.correlation_id
        )
        self.ai_service = ai_service or AIService(config=self.config.ai, correlation_id=self.correlation_id)
        self.code_extractor = CodeExtractor(correlation_id=self.correlation_id)
        self.markdown_generator = MarkdownGenerator()

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
            self.logger.info("Starting documentation generation process", extra={'correlation_id': self.correlation_id})

            # Extract code information
            extraction_context = ExtractionContext(
                module_name=context.metadata.get("module_name"),
                source_code=context.source_code,
                base_path=context.module_path,
                metrics_enabled=True,
                include_private=False,
                include_magic=False,
                include_nested=True,
                include_source=True
            )

            extraction_result = await self.code_extractor.extract_code(
                context.source_code, 
                extraction_context
            )

            # Update context with extracted information
            context.classes = [
                ExtractedClass(
                    name=cls.name,
                    lineno=cls.lineno,
                    source=cls.source,
                    docstring=cls.docstring,
                    metrics=cls.metrics,
                    dependencies=cls.dependencies,
                    decorators=cls.decorators,
                    complexity_warnings=cls.complexity_warnings,
                    methods=cls.methods,
                    attributes=cls.attributes,
                    instance_attributes=cls.instance_attributes,
                    bases=cls.bases,
                    metaclass=cls.metaclass,
                    is_exception=cls.is_exception,
                    docstring_info=cls.docstring_info
                )
                for cls in extraction_result.classes
            ]

            context.functions = [
                ExtractedFunction(
                    name=func.name,
                    lineno=func.lineno,
                    source=func.source,
                    docstring=func.docstring,
                    metrics=func.metrics,
                    dependencies=func.dependencies,
                    decorators=func.decorators,
                    complexity_warnings=func.complexity_warnings,
                    args=func.args,
                    returns=func.returns,
                    raises=func.raises,
                    body_summary=func.body_summary,
                    docstring_info=func.docstring_info,
                    is_async=func.is_async,
                    is_method=func.is_method,
                    parent_class=func.parent_class
                )
                for func in extraction_result.functions
            ]

            # Enhance with AI
            processing_result = await self.ai_service.enhance_and_format_docstring(context)

            # Process and validate
            docstring_data = DocstringData(
                summary=processing_result.content.get("summary", ""),
                description=processing_result.content.get("description", ""),
                args=processing_result.content.get("args", []),
                returns=processing_result.content.get("returns", {"type": "None", "description": ""}),
                raises=processing_result.content.get("raises", []),
                complexity=extraction_result.maintainability_index or 1
            )

            # Create documentation data with all required fields
            documentation_data = DocumentationData(
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
                glossary={},  # Added default empty glossary
                changes=[],   # Added default empty changes list
                complexity_scores={},  # Added default empty complexity scores
                metrics={},   # Added default empty metrics
                validation_status=False,  # Added default validation status
                validation_errors=[]  # Added default empty validation errors
            )

            # Generate markdown
            markdown_doc = self.markdown_generator.generate(documentation_data)

            self.logger.info("Documentation generation completed successfully", extra={'correlation_id': self.correlation_id})
            return context.source_code, markdown_doc

        except Exception as e:
            self.logger.error(f"Documentation generation failed: {e}", exc_info=True, extra={'correlation_id': self.correlation_id})
            raise DocumentationError(f"Failed to generate documentation: {e}")

    async def generate_module_documentation(self, file_path: Path, output_dir: Path, source_code: Optional[str] = None) -> None:
        """
        Generate documentation for a single module.

        Args:
            file_path: Path to the module file
            output_dir: Directory where documentation will be output
            source_code: The source code to use (optional)

        Raises:
            DocumentationError: If documentation generation fails
        """
        try:
            self.logger.info(f"Generating documentation for {file_path}", extra={'correlation_id': self.correlation_id})
            output_dir = ensure_directory(output_dir)
            output_path = output_dir / file_path.with_suffix(".md").name

            # Use the provided source_code if available
            if source_code is None:
                source_code = read_file_safe(file_path)
            else:
                # Optionally, write the fixed source code back to the file
                file_path.write_text(source_code, encoding="utf-8")

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

            self.logger.info(f"Documentation written to {output_path}", extra={'correlation_id': self.correlation_id})

        except DocumentationError as de:
            error_msg = f"Module documentation generation failed for {file_path}: {de}"
            self.logger.error(error_msg, extra={'correlation_id': self.correlation_id})
            raise DocumentationError(error_msg) from de
        except Exception as e:
            error_msg = f"Unexpected error generating documentation for {file_path}: {e}"
            self.logger.error(error_msg, extra={'correlation_id': self.correlation_id})
            raise DocumentationError(error_msg) from e

    async def generate_batch_documentation(
        self,
        file_paths: List[Path],
        output_dir: Path
    ) -> Dict[Path, bool]:
        """
        Generate documentation for multiple files.

        Args:
            file_paths: List of file paths to process
            output_dir: Output directory for documentation

        Returns:
            Dictionary mapping file paths to success status
        """
        results = {}
        for file_path in file_paths:
            try:
                await self.generate_module_documentation(file_path, output_dir)
                results[file_path] = True
            except DocumentationError as e:
                self.logger.error(f"Failed to generate docs for {file_path}: {e}", extra={'correlation_id': self.correlation_id})
                results[file_path] = False
            except Exception as e:
                self.logger.error(f"Unexpected error for {file_path}: {e}", extra={'correlation_id': self.correlation_id})
                results[file_path] = False
        return results

    async def __aenter__(self) -> "DocumentationOrchestrator":
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager."""
        if self.ai_service:
            await self.ai_service.close()
