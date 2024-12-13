# -*- coding: utf-8 -*-
"""This module contains the DocumentationOrchestrator class for generating documentation.

The DocumentationOrchestrator is responsible for:
- Extracting code elements from source code
- Generating documentation using AI services
- Creating markdown documentation
- Validating the generated documentation

Note:
    This file has been configured to ignore line too long errors (E501) for readability.

"""

import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from core.docstring_processor import DocstringProcessor
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.extraction.code_extractor import CodeExtractor, ExtractionResult
from core.markdown_generator import MarkdownGenerator
from core.response_parsing import ResponseParsingService
from core.ai_service import AIService
from core.prompt_manager import PromptManager
from core.types.base import (
    Injector,
    DocstringData,
    DocumentationContext,
    DocumentationData,
    ExtractionContext,
    ExtractedClass,
    ExtractedFunction,
    ProcessingResult,
)
from core.exceptions import DocumentationError
from utils import ensure_directory, read_file_safe_async
from core.console import (
    print_info,
    print_error,
    create_progress
)


class DocumentationOrchestrator:
    """
    Orchestrates the entire documentation generation process.
    """

    def __init__(
        self,
        ai_service: AIService,
        code_extractor: CodeExtractor,
        markdown_generator: MarkdownGenerator,
        prompt_manager: PromptManager,
        docstring_processor: DocstringProcessor,
        response_parser: ResponseParsingService,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Initialize DocumentationOrchestrator with necessary services.

        Args:
            ai_service: AI service for documentation generation.
            code_extractor: Service for extracting code elements.
            markdown_generator: Service for generating markdown documentation.
            prompt_manager: Service for creating prompts for the AI model.
            docstring_processor: Service for processing docstrings.
            response_parser: Service for parsing AI responses.
            correlation_id: Unique identifier for tracking operations.
        """
        self.correlation_id = correlation_id or str(uuid.uuid4())
        print_info(
            f"Initializing DocumentationOrchestrator with correlation ID: {self.correlation_id}"
        )
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__),
            extra={"correlation_id": self.correlation_id},
        )

        # Use constructor injection for dependencies
        self.ai_service = ai_service
        self.code_extractor = code_extractor
        self.markdown_generator = markdown_generator
        self.prompt_manager = prompt_manager
        self.docstring_processor = docstring_processor
        self.response_parser = response_parser

        self.progress = None  # Initialize progress here

    async def generate_documentation(
        self, context: DocumentationContext
    ) -> Tuple[str, str]:
        """
        Generates documentation for the given context.

        Args:
            context: Documentation context containing source code and metadata.

        Returns:
            Tuple[str, str]: The updated source code and generated markdown documentation.

        Raises:
            DocumentationError: If there's an issue generating documentation.
        """
        try:
            print_info(
                f"Starting documentation generation process with correlation ID: {self.correlation_id}"
            )

            if not context.source_code or not context.source_code.strip():
                raise DocumentationError("Source code is empty or missing")

            async with create_progress() as progress:
                task = progress.add_task("Generating documentation", total=100)
                progress.update(task, advance=20, description="Extracting code...")

                extraction_context = self._create_extraction_context(context)
                extraction_result = await self.code_extractor.extract_code(context.source_code)
                context.classes = [
                    self._create_extracted_class(cls) for cls in extraction_result.classes
                ]
                context.functions = [
                    self._create_extracted_function(func)
                    for func in extraction_result.functions
                ]

                # Create documentation prompt
                prompt = await self.prompt_manager.create_documentation_prompt(
                    module_name=context.metadata.get("module_name", ""),
                    file_path=str(context.module_path),
                    source_code=context.source_code,
                    classes=context.classes,
                    functions=context.functions,
                )

                # Generate documentation through AI service
                processing_result = await self.ai_service.generate_documentation(
                    DocumentationContext(
                        source_code=prompt,
                        module_path=context.module_path,
                        include_source=False,
                        metadata=context.metadata,
                    )
                )
                progress.update(
                    task, advance=50, description="Generating documentation..."
                )

                # Parse and validate the AI response
                parsed_response = await self.response_parser.parse_response(
                    processing_result.content,
                    expected_format="docstring",
                    validate_schema=True,
                )
                self.logger.info(
                    f"AI response parsed and validated with status: {parsed_response.validation_success}"
                )

                # Process and validate the docstring
                docstring_data = await self.docstring_processor.parse(parsed_response.content)
                is_valid, validation_errors = self.docstring_processor.validate(
                    docstring_data
                )
                self.logger.info(f"Docstring validation status: {is_valid}")

                if not is_valid:
                    self.logger.warning(
                        f"Docstring validation failed: {', '.join(validation_errors)}"
                    )
                    self._handle_docstring_validation_errors(validation_errors)

                documentation_data = self._create_documentation_data(
                    context, processing_result, extraction_result, docstring_data
                )

                markdown_doc = self.markdown_generator.generate(documentation_data)
                progress.update(task, advance=30, description="Generating markdown...")

            print_info(
                f"Documentation generation completed successfully with correlation ID: {self.correlation_id}"
            )

            return context.source_code, markdown_doc

        except DocumentationError as de:
            error_msg = (
                f"DocumentationError: {de} in documentation_generation for module_path: "
                f"{context.module_path} with correlation ID: {self.correlation_id}"
            )
            print_error(error_msg)
            self.logger.error(error_msg, extra={"correlation_id": self.correlation_id})
            raise
        except Exception as e:
            error_msg = (
                f"Unexpected error: {e} in documentation_generation for module_path: "
                f"{context.module_path} with correlation ID: {self.correlation_id}"
            )
            print_error(error_msg)
            self.logger.error(
                error_msg, exc_info=True, extra={"correlation_id": self.correlation_id}
            )
            raise DocumentationError(f"Failed to generate documentation: {e}") from e

    def _create_extraction_context(
        self, context: DocumentationContext
    ) -> ExtractionContext:
        """
        Create an extraction context from the given documentation context.

        Args:
            context: Documentation context to extract from.

        Returns:
            ExtractionContext: Context for code extraction.
        """
        return ExtractionContext(
            module_name=context.metadata.get("module_name", context.module_path.stem),
            source_code=context.source_code,
            base_path=context.module_path,
            metrics_enabled=True,
            include_private=False,
            include_magic=False,
            include_nested=True,
            include_source=True,
        )

    def _create_extracted_class(self, cls_data: ExtractedClass) -> ExtractedClass:
        """
        Creates an ExtractedClass instance from extracted data.

        Args:
            cls_data: Extracted class data.

        Returns:
            ExtractedClass: A formatted ExtractedClass instance.
        """
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
        )

    def _create_extracted_function(
        self, func_data: ExtractedFunction
    ) -> ExtractedFunction:
        """
        Creates an ExtractedFunction instance from extracted data.

        Args:
            func_data: Extracted function data.

        Returns:
            ExtractedFunction: A formatted ExtractedFunction instance.
        """
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
            parent_class=func_data.parent_class,
        )

    def _create_documentation_data(
        self,
        context: DocumentationContext,
        processing_result: ProcessingResult,
        extraction_result: ExtractionResult,
    ) -> DocumentationData:
        """
        Create DocumentationData from the given context and AI processing results.

        Args:
            context: The documentation context.
            processing_result: Result from AI documentation generation.
            extraction_result: Result from code extraction.

        Returns:
            DocumentationData: Structured documentation data.
        """
        docstring_data = DocstringData(
            summary=processing_result.content.get("summary", ""),
            description=processing_result.content.get("description", ""),
            args=processing_result.content.get("args", []),
            returns=processing_result.content.get(
                "returns", {"type": "None", "description": ""}
            ),
            raises=processing_result.content.get("raises", []),
            complexity=int(extraction_result.maintainability_index or 1),
        )

        return DocumentationData(
            module_name=str(context.metadata.get("module_name", "")),
            module_path=context.module_path,
            module_summary=str(processing_result.content.get("summary", "")),
            source_code=context.source_code,
            docstring_data=docstring_data,
            ai_content=processing_result.content,
            code_metadata={
                "classes": (
                    [cls.to_dict() for cls in context.classes]
                    if context.classes
                    else []
                ),
                "functions": (
                    [func.to_dict() for func in context.functions]
                    if context.functions
                    else []
                ),
                "constants": context.constants or [],
                "maintainability_index": extraction_result.maintainability_index,
                "dependencies": extraction_result.dependencies,
            },
            glossary={},
            changes=[],
            complexity_scores={},
            metrics={},
            validation_status=False,
            validation_errors=[],
        )

    def _validate_documentation_data(
        self, documentation_data: DocumentationData
    ) -> None:
        """
        Validates the provided documentation data for completeness.

        Args:
            documentation_data: The documentation data to validate.

        Raises:
            DocumentationError: If the documentation data is incomplete or invalid.
        """
        if not self.markdown_generator._has_complete_information(documentation_data):
            self.logger.warning(
                "Documentation generated with missing information",
                extra={"correlation_id": self.correlation_id},
            )

    async def generate_module_documentation(
        self, file_path: Path, output_dir: Path, source_code: Optional[str] = None
    ) -> None:
        """
        Generates documentation for a single module file.

        Args:
            file_path: Path to the source file.
            output_dir: Directory to write the output documentation.
            source_code: Optional source code to process; if not provided, it will be read from the file_path.

        Raises:
            DocumentationError: If there's an issue processing the module.
        """
        try:
            source_code = source_code or await read_file_safe_async(file_path)

            context = DocumentationContext(
                source_code=source_code,
                module_path=file_path,
                include_source=True,
                metadata={
                    "file_path": str(file_path),
                    "module_name": file_path.stem,
                    "creation_time": datetime.now().isoformat(),
                },
            )

            self.logger.info(
                f"Generating documentation for {file_path} with correlation ID: {self.correlation_id}, "
                f"module name: {context.metadata.get('module_name', 'Unknown')}"
            )
            output_dir = ensure_directory(output_dir)
            output_path = output_dir / file_path.with_suffix(".md").name

            updated_code, markdown_doc = await self.generate_documentation(context)

            # Write output files
            output_path.write_text(markdown_doc, encoding="utf-8")
            file_path.write_text(updated_code, encoding="utf-8")

            self.logger.info(
                f"Documentation written to {output_path} with correlation ID: {self.correlation_id}"
            )

        except DocumentationError as de:
            error_msg = f"Module documentation generation failed for {file_path}: {de}"
            self.logger.error(f"{error_msg} with correlation ID: {self.correlation_id}")
            raise DocumentationError(error_msg) from de
        except Exception as e:
            error_msg = (
                f"Unexpected error generating documentation for {file_path}: {e}"
            )
            self.logger.error(f"{error_msg} with correlation ID: {self.correlation_id}")
            raise DocumentationError(error_msg) from e

    async def generate_batch_documentation(
        self, file_paths: List[Path], output_dir: Path
    ) -> Dict[Path, bool]:
        """
        Generates documentation for multiple files in batch.

        Args:
            file_paths: List of paths to the source files.
            output_dir: Directory to write the output documentation.

        Returns:
            Dict[Path, bool]: A dictionary with file paths as keys and boolean values indicating success or failure.
        """
        results = {}
        for file_path in file_paths:
            try:
                await self.generate_module_documentation(file_path, output_dir)
                results[file_path] = True
            except DocumentationError as e:
                self.logger.error(
                    f"Failed to generate docs for {file_path}: {e} with correlation ID: {self.correlation_id}"
                )
                results[file_path] = False
            except Exception as e:
                self.logger.error(
                    f"Unexpected error for {file_path}: {e} with correlation ID: {self.correlation_id}"
                )
                results[file_path] = False
        return results

    async def __aenter__(self) -> "DocumentationOrchestrator":
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager."""
        if self.ai_service:
            await self.ai_service.close()
