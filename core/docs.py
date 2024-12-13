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
import ast

from core.docstring_processor import DocstringProcessor
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.extraction.code_extractor import CodeExtractor
from core.markdown_generator import MarkdownGenerator
from core.response_parsing import ResponseParsingService
from core.ai_service import AIService
from core.prompt_manager import PromptManager
from core.types.base import (
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
from core.console import print_info, print_error, create_progress


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

            source_code = context.source_code
            module_name = context.metadata.get("module_name", "")

            # Reuse existing progress bar if available
            if self.progress:
                self.progress.stop()
            progress = create_progress()
            progress.start()
            self.progress = progress

            extraction_task = self.progress.add_task(
                "Extracting code elements", total=100
            )

            # Initialize variables to ensure they are always defined
            classes, functions, variables, constants, module_docstring = (
                [],
                [],
                [],
                [],
                None,
            )

            try:
                self.progress.update(
                    extraction_task, advance=10, description="Validating source code..."
                )
                self._validate_source_code(source_code)

                self.progress.update(
                    extraction_task, advance=10, description="Parsing AST..."
                )
                tree = ast.parse(source_code)

                self.progress.update(
                    extraction_task,
                    advance=10,
                    description="Extracting dependencies...",
                )
                self.code_extractor.dependency_analyzer.analyze_dependencies(tree)

                self.progress.update(
                    extraction_task, advance=15, description="Extracting classes..."
                )
                classes = await self.code_extractor.class_extractor.extract_classes(
                    tree
                )

                self.progress.update(
                    extraction_task, advance=15, description="Extracting functions..."
                )
                functions = (
                    await self.code_extractor.function_extractor.extract_functions(tree)
                )

                self.progress.update(
                    extraction_task, advance=10, description="Extracting variables..."
                )
                variables = self.code_extractor._extract_variables(tree)

                self.progress.update(
                    extraction_task, advance=10, description="Extracting constants..."
                )
                constants = self.code_extractor._extract_constants(tree)

                self.progress.update(
                    extraction_task, advance=10, description="Extracting docstrings..."
                )
                module_docstring = self.code_extractor._extract_module_docstring(tree)

                self.progress.update(
                    extraction_task, advance=10, description="Calculating metrics..."
                )
                self.code_extractor.metrics.calculate_metrics(source_code, module_name)
            finally:
                if self.progress:
                    self.progress.stop()
                    self.progress = None

            # Create documentation prompt
            prompt = await self.prompt_manager.create_documentation_prompt(
                module_name=context.metadata.get("module_name", ""),
                file_path=str(context.module_path),
                source_code=context.source_code,
                classes=classes,
                functions=functions,
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
            docstring_data = self.docstring_processor(parsed_response.content)
            is_valid, validation_errors = self.docstring_processor.validate(
                docstring_data
            )
            self.logger.info(f"Docstring validation status: {is_valid}")

            if not is_valid:
                self.logger.warning(
                    f"Docstring validation failed: {', '.join(validation_errors)}"
                )

            documentation_data = self._create_documentation_data(
                context,
                processing_result,
                docstring_data,
                classes,
                functions,
                variables,
                constants,
                module_docstring,
            )

            markdown_doc = self.markdown_generator.generate(documentation_data)

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
            is_async=func_data.is_async,
            is_method=func_data.is_method,
            parent_class=func_data.parent_class,
        )

    def _create_documentation_data(
        self,
        context: DocumentationContext,
        processing_result: ProcessingResult,
        docstring_data: DocstringData,
        classes: List[ExtractedClass],
        functions: List[ExtractedFunction],
        variables: List[str],
        constants: List[str],
        module_docstring: Optional[str],
    ) -> DocumentationData:
        """
        Create DocumentationData from the given context and AI processing results.

        Args:
            context: The documentation context.
            processing_result: Result from AI documentation generation.
            docstring_data: Parsed docstring data.
            classes: List of extracted classes.
            functions: List of extracted functions.
            variables: List of extracted variables.
            constants: List of extracted constants.
            module_docstring: The module-level docstring.

        Returns:
            DocumentationData: Structured documentation data.
        """
        return DocumentationData(
            module_name=str(context.metadata.get("module_name", "")),
            module_path=context.module_path,
            module_summary=str(processing_result.content.get("summary", "")),
            source_code=context.source_code,
            docstring_data=docstring_data,
            ai_content=processing_result.content,
            code_metadata={
                "classes": [cls.__dict__ for cls in classes] if classes else [],
                "functions": [func.__dict__ for func in functions] if functions else [],
                "variables": variables or [],
                "constants": constants or [],
                "module_docstring": module_docstring,
                "maintainability_index": None,
                "dependencies": None,
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

    def _validate_source_code(self, source_code: str) -> None:
        """
        Validates the source code for syntax errors.

        Args:
            source_code: The source code to validate.

        Raises:
            DocumentationError: If the source code has syntax errors.
        """
        try:
            ast.parse(source_code)
        except SyntaxError as e:
            raise DocumentationError(f"Syntax error in source code: {e}")
