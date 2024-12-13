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
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import ast

# Group imports from core package
from core.docstring_processor import DocstringProcessor
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.extraction.code_extractor import CodeExtractor
from core.markdown_generator import MarkdownGenerator
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
    MetricData,
    ExtractedArgument,
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
        response_parser: Any,
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
        self.logger.info(
            "DocumentationOrchestrator initialized",
            extra={
                "correlation_id": self.correlation_id,
                "ai_service": str(ai_service),
            },
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
            module_name = (
                context.metadata.get("module_name", "") if context.metadata else ""
            )

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
                # Accessing protected method for internal use
                variables = self.code_extractor._extract_variables(tree)

                self.progress.update(
                    extraction_task, advance=10, description="Extracting constants..."
                )
                # Accessing protected method for internal use
                constants = self.code_extractor._extract_constants(tree)

                self.progress.update(
                    extraction_task, advance=10, description="Extracting docstrings..."
                )
                # Accessing protected method for internal use
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
                module_name=(
                    context.metadata.get("module_name", "") if context.metadata else ""
                ),
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

            documentation_data = DocumentationData(
                module_name=(
                    str(context.metadata.get("module_name", "")) if context.metadata else ""
                ),
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

    def _validate_source_code(self, source_code: str) -> None:
        """
        Validates the source code for any issues before processing.

        Args:
            source_code (str): The source code to validate.

        Raises:
            DocumentationError: If the source code is invalid or contains errors.
        """
        try:
            ast.parse(source_code)
        except SyntaxError as e:
            raise DocumentationError(f"Syntax error in source code: {e}")
        # Add more validation checks as needed

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
