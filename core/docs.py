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
from typing import Any, cast
from datetime import datetime
import ast
import time

# Group imports from core package
from core.docstring_processor import DocstringProcessor
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.markdown_generator import MarkdownGenerator
from core.ai_service import AIService
from core.prompt_manager import PromptManager
from core.types.base import (
    DocumentationContext,
    DocumentationData,
    ExtractedClass,
    ExtractedFunction,
)
from core.exceptions import DocumentationError
from core.metrics_collector import MetricsCollector
from core.console import print_info, print_error, print_success
from utils import ensure_directory, read_file_safe_async


class DocumentationOrchestrator:
    """
    Orchestrates the entire documentation generation process.
    """

    def __init__(
        self,
        ai_service: AIService,
        code_extractor: Any,  # Using Any to avoid circular import
        markdown_generator: MarkdownGenerator,
        prompt_manager: PromptManager,
        docstring_processor: DocstringProcessor,
        response_parser: Any,
        correlation_id: str | None = None,
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
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__),
            extra={"correlation_id": self.correlation_id},
        )
        self.metrics_collector = MetricsCollector(correlation_id=self.correlation_id)
        self.logger.info(
            "DocumentationOrchestrator initialized",
            extra={
                "correlation_id": self.correlation_id,
                "ai_service": str(ai_service),
            },
        )
        print_info("DocumentationOrchestrator initialized.")

        # Use constructor injection for dependencies
        self.ai_service = ai_service
        self.code_extractor = code_extractor
        self.markdown_generator = markdown_generator
        self.prompt_manager = prompt_manager
        self.docstring_processor = docstring_processor
        self.response_parser = response_parser

    async def generate_documentation(self, context: DocumentationContext) -> tuple[str, str]:
        """
        Generates documentation for the given context.

        Args:
            context: Documentation context containing source code and metadata.

        Returns:
            tuple[str, str]: The updated source code and generated markdown documentation.

        Raises:
            DocumentationError: If there's an issue generating documentation. 
        """
        start_time = time.time()
        module_name = ""
        try:
            # Validate source code
            if not context.source_code or not context.source_code.strip():
                self.logger.warning(f"Empty source code for {context.module_path}")
                return context.source_code, ""

            # Store original source code
            original_source = context.source_code
            module_name = context.metadata.get("module_name", "") if context.metadata else ""

            # Extract code elements  
            extraction_result = await self.code_extractor.extract_code(original_source)
            
            if not extraction_result:
                raise DocumentationError("Extraction failed - no result returned")
            
            if not extraction_result.source_code:
                raise DocumentationError("Extraction failed - no source code in result")
                
            # Convert classes and functions
            classes: list[ExtractedClass] = []
            functions: list[ExtractedFunction] = []
            
            if extraction_result.classes:
                for cls_data in extraction_result.classes:
                    if isinstance(cls_data, ExtractedClass):
                        classes.append(cls_data)
                    else:
                        cls_dict = cast(dict[str, Any], cls_data)
                        classes.append(ExtractedClass(
                            name=cls_dict.get("name", "Unknown"),
                            lineno=cls_dict.get("lineno", 0),
                            methods=cls_dict.get("methods", []),
                            bases=cls_dict.get("bases", [])
                        ))

            if extraction_result.functions:
                for func_data in extraction_result.functions:
                    if isinstance(func_data, ExtractedFunction):
                        functions.append(func_data)
                    else:
                        func_dict = cast(dict[str, Any], func_data)
                        functions.append(ExtractedFunction(
                            name=func_dict.get("name", "Unknown"),
                            lineno=func_dict.get("lineno", 0),
                            args=func_dict.get("args", []),
                            returns=func_dict.get("returns", {})
                        ))

            # Generate documentation prompt and process with AI
            await self.prompt_manager.create_documentation_prompt(
                module_name=module_name,
                file_path=str(context.module_path),
                source_code=original_source,
                classes=classes,
                functions=functions,
            )

            # Process with AI service - make a copy of context to avoid modifying original
            response_context = DocumentationContext(
                source_code=original_source,
                module_path=context.module_path,
                include_source=True,
                metadata=context.metadata
            )

            processing_result = await self.ai_service.generate_documentation(response_context)

            if not processing_result or not processing_result.content:
                raise DocumentationError("No content received from AI service")

            # Parse response
            parsed_response = await self.response_parser.parse_response(
                processing_result.content,
                expected_format="docstring",
                validate_schema=False
            )

            if not parsed_response or not parsed_response.content:
                raise DocumentationError("Failed to parse AI response")

            # Create documentation data with original source
            documentation_data = DocumentationData(
                module_name=module_name,
                module_path=context.module_path or Path(),
                module_summary=str(parsed_response.content.get("summary", "")),
                source_code=original_source,
                docstring_data=parsed_response.content,
                ai_content=processing_result.content,
                code_metadata={
                    "classes": [cls.__dict__ for cls in classes],
                    "functions": [func.__dict__ for func in functions],
                    "variables": extraction_result.variables or [],
                    "constants": extraction_result.constants or [],
                    "module_docstring": extraction_result.module_docstring
                }
            )

            # Generate markdown with explicit source code check
            if not documentation_data.source_code:
                raise DocumentationError("Source code missing in documentation data")
                
            markdown_doc = self.markdown_generator.generate(documentation_data)

            # Track metrics
            processing_time = time.time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=True,
                duration=processing_time,
                metadata={"module_name": module_name}
            )

            return original_source, markdown_doc

        except Exception as gen_error:
            self.logger.error(f"Documentation generation failed: {gen_error}", 
                            extra={"correlation_id": self.correlation_id})
            processing_time = time.time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=False,
                duration=processing_time,
                metadata={"module_name": module_name, "error": str(gen_error)}
            )
            raise DocumentationError(f"Failed to generate documentation: {gen_error}") from gen_error

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
        except SyntaxError as syntax_error:
            raise DocumentationError(f"Syntax error in source code: {syntax_error}") from syntax_error

    async def generate_module_documentation(
        self, file_path: Path, output_dir: Path, source_code: str | None = None
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
        start_time = time.time()
        try:
            # Ensure source_code is not None or empty
            if not source_code:
                self.logger.info(f"Attempting to read source code from {file_path}")
                print_info(f"Attempting to read source code from {file_path}")
                source_code = await read_file_safe_async(file_path)
                if source_code:
                    self.logger.info(f"Source code read from {file_path}. Length: {len(source_code)}")
                    print_info(f"Source code read from {file_path}. Length: {len(source_code)}")
                if not source_code or not source_code.strip():
                    error_msg = f"Source code is missing or empty for {file_path}"
                    self.logger.error(error_msg)
                    print_error(error_msg)
                    raise DocumentationError(error_msg)

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
                f"Generating documentation for {file_path}"
            )
            print_info(f"Generating documentation for {file_path}")
            output_dir = ensure_directory(output_dir)
            output_path = output_dir / file_path.with_suffix(".md").name

            updated_code, markdown_doc = await self.generate_documentation(context)

            output_path.write_text(markdown_doc, encoding="utf-8")
            if updated_code:
                file_path.write_text(updated_code, encoding="utf-8")

            processing_time = time.time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="module_documentation_generation",
                success=True,
                duration=processing_time,
                metadata={"module_path": str(file_path)},
            )
            self.logger.info(
                f"Documentation written to {output_path}"
            )
            print_success(f"Documentation written to {output_path} in {processing_time:.2f}s")

        except DocumentationError as doc_error:
            error_msg = f"Module documentation generation failed for {file_path}: {doc_error}"
            self.logger.error(error_msg)
            print_error(error_msg)
            raise DocumentationError(error_msg) from doc_error
        except Exception as gen_error:
            error_msg = f"Unexpected error generating documentation for {file_path}: {gen_error}"
            self.logger.error(error_msg)
            print_error(error_msg)
            processing_time = time.time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="module_documentation_generation",
                success=False,
                duration=processing_time,
                metadata={"module_path": str(file_path), "error": str(gen_error)},
            )
            raise DocumentationError(error_msg) from gen_error
