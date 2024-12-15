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
from typing import Any, Optional, Tuple, List
from datetime import datetime
import ast
import time

# Group imports from core package
from core.docstring_processor import DocstringProcessor
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.extraction.code_extractor import CodeExtractor
from core.markdown_generator import MarkdownGenerator
from core.ai_service import AIService
from core.prompt_manager import PromptManager
from core.types.base import (
    DocumentationContext,
    DocumentationData,
)
from core.exceptions import DocumentationError
from utils import ensure_directory, read_file_safe_async


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
            if not context.source_code or not context.source_code.strip():
                self.logger.warning(
                    f"Skipping documentation generation for {context.module_path} as the source code is empty or missing."
                )
                return context.source_code, ""

            source_code = context.source_code
            module_name = (
                context.metadata.get("module_name", "") if context.metadata else ""
            )

            # Initialize variables to ensure they are always defined
            classes, functions, variables, constants, module_docstring = (
                [],
                [],
                [],
                [],
                None,
            )

            self._validate_source_code(source_code)
            tree = ast.parse(source_code)

            self.code_extractor.dependency_analyzer.analyze_dependencies(tree)
            classes = await self.code_extractor.class_extractor.extract_classes(tree)
            functions = (
                await self.code_extractor.function_extractor.extract_functions(tree)
            )
            variables = self.code_extractor.extract_variables(tree)
            constants = self.code_extractor.extract_constants(tree)
            module_docstring = self.code_extractor.extract_module_docstring(tree)
            self.code_extractor.metrics.calculate_metrics(source_code, module_name)

            prompt = await self.prompt_manager.create_documentation_prompt(
                module_name=module_name,
                file_path=str(context.module_path),
                source_code=context.source_code,
                classes=classes,
                functions=functions,
            )

            processing_result = await self.ai_service.generate_documentation(
                DocumentationContext(
                    source_code=prompt,
                    module_path=context.module_path,
                    include_source=False,
                    metadata=context.metadata,
                )
            )

            parsed_response = await self.response_parser.parse_response(
                processing_result.content,
                expected_format="docstring",
                validate_schema=False,  # Removed validation
            )

            # Handle different response content types
            try:
                if isinstance(parsed_response.content, (str, dict)):
                    docstring_data = self.docstring_processor.parse(parsed_response.content)
                else:
                    raise ValueError(f"Unexpected response content type: {type(parsed_response.content)}")
            except Exception as ve:
                self.logger.error(f"Docstring processing error: {ve}")
                raise DocumentationError(f"Failed to process docstring: {ve}")

            documentation_data = DocumentationData(
                module_name=module_name,
                module_path=context.module_path,
                module_summary=str(processing_result.content.get("summary", "")),
                source_code=context.source_code,
                docstring_data=docstring_data,
                ai_content=processing_result.content,
                code_metadata={
                    "classes": [cls.__dict__ for cls in classes] if classes else [],
                    "functions": [func.__dict__ for func in functions]
                    if functions
                    else [],
                    "variables": variables or [],
                    "constants": constants or [],
                    "module_docstring": module_docstring,
                },
            )

            markdown_doc = self.markdown_generator.generate(documentation_data)
            return context.source_code, markdown_doc

        except DocumentationError as de:
            error_msg = f"DocumentationError: {de} in documentation_generation for module_path: {context.module_path}"
            self.logger.error(error_msg, extra={"correlation_id": self.correlation_id})
            raise
        except Exception as e:
            error_msg = (
                f"Unexpected error: {e} in documentation_generation for module_path: "
                f"{context.module_path}"
            )
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
            raise DocumentationError(f"Syntax error in source code: {e}") from e

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
            # Ensure source_code is not None or empty
            if source_code is None or not source_code.strip():
                self.logger.info(f"Attempting to read source code from {file_path}")
                source_code = await read_file_safe_async(file_path)
                self.logger.info(f"Source code read from {file_path}. Length: {len(source_code)}")
                if not source_code or not source_code.strip():
                    self.logger.error(f"Source code is missing or empty for {file_path}")
                    raise DocumentationError(f"Source code is missing or empty for {file_path}")

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
            output_dir = ensure_directory(output_dir)
            output_path = output_dir / file_path.with_suffix(".md").name

            updated_code, markdown_doc = await self.generate_documentation(context)

            output_path.write_text(markdown_doc, encoding="utf-8")
            if updated_code:
                file_path.write_text(updated_code, encoding="utf-8")

            self.logger.info(
                f"Documentation written to {output_path}"
            )

        except DocumentationError as de:
            error_msg = f"Module documentation generation failed for {file_path}: {de}"
            self.logger.error(error_msg)
            raise DocumentationError(error_msg) from de
        except Exception as e:
            error_msg = (
                f"Unexpected error generating documentation for {file_path}: {e}"
            )
            self.logger.error(error_msg)
            raise DocumentationError(error_msg) from e
