"""
docs.py - Main module for the documentation generator.
"""

import os
from pathlib import Path
from typing import Any, Optional, Union, List, Dict, Tuple
from core.config import Config
from core.types.base import (
    DocumentationContext,
    ProcessingResult,
    ExtractionResult,
    ExtractedClass,
    ExtractedFunction,
)
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.exceptions import (
    DocumentationError,
    InvalidSourceCodeError,
)
from utils import log_and_raise_error
from core.metrics_collector import MetricsCollector
import time

class DocumentationOrchestrator:
    """
    Orchestrates the documentation generation process.
    """

    def __init__(
        self,
        ai_service: Any,
        code_extractor: Any,
        markdown_generator: Any,
        prompt_manager: Any,
        docstring_processor: Any,
        response_parser: Any,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Initializes the DocumentationOrchestrator with necessary components.
        """
        self.ai_service = ai_service
        self.code_extractor = code_extractor
        self.markdown_generator = markdown_generator
        self.prompt_manager = prompt_manager
        self.docstring_processor = docstring_processor
        self.response_parser = response_parser
        self.correlation_id = correlation_id
        self.logger = LoggerSetup.get_logger(__name__, self.correlation_id)
        self.metrics_collector = MetricsCollector(self.correlation_id)

    async def generate_documentation(
        self, context: DocumentationContext, model: str = "gpt-4"
    ) -> ProcessingResult:
        """
        Generates documentation for the given context.

        Args:
            context: The context for documentation generation.

        Returns:
            A ProcessingResult containing the generated documentation.
        """
        if not context.source_code:
            message = (
                f"Source code is empty for module: {context.module_name}. "
                f"Skipping documentation generation."
            )
            self.logger.warning(message)
            return ProcessingResult(
                content={},
                status_code=204,
                message=message,
                metadata=context.metadata,
            )

        try:
            prompt_result = await self.prompt_manager.create_documentation_prompt(
                context
            )
        except Exception as e:
            log_and_raise_error(
                self.logger,
                e,
                PromptGenerationError,
                "Error generating documentation prompt",
                self.correlation_id,
                details={"context": context},
            )

        if prompt_result.content:
            try:
                processing_result: ProcessingResult = (
                    await self.ai_service.generate_documentation(context, model)
                )
            except Exception as e:
                await self._handle_generation_error(
                    e,
                    "Documentation generation failed",
                    context.module_name,
                    start_time=time.time(),
                    prompt=prompt_result.content.get("prompt", ""),
                )
            parsed_response = self.response_parser.parse(
                processing_result.content, self.prompt_manager.get_function_schema()
            )
            processing_result.content = parsed_response.content
            processing_result.status_code = 200
            processing_result.message = "Documentation generated successfully"

            return processing_result
        else:
            message = "Prompt content is empty. Cannot proceed with documentation generation."
            self.logger.warning(message)
            return ProcessingResult(
                content={}, status_code=204, message=message, metadata=context.metadata
            )

    async def _handle_generation_error(
        self,
        error: Exception,
        message: str,
        module_name: str,
        start_time: Optional[float] = None,
        prompt: Optional[str] = None,
        function_name: Optional[str] = None,
        class_name: Optional[str] = None,
    ) -> None:
        """
        Handles errors during documentation generation, including logging and metrics tracking.

        Args:
            error: The exception that occurred.
            message: The error message.
            module_name: The name of the module.
            start_time: The start time of the operation, if available.
            prompt: The prompt used, if available.
            function_name: The name of the function, if applicable.
            class_name: The name of the class, if applicable.
        """
        if start_time:
            end_time = time.time()
            processing_time = end_time - start_time
        else:
            processing_time = 0

        details = {
            "error_message": str(error),
            "module_name": module_name,
            "processing_time": processing_time,
        }

        if function_name:
            details["function_name"] = function_name
        if class_name:
            details["class_name"] = class_name
        if prompt:
            details["prompt_preview"] = prompt[:50]  # Log only a preview of the prompt

        await self.metrics_collector.track_operation(
            "generate_documentation",
            False,
            processing_time,
            metadata=details,
        )

        log_and_raise_error(
            self.logger,
            error,
            DocumentationError,
            message,
            self.correlation_id,
            **details,
        )

    async def generate_module_documentation(
        self, file_path: Path, output_path: Path, source_code: str
    ) -> None:
        """
        Orchestrates the documentation generation for a module.
        """
        self.logger.info(f"--- Processing Module: {file_path} ---")

        print_status(f"Preparing context for: {file_path.stem}")
        self.code_extractor.context.module_name = file_path.stem
        self.code_extractor.context.file_path = str(file_path)

        print_status(f"Generating documentation for: {file_path.stem}")
        extraction_result = self.code_extractor.extract(source_code)

        # Create a directory for the module within the output path
        module_output_dir = output_path / file_path.stem
        module_output_dir.mkdir(parents=True, exist_ok=True)

        # Process each class separately
        for class_ in extraction_result.classes:
            class_doc_context = DocumentationContext(
                module_path=file_path,
                module_name=file_path.stem,
                source_code=class_.code,
                classes=[class_],
                functions=[],  # Don't include module level functions
                metadata={"file_path": str(file_path)},
            )
            try:
                class_doc_result = await self.generate_documentation(
                    class_doc_context, "gpt-4"
                )
                # Save each class's documentation to a separate file
                class_output_file = module_output_dir / f"{class_.name}.md"
                await self.save_documentation(class_doc_result, class_output_file)
            except Exception as e:
                await self._handle_generation_error(
                    e,
                    f"Error generating documentation for class {class_.name}",
                    file_path.stem,
                    start_time=time.time(),
                    class_name=class_.name,
                )

        # Process each function separately (excluding those in classes)
        for function in extraction_result.functions:
            if function.parent_class is None:  # Check if not part of a class
                func_doc_context = DocumentationContext(
                    module_path=file_path,
                    module_name=file_path.stem,
                    source_code=function.code,
                    classes=[],
                    functions=[function],
                    metadata={"file_path": str(file_path)},
                )
                try:
                    func_doc_result = await self.generate_documentation(
                        func_doc_context, "gpt-4"
                    )
                    if func_doc_result and func_doc_result.content:
                        # Save each function's documentation to a separate file
                        output_file = module_output_dir / f"{function.name}.md"
                        await self.save_documentation(func_doc_result, output_file)
                except Exception as e:
                    await self._handle_generation_error(
                        e,
                        f"Error generating documentation for function {function.name}",
                        file_path.stem,
                        start_time=time.time(),
                        function_name=function.name,
                    )

    async def save_documentation(self, result: ProcessingResult, output_path: Path) -> None:
        """
        Saves the generated documentation to a file.

        Args:
            result: The result of the documentation generation.
            output_path: The path to save the documentation.
        """
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if result.content and isinstance(result.content, dict):
                markdown_content = self.markdown_generator.generate(result.content)
                with open(output_path, "w", encoding="utf-8") as output_file:
                    output_file.write(markdown_content)
                print_success(f"Documentation saved to: {output_path}")
            else:
                self.logger.warning(f"No documentation content to save for {output_path.name}.")
        except Exception as e:
            log_and_raise_error(
                self.logger,
                e,
                DocumentationError,
                "Error saving documentation",
                self.correlation_id,
                file_path=str(output_path),
            )