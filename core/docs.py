import uuid
from pathlib import Path
from typing import Any, cast, Optional, Dict
from datetime import datetime

# Group imports from core package
from core.docstring_processor import DocstringProcessor
from core.logger import LoggerSetup
from core.markdown_generator import MarkdownGenerator
from core.ai_service import AIService
from core.prompt_manager import PromptManager
from core.types.base import (
    DocumentationContext,
    DocumentationData,
    ExtractedClass,
    ExtractedFunction,
    ProcessingResult,
    MetricData,
    ExtractionResult,
    ParsedResponse
)
from core.types.docstring import DocstringData
from core.exceptions import DocumentationError
from core.metrics_collector import MetricsCollector
from core.extraction.code_extractor import CodeExtractor
from core.response_parsing import ResponseParsingService  # Correct import for ResponseParser
from utils import ensure_directory, read_file_safe_async


class DocumentationOrchestrator:
    """
    Orchestrates the entire documentation generation process.
    
    This class coordinates the interaction between various components to generate
    documentation, ensuring proper type usage and data flow throughout the process.
    """

    def __init__(
        self,
        ai_service: AIService,
        code_extractor: CodeExtractor,
        markdown_generator: MarkdownGenerator,
        prompt_manager: PromptManager,
        docstring_processor: DocstringProcessor,
        response_parser: ResponseParsingService,
        correlation_id: str | None = None,
    ) -> None:
        """Initialize DocumentationOrchestrator with typed dependencies."""
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}",
            correlation_id=self.correlation_id,
        )

        # Initialize metrics collection
        self.metrics_collector = MetricsCollector(correlation_id=self.correlation_id)

        # Store typed dependencies
        self.ai_service = ai_service
        self.code_extractor = code_extractor
        self.markdown_generator = markdown_generator
        self.prompt_manager = prompt_manager
        self.docstring_processor = docstring_processor
        self.response_parser = response_parser

    async def generate_documentation(
        self, 
        context: DocumentationContext
    ) -> tuple[str, str]:
        """
        Generates documentation for the given context.

        Args:
            context: Structured documentation context containing all necessary information

        Returns:
            Tuple of (original_source, markdown_documentation)

        Raises:
            DocumentationError: If documentation generation fails
        """
        start_time = datetime.now()
        module_name = ""
        log_extra = {"correlation_id": self.correlation_id}

        try:
            # Step 1: Validate source code
            if not context or not context.source_code:
                raise DocumentationError(
                    f"Source code is missing or context is invalid for {context.module_path}"
                )

            original_source = context.source_code.strip()
            if not original_source:
                raise DocumentationError(
                    f"Source code is empty after stripping whitespace for {context.module_path}"
                )

            module_name = context.metadata.get("module_name", "") if context.metadata else ""

            # Step 2: Extract code elements
            extraction_result: ExtractionResult = await self.code_extractor.extract_code(
                original_source
            )

            if not extraction_result.source_code:
                raise DocumentationError("Extraction failed - no valid code elements found.")

            # Convert extracted elements to proper types
            classes = self._convert_to_extracted_classes(extraction_result.classes)
            functions = self._convert_to_extracted_functions(extraction_result.functions)

            # Step 3: Create AI prompt
            prompt_result = await self.prompt_manager.create_documentation_prompt(
                DocumentationContext(
                    source_code=original_source,
                    module_path=context.module_path,
                    include_source=True,
                    metadata=context.metadata,
                    classes=classes,
                    functions=functions
                )
            )

            # Step 4: Generate documentation with AI service
            processing_result: ProcessingResult = await self.ai_service.generate_documentation(
                context,
                schema=None  # Add schema if needed
            )

            # Step 5: Parse AI response
            parsed_response: ParsedResponse = await self.response_parser.parse_response(
                processing_result.content,
                expected_format="docstring",
                validate_schema=True
            )

            # Step 6: Create documentation data
            documentation_data = DocumentationData(
                module_name=module_name,
                module_path=context.module_path or Path(),
                module_summary=str(parsed_response.content.get("summary", "")),
                source_code=original_source,
                docstring_data=self._create_docstring_data(parsed_response.content),
                ai_content=processing_result.content,
                code_metadata={
                    "classes": [
                        cls.to_dict() for cls in classes
                    ],  # Convert to dictionaries
                    "functions": [
                        func.to_dict() for func in functions
                    ],  # Convert to dictionaries
                    "variables": extraction_result.variables or [],
                    "constants": extraction_result.constants or [],
                    "module_docstring": extraction_result.module_docstring,
                    "source_code": original_source,
                },
            )

            # Step 7: Generate markdown
            markdown_doc = self.markdown_generator.generate(documentation_data)

            # Step 8: Track metrics
            await self._track_generation_metrics(
                start_time=start_time,
                module_name=module_name,
                processing_result=processing_result
            )

            return original_source, markdown_doc

        except Exception as error:
            await self._handle_generation_error(
                error=error,
                start_time=start_time,
                module_name=module_name
            )
            raise

    def _convert_to_extracted_classes(
        self, 
        classes: list[dict[str, Any]]
    ) -> list[ExtractedClass]:
        """Convert raw class data to ExtractedClass instances."""
        converted_classes = []
        for cls_data in classes:
            if isinstance(cls_data, ExtractedClass):
                converted_classes.append(cls_data)
            else:
                converted_classes.append(ExtractedClass(
                    name=cls_data.get("name", "Unknown"),
                    lineno=cls_data.get("lineno", 0),
                    source=cls_data.get("source"),
                    docstring=cls_data.get("docstring"),
                    methods=self._convert_to_extracted_functions(
                        cls_data.get("methods", [])
                    ),
                    bases=cls_data.get("bases", []),
                    metrics=cls_data.get("metrics", {}),
                    inheritance_chain=cls_data.get("inheritance_chain", [])
                ))
        return converted_classes

    def _convert_to_extracted_functions(
        self, 
        functions: list[dict[str, Any]]
    ) -> list[ExtractedFunction]:
        """Convert raw function data to ExtractedFunction instances."""
        converted_functions = []
        for func_data in functions:
            if isinstance(func_data, ExtractedFunction):
                converted_functions.append(func_data)
            else:
                converted_functions.append(ExtractedFunction(
                    name=func_data.get("name", "Unknown"),
                    lineno=func_data.get("lineno", 0),
                    source=func_data.get("source"),
                    docstring=func_data.get("docstring"),
                    args=func_data.get("args", []),
                    returns=func_data.get("returns", {}),
                    metrics=func_data.get("metrics", {}),
                ))
        return converted_functions

    def _create_docstring_data(self, content: Dict[str, Any]) -> DocstringData:
        """Create DocstringData from content dict."""
        return DocstringData(
            summary=str(content.get("summary", "")),
            description=str(content.get("description", "")),
            args=content.get("args", []),
            returns=content.get("returns", {"type": "Any", "description": ""}),
            raises=content.get("raises", []),
            complexity=int(content.get("complexity", 1)),
        )


    async def _track_generation_metrics(
        self,
        start_time: datetime,
        module_name: str,
        processing_result: ProcessingResult
    ) -> None:
        """Track metrics for documentation generation."""
        processing_time = (datetime.now() - start_time).total_seconds()
        await self.metrics_collector.track_operation(
            operation_type="documentation_generation",
            success=True,
            duration=processing_time,
            metadata={
                "module_name": module_name,
                "processing_time": processing_time,
                "token_usage": processing_result.usage
            }
        )

    async def _handle_generation_error(
        self,
        error: Exception,
        start_time: datetime,
        module_name: str
    ) -> None:
        """Handle errors during documentation generation."""
        processing_time = (datetime.now() - start_time).total_seconds()
        self.logger.error(
            f"Error during documentation generation: {error}",
            exc_info=True,
            extra={"correlation_id": self.correlation_id},
        )
        await self.metrics_collector.track_operation(
            operation_type="documentation_generation",
            success=False,
            duration=processing_time,
            metadata={"module_name": module_name, "error": str(error)},
        )

    async def generate_module_documentation(
        self, file_path: Path, output_dir: Path, source_code: str | None = None
    ) -> None:
        """
        Generates documentation for a single module file.
        ...
        """
        start_time = datetime.now()
        log_extra = {"correlation_id": self.correlation_id}
        try:
            # Validate file type
            if not file_path.suffix == ".py":
                self.logger.warning(f"Skipping non-Python file: {file_path}", extra=log_extra)
                return  # Early exit

            # Read source code if not provided
            if source_code is None:
                self.logger.info(
                    f"Attempting to read source code from {file_path}", extra=log_extra
                )
                source_code = await read_file_safe_async(file_path)

            if source_code:
                self.logger.info(
                    f"Source code read from {file_path}. Length: {len(source_code)}",
                    extra=log_extra,
                )
            else:
                error_msg = f"Source code is missing or empty for {file_path}"
                self.logger.warning(error_msg, extra=log_extra)
                return  # Early exit for empty files

            # Prepare context for documentation generation
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

            self.logger.info(f"Generating documentation for {file_path}", extra=log_extra)
            output_dir = ensure_directory(output_dir)
            output_path = output_dir / file_path.with_suffix(".md").name

            # Generate documentation
            updated_code, markdown_doc = await self.generate_documentation(context)

            # Write outputs
            output_path.write_text(markdown_doc, encoding="utf-8")
            if updated_code:
                file_path.write_text(updated_code, encoding="utf-8")

            processing_time = (datetime.now() - start_time).total_seconds()
            await self.metrics_collector.track_operation(
                operation_type="module_documentation_generation",
                success=True,
                duration=processing_time,
                metadata={"module_path": str(file_path)},
            )
            self.logger.info(f"Documentation written to {output_path}", extra=log_extra)
            self.logger.info(
                f"Documentation generation completed in {processing_time:.2f}s",
                extra=log_extra
            )

        except DocumentationError as doc_error:
            error_msg = f"Module documentation generation failed for {file_path}: {doc_error}"
            self.logger.error(error_msg, exc_info=True, extra=log_extra)
            raise DocumentationError(error_msg) from doc_error
        except Exception as gen_error:
            error_msg = f"Unexpected error generating documentation for {file_path}: {gen_error}"
            self.logger.error(error_msg, exc_info=True, extra=log_extra)
            processing_time = (datetime.now() - start_time).total_seconds()
            await self.metrics_collector.track_operation(
                operation_type="module_documentation_generation",
                success=False,
                duration=processing_time,
                metadata={"module_path": str(file_path), "error": str(gen_error)},
            )
            raise DocumentationError(error_msg) from gen_error
