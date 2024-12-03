# docs.py
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from core.logger import LoggerSetup
from core.response_parsing import ResponseParsingService
from core.extraction.code_extractor import CodeExtractor, ExtractedClass, ExtractedFunction
from core.markdown_generator import MarkdownGenerator
from core.types import DocumentationContext, AIHandler
from exceptions import DocumentationError

class DocStringManager:
    """Orchestrates documentation generation with centralized response parsing."""
    
    def __init__(
        self,
        context: DocumentationContext,
        ai_handler: AIHandler,
        response_parser: Optional[ResponseParsingService] = None,
        markdown_generator: Optional[MarkdownGenerator] = None
    ):
        self.logger = LoggerSetup.get_logger(__name__)
        self.context = context
        self.ai_handler = ai_handler
        self.response_parser = response_parser or ResponseParsingService()
        self.markdown_generator = markdown_generator or MarkdownGenerator()
        self.code_extractor = CodeExtractor()

    async def generate_documentation(self) -> str:
        """Generate complete documentation using centralized parsing."""
        try:
            # Extract code elements
            extraction_result = self.code_extractor.extract_code(self.context.source_code)
            if not extraction_result:
                raise DocumentationError("Code extraction failed")

            # Parse AI-generated documentation if available
            ai_docs = {}
            if self.context.ai_generated:
                try:
                    parsed_response = await self.response_parser.parse_response(
                        response=self.context.ai_generated,
                        expected_format='json' if isinstance(self.context.ai_generated, str) else 'docstring'
                    )

                    if parsed_response.validation_success:
                        ai_docs = parsed_response.content
                    else:
                        self.logger.warning(
                            f"AI documentation parsing had errors: {parsed_response.errors}"
                        )

                except Exception as e:
                    self.logger.warning(f"Failed to parse AI-generated documentation: {e}", exc_info=True)

            # Create documentation context
            doc_context = {
                'module_name': self.context.module_path.stem if self.context.module_path else "Unknown",
                'file_path': str(self.context.module_path) if self.context.module_path else "",
                'description': extraction_result.module_docstring or "No description available.",
                'classes': extraction_result.classes,
                'functions': extraction_result.functions,
                'constants': extraction_result.constants,
                'changes': self.context.changes or [],  # Assuming changes are provided in the context
                'source_code': self.context.source_code if self.context.include_source else None,
                'ai_documentation': ai_docs  # Include AI documentation
            }

            # Generate markdown
            documentation = self.markdown_generator.generate(doc_context)

            # Log successful generation
            self.logger.debug(f"Generated documentation for {self.context.module_path}")

            return documentation

        except Exception as e:
            self.logger.error(f"Documentation generation failed: {e}", exc_info=True)
            raise DocumentationError(f"Failed to generate documentation: {e}")


    def _format_class_info(self, cls: ExtractedClass) -> Dict[str, Any]:
        """Format class information for markdown generation."""
        return {
            'name': cls.name,
            'bases': cls.bases,
            'docstring': cls.docstring,
            'methods': [self._format_method_info(method) for method in cls.methods],
            'metrics': cls.metrics,
            'attributes': [
                {
                    'name': attr['name'],
                    'type': attr['type'],
                    'value': attr['value']
                }
                for attr in (cls.attributes + cls.instance_attributes)
            ]
        }

    def _format_method_info(self, method: ExtractedFunction) -> Dict[str, Any]:
        """Format method information for markdown generation."""
        return {
            'name': method.name,
            'args': [
                {
                    'name': arg.name,
                    'type': arg.type_hint or 'Any',
                    'default_value': arg.default_value if not arg.is_required else None
                }
                for arg in method.args
            ],
            'return_type': method.return_type or 'None',
            'docstring': method.docstring,
            'metrics': method.metrics,
            'is_async': method.is_async
        }

    def _format_function_info(self, func: ExtractedFunction) -> Dict[str, Any]:
        """Format function information for markdown generation."""
        return {
            'name': func.name,
            'args': [
                {
                    'name': arg.name,
                    'type': arg.type_hint or 'Any',
                    'default_value': arg.default_value if not arg.is_required else None
                }
                for arg in func.args
            ],
            'return_type': func.return_type or 'None',
            'docstring': func.docstring,
            'metrics': func.metrics,
            'is_async': func.is_async
        }

    def _format_constant_info(self, const: Dict[str, Any]) -> Dict[str, Any]:
        """Format constant information for markdown generation."""
        return {
            'name': const['name'],
            'type': const['type'],
            'value': const['value']
        }

    
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

            # Update documentation context
            self.context = DocumentationContext(
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

            return self.context.source_code, documentation

        except Exception as e:
            self.logger.error(f"File processing failed: {e}", exc_info=True)
            raise DocumentationError(f"Failed to process file: {e}")