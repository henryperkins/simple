from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from core.logger import LoggerSetup
from core.response_parsing import ResponseParsingService
from core.extraction.code_extractor import CodeExtractor
from core.extraction.types import ExtractedClass, ExtractedFunction
from core.docstring_processor import DocstringProcessor
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
        docstring_processor: Optional[DocstringProcessor] = None,
        markdown_generator: Optional[MarkdownGenerator] = None
    ):
        """
        Initialize documentation manager.

        Args:
            context: Documentation context
            ai_handler: AI processing handler
            response_parser: Optional response parsing service
            docstring_processor: Optional docstring processor
            markdown_generator: Optional markdown generator
        """
        self.logger = LoggerSetup.get_logger(__name__)
        self.context = context
        self.ai_handler = ai_handler
        self.response_parser = response_parser or ResponseParsingService()
        self.docstring_processor = docstring_processor or DocstringProcessor(self.response_parser)
        self.markdown_generator = markdown_generator or MarkdownGenerator()
        self.code_extractor = CodeExtractor()

    async def generate_documentation(self) -> str:
        """Generate complete documentation using centralized parsing."""
        try:
            # Extract code elements
            extraction_result = self.code_extractor.extract_code(self.context.source_code)
            if not extraction_result:
                raise DocumentationError("Code extraction failed")

            # Process AI-generated documentation if available
            ai_docs = {}
            if self.context.ai_generated:
                try:
                    # Use centralized parser
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
                'metrics': extraction_result.metrics,
                'source_code': self.context.source_code if self.context.include_source else None,
                'ai_documentation': ai_docs
            }

            # Generate markdown
            documentation = self.markdown_generator.generate(doc_context)

            # Log successful generation
            self.logger.debug(f"Generated documentation for {self.context.module_path}")

            return documentation

        except Exception as e:
            self.logger.error(f"Documentation generation failed: {e}", exc_info=True)
            raise DocumentationError(f"Failed to generate documentation: {e}")
    
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
        
    def _format_constants(self, constants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                'name': str(const.get('name', '')),  # Use safe .get() and convert to string
                'type': str(const.get('type', '')),
                'value': str(const.get('value', ''))
            } for const in constants
        ]

    def _format_classes(self, classes: List[ExtractedClass]) -> List[Dict[str, Any]]:
        return [
            {
                'name': cls.name,
                'docstring': cls.docstring,
                'methods': [
                    {   # Format methods correctly
                        'name': m.name,
                        'docstring': m.docstring,
                        'args': m.args,
                        'return_type': m.return_type,
                        'metrics': m.metrics,
                        'source': m.source
                    } for m in cls.methods
                ],
                'bases': cls.bases,
                'metrics': cls.metrics,
                'source': cls.source
            } for cls in classes
        ]

    def _format_functions(self, functions: List[ExtractedFunction]) -> List[Dict[str, Any]]:
        return [
            {
                'name': func.name,
                'docstring': func.docstring,
                'args': func.args,
                'return_type': func.return_type,
                'metrics': func.metrics,
                'source': func.source,
            } for func in functions
        ]