from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING, Type
from pathlib import Path
from datetime import datetime
import json

if TYPE_CHECKING:
    from ai_interaction import AIInteractionHandler
from core.logger import LoggerSetup
from core.types import DocumentationContext, ExtractionContext, ExtractionResult
from core.extraction.code_extractor import CodeExtractor
from core.docstring_processor import DocstringProcessor
from core.response_parsing import ResponseParsingService
from core.metrics import Metrics
from core.schema_loader import load_schema
from exceptions import DocumentationError

logger = LoggerSetup.get_logger(__name__)


class DocumentationOrchestrator:
    def __init__(
        self,
        ai_handler: Optional['AIInteractionHandler'] = None,  # Use forward reference
        docstring_processor: Optional[DocstringProcessor] = None,
        code_extractor: Optional[CodeExtractor] = None,
        metrics: Optional[Metrics] = None,
        response_parser: Optional[ResponseParsingService] = None
    ) -> None:
        """
        Initialize the documentation orchestrator.

        Args:
            ai_handler: Handler for AI interactions
            docstring_processor: Processor for docstrings
            code_extractor: Extractor for code elements
            metrics: Metrics calculator
            response_parser: Service for parsing AI responses

        Raises:
            ValueError: If ai_handler is not provided
        """
        self.logger = LoggerSetup.get_logger(__name__)
        if ai_handler is None:
            raise ValueError("ai_handler is required for DocumentationOrchestrator")
        self.ai_handler: 'AIInteractionHandler' = ai_handler
        self.docstring_schema: Dict[str, Any] = load_schema("docstring_schema")  # Load schema
        self.metrics: Metrics = metrics or Metrics()
        self.code_extractor: CodeExtractor = code_extractor or CodeExtractor(ExtractionContext())
        self.docstring_processor: DocstringProcessor = docstring_processor or DocstringProcessor(metrics=self.metrics)
        self.response_parser: ResponseParsingService = response_parser or ResponseParsingService()

    async def generate_documentation(self, context: DocumentationContext) -> Tuple[str, str]:
        """
        Generate documentation for the provided source code.

        Args:
            context: DocumentationContext containing the source code and metadata.

        Returns:
            Tuple containing the updated source code and generated documentation.

        Raises:
            DocumentationError: If documentation generation fails.
        """
        try:
            self.logger.info("Starting documentation generation process")

            # Step 1: Extract code elements
            try:
                extraction_result: Optional[ExtractionResult] = await self.code_extractor.extract_code(context.source_code)
                if not extraction_result:
                    self.logger.error("Failed to extract code elements.")
                    raise DocumentationError("Failed to extract code elements")
                self.logger.info("Code extraction completed successfully")
            except Exception as e:
                error_msg = f"Code extraction failed: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise DocumentationError(error_msg) from e

            # Step 2: Generate enriched prompt for AI model
            try:
                extraction_dict = extraction_result.to_dict()
                prompt: str = await self.ai_handler.create_dynamic_prompt(
                    extraction_dict
                )
                self.logger.info("Prompt generated successfully for AI model")
            except Exception as e:
                error_msg = f"Prompt generation failed: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise DocumentationError(error_msg) from e

            # Step 3: Interact with AI to generate docstrings
            try:
                ai_response: Union[str, Dict[str, Any]] = await self.ai_handler._interact_with_ai(prompt)
                self.logger.info("AI interaction completed successfully")
            except Exception as e:
                error_msg = f"AI interaction failed: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise DocumentationError(error_msg) from e

            # Step 4: Parse AI response and validate
            try:
                parsed_response = await self.response_parser.parse_response(
                    ai_response, expected_format="docstring"
                )
                if not parsed_response.validation_success:
                    self.logger.error("Validation failed for AI response.")
                    raise DocumentationError("Failed to validate AI response.")
                self.logger.info("AI response parsed and validated successfully")
            except Exception as e:
                error_msg = f"AI response parsing or validation failed: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise DocumentationError(error_msg) from e

            # Step 5: Integrate AI-generated docstrings
            try:
                updated_code, updated_documentation = await self.ai_handler._integrate_ai_response(
                    parsed_response.content, extraction_result
                )
                self.logger.info("Docstring integration completed successfully")
            except Exception as e:
                error_msg = f"AI response integration failed: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise DocumentationError(error_msg) from e

            self.logger.info("Documentation generation completed successfully")
            return updated_code, updated_documentation

        except DocumentationError as de:
            self.logger.error(f"DocumentationError encountered: {de}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error during documentation generation: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise DocumentationError(error_msg)

    async def generate_module_documentation(
        self, file_path: Path, output_dir: Path
    ) -> None:
        """
        Generate documentation for a single module.

        Args:
            file_path: Path to the Python file
            output_dir: Output directory for documentation

        Raises:
            DocumentationError: If module documentation generation fails
        """
        try:
            self.logger.info(f"Generating documentation for {file_path}")

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / file_path.with_suffix(".md").name

            # Read source code
            source_code = file_path.read_text(encoding="utf-8")

            # Create documentation context
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

            # Generate documentation
            updated_code, documentation = await self.generate_documentation(context)

            # Write output
            output_path.write_text(documentation, encoding="utf-8")
            file_path.write_text(updated_code, encoding="utf-8")
            self.logger.info(f"Documentation written to {output_path}")

        except Exception as e:
            error_msg = f"Module documentation generation failed for {file_path}: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise DocumentationError(error_msg)

    async def generate_batch_documentation(
        self,
        file_paths: List[Path],
        output_dir: Path
    ) -> Dict[Path, bool]:
        """
        Generate documentation for multiple files.

        Args:
            file_paths: List of Python file paths
            output_dir: Output directory for documentation

        Returns:
            Dict mapping file paths to success status
        """
        results = {}
        for file_path in file_paths:
            try:
                await self.generate_module_documentation(file_path, output_dir)
                results[file_path] = True
            except DocumentationError as e:
                self.logger.error(f"Failed to generate docs for {file_path}: {e}")
                results[file_path] = False
        return results

    def _generate_markdown(self, context: Dict[str, Any]) -> str:
        """Generate markdown documentation from context."""
        # Placeholder for markdown generation logic
        return "Generated markdown documentation"

    async def __aenter__(self) -> "DocumentationOrchestrator":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any]
    ) -> None:
        """Async context manager exit."""
        pass


def load_schema(schema_name: str) -> Dict[str, Any]:
    """Load the schema from a JSON file in the 'schemas' directory."""
    schema_path = Path(__file__).parent.parent / 'schemas' / f"{schema_name}.json"

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as file:
        return json.load(file)
