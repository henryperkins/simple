"""
Documentation Generator main module.

Handles initialization, processing, and cleanup of documentation generation.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from ai_interaction import AIInteractionHandler
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.logger import LoggerSetup
from core.metrics import Metrics, MetricsCollector
from core.response_parsing import ResponseParsingService
from core.docstring_processor import DocstringProcessor
from core.extraction.code_extractor import CodeExtractor
from core.types import ExtractionContext
from docs import DocumentationOrchestrator
from monitoring import SystemMonitor
from exceptions import ConfigurationError, DocumentationError

load_dotenv()
logger = LoggerSetup.get_logger(__name__)

class DocumentationGenerator:
    """Documentation Generator orchestrating all components."""

    def __init__(self, config: Optional[AzureOpenAIConfig] = None) -> None:
        """Initialize the documentation generator with all necessary components."""
        self.logger = LoggerSetup.get_logger(__name__)
        try:
            # Load configuration
            self.config = config or AzureOpenAIConfig.from_env()

            # Initialize core components
            self.metrics = Metrics()
            self.metrics_collector = MetricsCollector()
            self.cache = self._setup_cache()
            self.token_manager = TokenManager(
                model=self.config.model_name,
                deployment_id=self.config.deployment_id,
                config=self.config,
                metrics_collector=self.metrics_collector,
            )

            # Load schema
            self.docstring_schema = self.load_schema("docstring_schema")

            # Initialize API and response handling
            self.response_parser = ResponseParsingService()
            self.api_client = AIInteractionHandler(
                config=self.config,
                cache=self.cache,
                token_manager=self.token_manager,
                response_parser=self.response_parser,
                metrics=self.metrics,
                docstring_schema=self.docstring_schema  # Pass the schema here
            )

            # Initialize processors
            self.docstring_processor = DocstringProcessor(metrics=self.metrics)

            # Initialize extractors
            self.code_extractor = CodeExtractor(ExtractionContext())

            # Initialize orchestrators
            self.doc_orchestrator = DocumentationOrchestrator(
                ai_handler=self.api_client,
                docstring_processor=self.docstring_processor,
                code_extractor=self.code_extractor,
                metrics=self.metrics,
                response_parser=self.response_parser,
            )

            # Initialize system monitoring
            self.system_monitor = SystemMonitor(
                check_interval=60,
                token_manager=self.token_manager,
                metrics_collector=self.metrics_collector
            )

            self._initialized = False

        except Exception as e:
            error_msg = f"Failed to initialize DocumentationGenerator: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise ConfigurationError(error_msg) from e

    def load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load the schema from a JSON file."""
        with open(f"{schema_name}.json", "r") as file:
            return json.load(file)

    def _setup_cache(self) -> Optional[Cache]:
        """Setup cache if enabled in configuration."""
        try:
            if self.config.cache_enabled:
                return Cache(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password,
                    enabled=True,
                )
            return None
        except Exception as e:
            error_msg = f"Error setting up cache: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise ConfigurationError(error_msg) from e

    async def initialize(self) -> None:
        """Initialize components that depend on runtime arguments."""
        try:
            await self.system_monitor.start()
            self._initialized = True
            self.logger.info("All components initialized successfully")
        except Exception as e:
            error_msg = f"Initialization failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            await self.cleanup()
            raise ConfigurationError(error_msg) from e

    async def process_file(self, file_path: Path, output_path: Path) -> bool:
        """Process a single Python file to generate documentation."""
        try:
            self.logger.info(f"Processing file: {file_path}")
            source_code = file_path.read_text(encoding="utf-8")

            # Process code and generate documentation
            try:
                updated_code, documentation = await self.api_client.process_code(source_code)
            except DocumentationError as e:
                self.logger.error(f"Documentation generation error for {file_path}: {e}")
                return False

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save processed files
            output_path.write_text(documentation, encoding="utf-8")
            file_path.write_text(updated_code, encoding="utf-8")

            self.logger.info(f"Successfully processed {file_path}")
            return True

        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False

    async def process_repository(
        self, repo_path: str, output_dir: Path = Path("docs")
    ) -> bool:
        """
        Process an entire repository and generate documentation.

        Args:
            repo_path: Local path to repository
            output_dir: Output directory for documentation

        Returns:
            bool: True if processing was successful
        """
        if not self._initialized:
            raise RuntimeError("DocumentationGenerator not initialized")

        try:
            repo_path = Path(repo_path)

            if not repo_path.exists():
                raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Process all Python files
            python_files = list(repo_path.rglob("*.py"))
            if not python_files:
                self.logger.warning("No Python files found in repository")
                return False

            success = True
            for file_path in python_files:
                relative_path = file_path.relative_to(repo_path)
                output_path = output_dir / relative_path.with_suffix(".md")

                if not await self.process_file(file_path, output_path):
                    success = False

            return success

        except Exception as e:
            error_msg = f"Repository processing failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            return False

    async def display_metrics(self) -> None:
        """Display collected metrics in the terminal."""
        try:
            # Retrieve collected metrics from MetricsCollector
            collected_metrics = self.metrics_collector.get_metrics()

            # Retrieve system metrics from SystemMonitor
            system_metrics = self.system_monitor.get_metrics()

            # Format and print metrics
            print("=== Documentation Generation Metrics ===")
            for metric in collected_metrics['operations']:
                print(f"Operation: {metric['operation_type']}")
                print(f"Success: {metric['success']}")
                print(f"Duration: {metric['duration']} seconds")
                print(f"Usage: {metric['usage']}")
                print(f"Validation Success: {metric['validation_success']}")
                print(f"Timestamp: {metric['timestamp']}")
                print("-" * 40)

            print("=== System Performance Metrics ===")
            print(system_metrics)
            print("-" * 40)

        except Exception as e:
            self.logger.error(f"Error displaying metrics: {e}")
            print(f"Error displaying metrics: {e}")

    async def cleanup(self) -> None:
        """Clean up all resources."""
        try:
            if self.api_client:
                await self.api_client.close()
            if self.cache:
                await self.cache.close()
            if self.system_monitor:
                await self.system_monitor.stop()
            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)

async def main(args: argparse.Namespace) -> int:
    """Main application entry point."""
    doc_generator = None
    try:
        # Initialize documentation generator
        config = AzureOpenAIConfig.from_env()
        doc_generator = DocumentationGenerator(config)
        await doc_generator.initialize()

        # Process based on arguments
        success = False
        if args.repository:
            success = await doc_generator.process_repository(args.repository, Path(args.output))
        elif args.files:
            success = True
            for file_path in args.files:
                output_path = Path(args.output) / Path(file_path).with_suffix(".md")
                if not await doc_generator.process_file(Path(file_path), output_path):
                    success = False

        # Display metrics after processing
        await doc_generator.display_metrics()

        return 0 if success else 1

    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1

    finally:
        if doc_generator:
            await doc_generator.cleanup()

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate documentation for Python files or repositories."
    )
    parser.add_argument(
        "--repository",
        type=str,
        help="Local path of the repository to process",
    )
    parser.add_argument("--files", nargs="+", help="Python files to process")
    parser.add_argument(
        "--output",
        type=str,
        default="docs",
        help="Output directory for documentation (default: docs)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        exit_code = asyncio.run(main(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
