"""
Documentation Generator main module.

Handles initialization, processing, and cleanup of documentation generation.
"""

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

from ai_interaction import AIInteractionHandler
from api_client import APIClient
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.logger import LoggerSetup
from core.metrics import MetricsCollector
from core.monitoring import SystemMonitor
from core.response_parsing import ResponseParsingService
from docstring_processor import DocstringProcessor
from docs import DocumentationOrchestrator
from repository_handler import RepositoryHandler
from token_management import TokenManager
from exceptions import ConfigurationError
from utils import FileUtils

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
            self.metrics = MetricsCollector()
            self.cache = self._setup_cache()
            self.token_manager = TokenManager(
                model=self.config.model_name,
                deployment_id=self.config.deployment_id,
                config=self.config,
                metrics_collector=self.metrics
            )
            
            # Initialize API and response handling
            self.response_parser = ResponseParsingService()
            self.api_client = APIClient(
                config=self.config,
                response_parser=self.response_parser
            )
            
            # Initialize processors
            self.docstring_processor = DocstringProcessor(metrics=self.metrics)
            
            # Initialize handlers
            self.ai_handler = AIInteractionHandler(
                config=self.config,
                cache=self.cache,
                token_manager=self.token_manager,
                response_parser=self.response_parser,
                metrics=self.metrics
            )

            # Initialize orchestrators
            self.doc_orchestrator = DocumentationOrchestrator(
                ai_handler=self.ai_handler,
                docstring_processor=self.docstring_processor
            )

            # Initialize monitoring
            self.system_monitor = SystemMonitor(
                check_interval=60,
                token_manager=self.token_manager
            )

            self._initialized = False
            self.repo_handler = None

        except Exception as e:
            self.logger.error(f"Failed to initialize DocumentationGenerator: {e}")
            raise ConfigurationError(f"Initialization failed: {e}") from e

    def _setup_cache(self) -> Optional[Cache]:
        """Setup cache if enabled in configuration."""
        if self.config.cache_enabled:
            return Cache(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                enabled=True
            )
        return None

    async def initialize(self, base_path: Optional[Path] = None) -> None:
        """Initialize components that depend on runtime arguments."""
        try:
            if base_path:
                self.repo_handler = RepositoryHandler(repo_path=base_path)
                await self.repo_handler.__aenter__()
                self.logger.info(f"Repository handler initialized with path: {base_path}")

            await self.system_monitor.start()
            self._initialized = True
            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            await self.cleanup()
            raise ConfigurationError(f"Failed to initialize: {e}") from e

    async def process_file(self, file_path: Path, output_path: Path) -> bool:
        """
        Process a single Python file and generate documentation.

        Args:
            file_path: Path to the Python file
            output_path: Path where documentation should be written

        Returns:
            bool: True if processing was successful
        """
        if not self._initialized:
            raise RuntimeError("DocumentationGenerator not initialized")

        try:
            self.logger.info(f"Processing file: {file_path}")
            
            # Validate input file
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            if not file_path.suffix == '.py':
                raise ValueError(f"Not a Python file: {file_path}")

            # Create output directory
            FileUtils.ensure_directory(output_path.parent)

            # Generate documentation
            source_code = FileUtils.read_file_safe(file_path)
            updated_code, documentation = await self.doc_orchestrator.generate_documentation(
                source_code=source_code,
                file_path=file_path,
                module_name=file_path.stem
            )

            # Write output files
            if updated_code:
                file_path.write_text(updated_code, encoding='utf-8')
            if documentation:
                output_path.write_text(documentation, encoding='utf-8')

            self.logger.info(f"Documentation generated: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return False

    async def process_repository(
        self,
        repo_path_or_url: str,
        output_dir: Path = Path("docs")
    ) -> bool:
        """
        Process an entire repository and generate documentation.

        Args:
            repo_path_or_url: Local path or Git URL to repository
            output_dir: Output directory for documentation

        Returns:
            bool: True if processing was successful
        """
        if not self._initialized:
            raise RuntimeError("DocumentationGenerator not initialized")

        is_url = bool(urlparse(repo_path_or_url).scheme)
        temp_dir = None

        try:
            # Handle repository setup
            if is_url:
                temp_dir = Path(tempfile.mkdtemp())
                repo_path = await self.repo_handler.clone_repository(
                    repo_path_or_url,
                    temp_dir
                )
            else:
                repo_path = Path(repo_path_or_url)

            if not repo_path.exists():
                raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

            # Create output directory
            FileUtils.ensure_directory(output_dir)

            # Process all Python files
            python_files = FileUtils.filter_files(repo_path, pattern='*.py')
            if not python_files:
                self.logger.warning("No Python files found in repository")
                return False

            success = True
            for file_path in python_files:
                relative_path = file_path.relative_to(repo_path)
                output_path = output_dir / relative_path.with_suffix('.md')
                
                if not await self.process_file(file_path, output_path):
                    success = False

            return success

        except Exception as e:
            self.logger.error(f"Repository processing failed: {e}")
            return False

        finally:
            # Cleanup temporary directory if used
            if temp_dir and temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)

    async def display_metrics(self) -> None:
        """Display collected metrics in the terminal."""
        try:
            # Retrieve collected metrics from MetricsCollector
            collected_metrics = self.metrics.get_metrics()

            # Retrieve system metrics from SystemMonitor
            system_metrics = self.system_monitor.get_metrics()

            # Format and print metrics
            print("=== Documentation Generation Metrics ===")
            for metric in collected_metrics:
                print(f"Operation: {metric['operation_type']}")
                print(f"Success: {metric['success']}")
                print(f"Duration: {metric['duration']} seconds")
                print(f"Usage: {metric['usage']}")
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
        components = [
            (self.system_monitor, "System Monitor"),
            (self.repo_handler, "Repository Handler"),
            (self.ai_handler, "AI Handler"),
            (self.cache, "Cache"),
        ]

        for component, name in components:
            if component:
                try:
                    if hasattr(component, 'close'):
                        await component.close()
                    elif hasattr(component, '__aexit__'):
                        await component.__aexit__(None, None, None)
                    self.logger.debug(f"{name} cleaned up successfully")
                except Exception as e:
                    self.logger.error(f"Error cleaning up {name}: {e}")

        self._initialized = False

async def main(args: argparse.Namespace) -> int:
    """Main application entry point."""
    doc_generator = None
    try:
        # Initialize documentation generator
        config = AzureOpenAIConfig.from_env()
        doc_generator = DocumentationGenerator(config)
        await doc_generator.initialize()

        # Process based on arguments
        if args.repository:
            success = await doc_generator.process_repository(
                args.repository,
                Path(args.output)
            )
        elif args.files:
            success = True
            for file_path in args.files:
                output_path = Path(args.output) / Path(file_path).with_suffix('.md').name
                if not await doc_generator.process_file(Path(file_path), output_path):
                    success = False

        # Display metrics after processing
        await doc_generator.display_metrics()

        return 0 if success else 1

    except Exception as e:
        logger.error(f"Application error: {e}")
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
        help="Local path or Git URL of the repository to process"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Python files to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs",
        help="Output directory for documentation (default: docs)"
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
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)