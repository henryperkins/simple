"""
Documentation Generator main module.

Handles initialization, processing, and cleanup of documentation generation.
"""

import argparse
import asyncio
import sys
import re
from pathlib import Path
from typing import Optional, Dict, Any
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
from core.docs import DocumentationOrchestrator
from core.monitoring import SystemMonitor
from exceptions import ConfigurationError, DocumentationError
from core.schema_loader import load_schema
from api.token_management import TokenManager
from api.api_client import APIClient
from repository_handler import RepositoryHandler
from core.utils import GitUtils

load_dotenv()
logger = LoggerSetup.get_logger(__name__)

class DocumentationGenerator:
    """Documentation Generator orchestrating all components."""

    def __init__(self, config: Optional[AzureOpenAIConfig] = None) -> None:
        """Initialize DocumentationGenerator with all necessary components."""
        self.config = config or AzureOpenAIConfig.from_env()
        self.logger = LoggerSetup.get_logger(__name__)
        
        # Initialize APIClient
        self.api_client = APIClient(
            config=self.config,
            response_parser=ResponseParsingService(),
            token_manager=TokenManager()
        )
        
        # Initialize AIInteractionHandler
        self.ai_handler = AIInteractionHandler(
            config=self.config,
            cache=Cache(),
            token_manager=TokenManager(),
            response_parser=ResponseParsingService(),
            metrics=Metrics(),
            docstring_schema=load_schema("docstring_schema")
        )
        
        # Initialize DocumentationOrchestrator
        self.doc_orchestrator = DocumentationOrchestrator(
            ai_handler=self.ai_handler,
            docstring_processor=DocstringProcessor(metrics=Metrics()),
            code_extractor=CodeExtractor(ExtractionContext()),
            metrics=Metrics(),
            response_parser=ResponseParsingService()
        )
        
        # Initialize SystemMonitor
        self.system_monitor = SystemMonitor()
        self.metrics_collector = MetricsCollector()

        self.logger.info("All components initialized successfully")

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
        """Process a single file with validation."""
        try:
            self.logger.info(f"Processing file: {file_path}")
            
            if not file_path.exists():
                self.logger.error(f"File not found: {file_path}")
                return False
                
            source_code = file_path.read_text(encoding="utf-8")
            if not source_code.strip():
                self.logger.error(f"Empty file: {file_path}")
                return False

            # Process with error handling
            try:
                result = await self.ai_handler.process_code(source_code)
                if not result:
                    self.logger.error(f"No documentation generated for {file_path}")
                    return False
                    
                updated_code = result.get('code')
                documentation = result.get('documentation')
                
                if not updated_code or not documentation:
                    self.logger.error(f"Invalid result for {file_path}")
                    return False
                    
                # Save results
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(documentation, encoding="utf-8")
                file_path.write_text(updated_code, encoding="utf-8")
                
                self.logger.info(f"Successfully processed {file_path}")
                return True
                
            except Exception as e:
                self.logger.error(f"Processing error for {file_path}: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return False

    async def process_repository(
        self, repo_path: str, output_dir: Path = Path("docs")
    ) -> bool:
        try:
            if self._is_valid_url(repo_path):
                self.logger.info(f"Cloning repository from URL: {repo_path}")
                async with RepositoryHandler(repo_path=Path.cwd()) as repo_handler:
                    cloned_repo_path = await repo_handler.clone_repository(repo_path)
                    self.logger.info(f"Repository cloned to: {cloned_repo_path}")
                    # Proceed with processing the cloned repository
                    success = await self._process_local_repository(cloned_repo_path, output_dir)
                    return success
            else:
                # Assume it's a local path
                self.logger.info(f"Processing local repository at: {repo_path}")
                local_repo_path = Path(repo_path)
                if not local_repo_path.exists():
                    raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
                success = await self._process_local_repository(local_repo_path, output_dir)
                return success
        except Exception as e:
            self.logger.error(f"Repository processing failed: {e}")
            return False

    def _is_valid_url(self, url: str) -> bool:
        """Simple regex-based URL validation."""
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https:// or ftp://
            r'(?:\S+(?::\S*)?@)?'  # user and password
            r'(?:'
            r'(?:(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])\.){3}'
            r'(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-5])'  # IP address
            r'|'
            r'(?:(?:[a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,})'  # Domain name
            r')'
            r'(?::\d{2,5})?'  # Port
            r'(?:/\S*)?$', re.IGNORECASE)
        return re.match(regex, url) is not None

    async def _process_local_repository(self, repo_path: Path, output_dir: Path) -> bool:
        """Internal method to process a local repository."""
        try:
            # Get Python files using the corrected GitUtils method
            python_files = GitUtils.get_python_files(repo_path)
            
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each Python file
            for file_path in python_files:
                output_file = output_dir / (file_path.stem + ".md")
                success = await self.process_file(file_path, output_file)
                if not success:
                    self.logger.error(f"Failed to process file: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error processing local repository: {e}")
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
        """Cleanup resources after documentation generation."""
        try:
            # Properly close APIClient
            await self.api_client.close()
            
            # Properly close AIInteractionHandler
            await self.ai_handler.close()
            
            # Properly close MetricsCollector
            await self.metrics_collector.close()
    
            # Stop system monitor
            await self.system_monitor.stop()
            
            self.logger.info("Cleanup completed successfully")
        except AttributeError as ae:
            self.logger.error(f"Cleanup error: {ae}")
        except Exception as e:
            self.logger.error(f"Unexpected error during cleanup: {e}")

async def main(args: argparse.Namespace) -> int:
    """Main application entry point."""
    doc_generator = DocumentationGenerator()
    try:
        await doc_generator.initialize()
        
        if args.repository:
            success = await doc_generator.process_repository(args.repository, Path(args.output))
            if success:
                print("Repository documentation generated successfully.")
            else:
                print("Failed to generate repository documentation.")
        
        if args.files:
            for file in args.files:
                file_path = Path(file)
                output_path = Path(args.output) / file_path.stem
                success = await doc_generator.process_file(file_path, output_path)
                if success:
                    print(f"Documentation for {file} generated successfully.")
                else:
                    print(f"Failed to generate documentation for {file}.")
        
        await doc_generator.display_metrics()
    
    except DocumentationError as de:
        logger.error(f"Documentation generation failed: {de}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    finally:
        await doc_generator.cleanup()
    
    return 0

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
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Documentation generation interrupted by user.")
    except Exception as e:
        logger.error(f"Failed to run documentation generator: {e}")
