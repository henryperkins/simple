"""
Main Application Module

Orchestrates the documentation generation process using Azure OpenAI,
handling configuration, initialization, and cleanup of components.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional, List, Tuple
import argparse
from dotenv import load_dotenv
from core.logger import LoggerSetup
from core.cache import Cache
from core.monitoring import MetricsCollector
from core.config import AzureOpenAIConfig
from api.token_management import TokenManager
from ai_interaction import AIInteractionHandler
from exceptions import (
    ConfigurationError,
    ProcessingError,
    ValidationError
)

load_dotenv()
# Initialize logger and load configuration
logger = LoggerSetup.get_logger(__name__)
config = AzureOpenAIConfig.from_env()

class DocumentationGenerator:
    """
    Main application class for generating documentation using Azure OpenAI.

    Handles initialization of components, processing of source files,
    and cleanup of resources.
    """

    def __init__(self):
        """Initialize the documentation generator."""
        self.cache = None
        self.metrics = None
        self.token_manager = None
        self.ai_handler = None

    async def initialize(self) -> None:
        """
        Initialize all components needed for documentation generation.

        Raises:
            ConfigurationError: If component initialization fails
        """
        try:
            # Initialize cache if enabled
            if config.cache_enabled:
                self.cache = Cache(
                    host=config.redis_host,
                    port=config.redis_port,
                    db=config.redis_db,
                    password=config.redis_password,
                    enabled=config.cache_enabled
                )
                logger.info("Cache initialized successfully")

            # Initialize metrics collector
            self.metrics = MetricsCollector()
            logger.info("Metrics collector initialized")

            # Initialize token manager
            self.token_manager = TokenManager(
                model=config.model_name,
                deployment_name=config.deployment_name
            )
            logger.info("Token manager initialized")

            # Initialize AI interaction handler
            self.ai_handler = AIInteractionHandler(
                cache=self.cache,
                metrics_collector=self.metrics,
                token_manager=self.token_manager
            )
            logger.info("AI interaction handler initialized")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise ConfigurationError(f"Failed to initialize components: {str(e)}")

    async def cleanup(self) -> None:
        """Cleanup and close all components."""
        try:
            if self.ai_handler:
                await self.ai_handler.close()
            if self.cache:
                await self.cache.close()
            if self.metrics:
                await self.metrics.close()
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

    async def process_file(
        self,
        file_path: Path
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Process a single Python file for documentation generation.

        Args:
            file_path (Path): Path to the Python file

        Returns:
            Tuple[Optional[str], Optional[str]]: Updated code and documentation

        Raises:
            ProcessingError: If file processing fails
        """
        try:
            if not file_path.exists():
                raise ValidationError(f"File not found: {file_path}")

            # Read source code
            source_code = file_path.read_text(encoding='utf-8')
            if not source_code.strip():
                raise ValidationError(f"Empty file: {file_path}")

            # Generate cache key
            cache_key = f"doc:{file_path.stem}:{hash(source_code)}"

            # Process code
            updated_code, documentation = await self.ai_handler.process_code(
                source_code,
                cache_key
            )

            return updated_code, documentation

        except ValidationError as ve:
            logger.error(f"Validation error for file {file_path}: {str(ve)}")
            raise ve
        except ProcessingError as pe:
            logger.error(f"Processing error for file {file_path}: {str(pe)}")
            raise pe
        except Exception as e:
            logger.error(f"Unexpected error processing file {file_path}: {str(e)}")
            raise ProcessingError(f"File processing failed: {str(e)}")

    async def save_results(
        self,
        file_path: Path,
        updated_code: str,
        documentation: str
    ) -> None:
        """
        Save processing results to files.

        Args:
            file_path (Path): Original file path
            updated_code (str): Updated source code
            documentation (str): Generated documentation
        """
        try:
            # Save updated code
            file_path.write_text(updated_code, encoding='utf-8')

            # Save documentation to markdown file
            doc_path = file_path.with_suffix('.md')
            doc_path.write_text(documentation, encoding='utf-8')

            logger.info(f"Results saved for {file_path}")

        except Exception as e:
            logger.error(f"Failed to save results for {file_path}: {str(e)}")
            raise ProcessingError(f"Failed to save results: {str(e)}")

    async def process_files(self, file_paths: List[Path]) -> None:
        """
        Process multiple Python files for documentation generation.

        Args:
            file_paths (List[Path]): List of file paths to process
        """
        for file_path in file_paths:
            try:
                logger.info(f"Processing file: {file_path}")

                updated_code, documentation = await self.process_file(file_path)

                if updated_code and documentation:
                    await self.save_results(
                        file_path,
                        updated_code,
                        documentation
                    )
                    logger.info(f"Successfully processed {file_path}")
                else:
                    logger.warning(f"No results generated for {file_path}")

            except ValidationError as ve:
                logger.error(f"Validation error for file {file_path}: {str(ve)}")
            except ProcessingError as pe:
                logger.error(f"Processing error for file {file_path}: {str(pe)}")
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {str(e)}")

async def main(args: argparse.Namespace) -> int:
    """
    Main application entry point.

    Args:
        args (argparse.Namespace): Command line arguments

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    generator = DocumentationGenerator()

    try:
        # Initialize components
        await generator.initialize()

        # Process input files
        file_paths = [Path(f) for f in args.files]
        await generator.process_files(file_paths)

        logger.info("Documentation generation completed successfully")
        return 0

    except ConfigurationError as ce:
        logger.error(f"Configuration error: {str(ce)}")
        return 1
    except Exception as e:
        logger.error(f"Documentation generation failed: {str(e)}")
        return 1

    finally:
        await generator.cleanup()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate documentation for Python files using Azure OpenAI"
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='Python files to process'
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
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)