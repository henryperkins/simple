"""
Main Application Module

Orchestrates the documentation generation process using Azure OpenAI,
handling configuration, initialization, and cleanup of components.
"""
import os
import asyncio
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import argparse
import ast
from dotenv import load_dotenv
from core.logger import LoggerSetup
from core.cache import Cache
from core.monitoring import MetricsCollector
from core.config import AzureOpenAIConfig
from api.token_management import TokenManager
from ai_interaction import AIInteractionHandler
from repository_handler import RepositoryHandler
from exceptions import (
    ConfigurationError,
    ProcessingError,
    ValidationError
)

load_dotenv()

# Debug: Print the Azure OpenAI endpoint to verify it's loaded correctly
print("Azure OpenAI Endpoint:", os.getenv("AZURE_OPENAI_ENDPOINT"))

# Initialize logger and load configuration
logger = LoggerSetup.get_logger(__name__)


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
        self.config = AzureOpenAIConfig.from_env()

    async def initialize(self) -> None:
        """
        Initialize all components needed for documentation generation.

        Raises:
            ConfigurationError: If component initialization fails
        """
        try:
            # Initialize cache if enabled
            if self.config.cache_enabled:
                self.cache = Cache(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password,
                    enabled=self.config.cache_enabled
                )
                logger.info("Cache initialized successfully")

            # Initialize metrics collector
            self.metrics = MetricsCollector()
            logger.info("Metrics collector initialized")

            # Initialize token manager
            self.token_manager = TokenManager(
                model=self.config.model_name,
                deployment_name=self.config.deployment_name,
                config=self.config
            )
            logger.info("Token manager initialized")

            # Initialize AI interaction handler
            self.ai_handler = AIInteractionHandler(
                cache=self.cache,
                metrics_collector=self.metrics,
                token_manager=self.token_manager,
                config=self.config  # Pass the config here
            )
            logger.info("AI interaction handler initialized")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise ConfigurationError(
                f"Failed to initialize components: {str(e)}"
            )

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

    async def process_file(self, file_path: Path) -> Tuple[Optional[str], Optional[str]]:
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
                # For test files, create a basic template if empty
                if 'tests' in str(file_path) and str(file_path).endswith('.py'):
                    source_code = self._create_test_template(file_path)
                else:
                    raise ValidationError(f"Empty file: {file_path}")
                
            # Generate cache key
            cache_key = f"doc:{file_path.stem}:{hash(source_code)}"

            # Process code
            updated_code, documentation = await self.ai_handler.process_code(
                source_code,
                cache_key=cache_key
            )

            return updated_code, documentation

        except ValidationError as ve:
            logger.error(f"Validation error for file {file_path}: {str(ve)}")
            raise
        except ProcessingError as pe:
            logger.error(f"Processing error for file {file_path}: {str(pe)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing file {file_path}: {str(e)}")
            raise ProcessingError(f"File processing failed: {str(e)}")

    def _create_test_template(self, node: ast.AST, metadata: Dict[str, Any] = None) -> str:
        """
        Create a test template for the given node.
        
        Args:
            node: AST node to generate test template for
            metadata: Additional metadata about the node (default: None)
            
        Returns:
            str: Generated test template
        """
        if metadata is None:
            metadata = {}
            
        node_name = getattr(node, 'name', 'unknown')
        
        template = [
            f"def test_{node_name.lower()}():",
            "    # Setup",
            "    # TODO: Initialize test data and dependencies",
            "",
            "    # Exercise", 
            f"    # TODO: Call {node_name} with test inputs",
            "",
            "    # Verify",
            "    # TODO: Add assertions to verify expected behavior",
            "",
            "    # Cleanup",
            "    # TODO: Clean up any test resources"
        ]
        
        if isinstance(node, ast.ClassDef):
            template.extend([
                "",
                "    # Additional test cases:",
                f"    # TODO: Add edge cases for {node_name}",
                "    # TODO: Add error condition tests",
                "    # TODO: Add integration tests if needed"
            ])
            
        return "\n    ".join(template)

    async def save_results(self, file_path: Path, updated_code: str, documentation: str) -> None:
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
                    await self.save_results(file_path, updated_code, documentation)
                    logger.info(f"Successfully processed {file_path}")
                else:
                    logger.warning(f"No results generated for {file_path}")

            except ValidationError as ve:
                logger.error(f"Validation error for file {file_path}: {str(ve)}")
            except ProcessingError as pe:
                logger.error(f"Processing error for file {file_path}: {str(pe)}")
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {str(e)}")


async def process_repository(args: argparse.Namespace) -> int:
    """
    Process repository for documentation generation.

    Args:
        args (argparse.Namespace): Command line arguments

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    repo_handler = RepositoryHandler()
    generator = DocumentationGenerator()

    try:
        # Initialize components
        await generator.initialize()

        # Clone repository
        repo_path = repo_handler.clone_repository(args.repository)
        logger.info(f"Repository cloned to: {repo_path}")

        # Get Python files
        python_files = repo_handler.get_python_files()
        if not python_files:
            logger.warning("No Python files found in repository")
            return 0

        # Process files
        await generator.process_files([Path(f) for f in python_files])
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
        repo_handler.cleanup()


async def main(args: argparse.Namespace) -> int:
    """
    Main application entry point for processing local files.

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
        '--repository',
        type=str,
        help='URL of the git repository to process'
    )
    parser.add_argument(
        '--files',
        nargs='*',
        help='Python files to process (alternative to repository)'
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_arguments()
        if args.repository:
            exit_code = asyncio.run(process_repository(args))
        elif args.files:
            exit_code = asyncio.run(main(args))
        else:
            logger.error("Either --repository or --files must be specified")
            exit_code = 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
