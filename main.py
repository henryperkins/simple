"""
Documentation Generator main module.

Handles initialization, processing, and cleanup of documentation generation.
"""

import argparse
import asyncio
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import ast
from dotenv import load_dotenv
from tqdm import tqdm

from ai_interaction import AIInteractionHandler
from api.token_management import TokenManager
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.docs import DocStringManager
from core.logger import LoggerSetup
from core.metrics import MetricsCollector
from core.monitoring import SystemMonitor
from core.response_parsing import ResponseParsingService
from core.types import DocumentationContext
from exceptions import ConfigurationError, DocumentationError
from repository_handler import RepositoryHandler

load_dotenv()
logger = LoggerSetup.get_logger(__name__)


class DocumentationGenerator:
    """Documentation Generator."""

    def __init__(
        self,
        config: AzureOpenAIConfig,
        cache: Optional[Cache],
        metrics: MetricsCollector,
        token_manager: TokenManager,
        ai_handler: AIInteractionHandler,
        response_parser: ResponseParsingService,  # Add response_parser here
        system_monitor: Optional[SystemMonitor] = None,
    ) -> None:
        """Initialize the DocumentationGenerator."""
        self.logger = LoggerSetup.get_logger(__name__)
        self.config = config
        self.cache = cache
        self.metrics = metrics
        self.token_manager = token_manager
        self.ai_handler = ai_handler
        self.response_parser = response_parser  # Initialize the response_parser
        self.system_monitor = system_monitor
        self.repo_handler: Optional[RepositoryHandler] = None
        self._initialized = False

    async def initialize(self, base_path: Optional[Path] = None) -> None:
        """Initialize components that depend on runtime arguments."""
        try:
            if base_path:
                self.repo_handler = RepositoryHandler(repo_path=base_path)
                await self.repo_handler.__aenter__()
                self.logger.info(f"Repository handler initialized with path: {base_path}")

            self._initialized = True
            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            await self.cleanup()
            raise ConfigurationError(f"Failed to initialize components: {str(e)}") from e

    async def cleanup(self) -> None:
        """Clean up all resources safely."""
        cleanup_errors = []

        components = [
            (self.system_monitor, "System Monitor", lambda x: x.stop() if hasattr(x, "stop") else None),
            (self.repo_handler, "Repository Handler", lambda x: x.__aexit__(None, None, None) if hasattr(x, "__aexit__") else None),
            (self.ai_handler, "AI Handler", lambda x: x.close() if hasattr(x, "close") else None),
            (self.token_manager, "Token Manager", None),
            (self.cache, "Cache", lambda x: x.close() if hasattr(x, "close") else None),
        ]

        for component, name, cleanup_func in components:
            if component is not None:
                try:
                    if cleanup_func:
                        result = cleanup_func(component)
                        if asyncio.iscoroutine(result):
                            await result
                    self.logger.info(f"{name} cleaned up successfully")
                except Exception as e:
                    error_msg = f"Error cleaning up {name}: {str(e)}"
                    self.logger.error(error_msg)
                    cleanup_errors.append(error_msg)

        self._initialized = False

        if cleanup_errors:
            self.logger.error("Some components failed to cleanup properly")
            for error in cleanup_errors:
                self.logger.error(f"- {error}")

    async def process_file(self, file_path: Path, output_base: Path) -> Optional[Tuple[str, str]]:
        """
        Process a single Python file to generate documentation.

        Args:
            file_path (Path): Path to the Python file to process.
            output_base (Path): Base directory for output documentation.

        Returns:
            Optional[Tuple[str, str]]: A tuple of (updated_code, documentation) if successful, otherwise None.
        """
        logger = LoggerSetup.get_logger(__name__)
        if not self._initialized:
            raise RuntimeError("DocumentationGenerator not initialized")

        start_time = datetime.now()

        try:
            # Normalize paths
            file_path = Path(file_path).resolve()
            output_base = Path(output_base).resolve()
            logger.debug(f"Processing file: {file_path}, output_base: {output_base}")

            # Validate the input file
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            if not file_path.suffix == ".py":
                raise ValueError(f"File is not a Python file: {file_path}")

            # Read the file source code
            try:
                source_code = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                source_code = file_path.read_text(encoding="latin-1")
                logger.warning(f"Used latin-1 encoding fallback for file: {file_path}")

            # Ensure source code is non-empty
            if not source_code.strip():
                raise ValueError(f"Source code is empty: {file_path}")

            # Validate Python syntax
            try:
                ast.parse(source_code)
            except SyntaxError as e:
                logger.error(f"Syntax error in {file_path}: {e}")
                return None

            logger.debug("File syntax is valid.")

            # Generate cache key and process code
            cache_key = f"doc:{file_path.stem}:{hash(source_code.encode())}"
            if not self.ai_handler:
                raise RuntimeError("AI handler not initialized")
            result = await self.ai_handler.process_code(source_code, cache_key)

            if not result:
                raise ValueError(f"AI processing failed for: {file_path}")

            updated_code, ai_docs = result

            # Derive module_name
            module_name = file_path.stem or "UnknownModule"
            if not module_name.strip():  # Ensure the module_name is valid
                raise ValueError(f"Invalid module name derived from file path: {file_path}")

            logger.debug(f"Derived module_name: {module_name}")

            # Create metadata
            metadata = {
                "file_path": str(file_path),
                "module_name": module_name,
                "creation_time": datetime.now().isoformat(),
            }
            logger.debug(f"Constructed metadata: {metadata}")

            # Validate metadata
            if not isinstance(metadata, dict):
                raise TypeError(f"Metadata is not a dictionary: {metadata}")

            # Create the DocumentationContext
            context = DocumentationContext(
                source_code=updated_code,
                module_path=file_path,
                include_source=True,
                metadata=metadata,
                ai_generated=ai_docs,
            )
            logger.debug(f"Documentation context created: {context.__dict__}")

            # Generate the documentation
            doc_manager = DocStringManager(
                context=context,
                ai_handler=self.ai_handler,
                response_parser=self.response_parser,
            )
            documentation = await doc_manager.generate_documentation()

            # Handle the output directory and file creation
            output_dir = output_base / "docs"
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Output directory created at: {output_dir}")

            # Calculate output path and write the file
            output_path = output_dir / file_path.relative_to(output_base).with_suffix(".md")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(documentation, encoding="utf-8")
            logger.info(f"Documentation written to: {output_path}")

            # Return result
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Processed {file_path} in {duration:.2f} seconds")
            return updated_code, documentation

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None

    async def process_repository(self, repo_path_or_url: str, output_file: str = "docs/Documentation.md") -> int:
        """
        Process an entire repository to generate a single markdown documentation file.

        Args:
            repo_path_or_url (str): Local path or Git URL to the repository.
            output_file (str): Path to the output markdown file.

        Returns:
            int: 0 if processing is successful, 1 otherwise.
        """
        logger = LoggerSetup.get_logger(__name__)
        if not self._initialized:
            raise RuntimeError("DocumentationGenerator not initialized")

        start_time = datetime.now()
        processed_files = 0
        failed_files = []
        combined_documentation = ""
        toc_entries = []
        is_url = urlparse(repo_path_or_url).scheme != ""

        try:
            # Handle repository setup
            if is_url:
                logger.info(f"Cloning repository from URL: {repo_path_or_url}")
                if not self.repo_handler:
                    self.repo_handler = RepositoryHandler(repo_path=Path(tempfile.mkdtemp()))
                repo_path = await self.repo_handler.clone_repository(repo_path_or_url)
            else:
                repo_path = Path(repo_path_or_url).resolve()
                if not repo_path.exists():
                    raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
                if not self.repo_handler:
                    self.repo_handler = RepositoryHandler(repo_path=repo_path)

            logger.info(f"Starting repository processing: {repo_path}")
            python_files = self.repo_handler.get_python_files()

            if not python_files:
                logger.warning("No Python files found")
                return 0

            # Process files and accumulate markdown documentation
            with tqdm(python_files, desc="Processing files") as progress:
                for file_path in progress:
                    try:
                        result = await self.process_file(file_path, repo_path)
                        if result:
                            updated_code, module_doc = result
                            processed_files += 1
                            
                            # Add module to TOC and combined file
                            module_name = Path(file_path).stem
                            toc_entries.append(f"- [{module_name}](#{module_name.lower().replace('_', '-')})")
                            combined_documentation += module_doc + "\n\n"
                        else:
                            failed_files.append((file_path, "Processing failed"))
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        failed_files.append((file_path, str(e)))

                    progress.set_postfix(processed=processed_files, failed=len(failed_files))

            # Create unified documentation file
            toc_section = "# Table of Contents\n\n" + "\n".join(toc_entries) + "\n\n"
            full_documentation = toc_section + combined_documentation

            output_dir = Path(output_file).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(full_documentation)
            
            logger.info(f"Final documentation written to: {output_file}")

            # Log summary
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info("\nProcessing Summary:")
            logger.info(f"Total files: {len(python_files)}")
            logger.info(f"Successfully processed: {processed_files}")
            logger.info(f"Failed: {len(failed_files)}")
            logger.info(f"Total processing time: {processing_time:.2f} seconds")

            if failed_files:
                logger.error("\nFailed files:")
                for file_path, error in failed_files:
                    logger.error(f"- {file_path}: {error}")

            return 0 if processed_files > 0 else 1

        finally:
            if is_url and self.repo_handler:
                await self.repo_handler.cleanup()
                logger.info("Repository cleanup completed")

async def main(args: argparse.Namespace) -> int:
    """
    Main application entry point.
    """
    try:
        # Initialize shared dependencies in main
        config = AzureOpenAIConfig.from_env()
        cache = Cache(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password,
            enabled=config.cache_enabled,
        )
        metrics = MetricsCollector()
        token_manager = TokenManager(
            model=config.model_name,
            deployment_id=config.deployment_id,
            config=config,
            metrics_collector=metrics,
        )
        response_parser = ResponseParsingService()

        # Instantiate AIInteractionHandler
        ai_handler = AIInteractionHandler(
            config=config,
            cache=cache,
            token_manager=token_manager,
            response_parser=response_parser,
            metrics=metrics,
        )
        system_monitor = SystemMonitor(check_interval=60, token_manager=token_manager)

        # Instantiate the DocumentationGenerator with dependencies
        generator = DocumentationGenerator(
            config=config,
            cache=cache,
            metrics=metrics,
            token_manager=token_manager,
            ai_handler=ai_handler,
            response_parser=response_parser,
            system_monitor=system_monitor,
        )

        # Initialize and process based on arguments
        await generator.initialize()
        if args.repository:
            return await generator.process_repository(args.repository, output_file=args.output)
        elif args.files:
            success = True
            for file_path in args.files:
                try:
                    result = await generator.process_file(Path(file_path), Path("."))
                    if not result:
                        success = False
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    success = False
            return 0 if success else 1
        else:
            logger.error("No input specified")
            return 1

    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1
    finally:
        # Ensure cleanup happens
        await generator.cleanup()

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate documentation for Python files or repositories."
    )
    parser.add_argument(
        "--repository",
        type=str,
        help="Local path or Git URL of the repository to process",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Python files to process",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/Documentation.md",
        help="Path to the unified documentation markdown file (default: docs/Documentation.md)",
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
