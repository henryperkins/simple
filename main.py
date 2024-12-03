"""
Documentation Generator main module.
Handles initialization, processing, and cleanup of documentation generation.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import argparse
from dotenv import load_dotenv
from tqdm import tqdm
from urllib.parse import urlparse
import tempfile
import ast
from core.logger import LoggerSetup
from core.cache import Cache
from core.monitoring import SystemMonitor
from core.config import AzureOpenAIConfig
from api.token_management import TokenManager
from ai_interaction import AIInteractionHandler
from core.types import DocumentationContext
from core.docs import DocStringManager
from repository_handler import RepositoryHandler
from exceptions import ConfigurationError, DocumentationError
from core.response_parsing import ResponseParsingService
from core.metrics import MetricsCollector

load_dotenv()
logger = LoggerSetup.get_logger(__name__)

class DocumentationGenerator:
    """
    Documentation Generator Class.
    Handles initialization, processing, and cleanup of components for documentation generation.
    """

    def __init__(self) -> None:
        """Initialize the documentation generator with empty component references."""
        self.logger = LoggerSetup.get_logger(__name__)
        self.config: Optional[AzureOpenAIConfig] = None
        self.cache: Optional[Cache] = None
        self.metrics: Optional[MetricsCollector] = None
        self.token_manager: Optional[TokenManager] = None
        self.ai_handler: Optional[AIInteractionHandler] = None
        self.repo_handler: Optional[RepositoryHandler] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.response_parser: Optional[ResponseParsingService] = None
        self._initialized = False

    async def initialize(self, base_path: Optional[Path] = None) -> None:
        """
        Initialize all components in correct order with proper dependency injection.

        Args:
            base_path: Optional base path for repository operations

        Raises:
            ConfigurationError: If initialization fails
        """
        try:
            # 1. First load configuration
            self.config = AzureOpenAIConfig.from_env()
            if not self.config.validate():
                raise ConfigurationError("Invalid configuration")

            # 2. Initialize metrics collector
            self.metrics = MetricsCollector()
            self.logger.info("Metrics collector initialized")

            # 3. Initialize response parser
            self.response_parser = ResponseParsingService()
            self.logger.info("Response parser initialized")

            # 4. Initialize cache if enabled
            if self.config.cache_enabled:
                self.cache = Cache(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password,
                    enabled=True
                )
                await self.cache._initialize_connection()
                self.logger.info("Cache initialized")

            # 5. Initialize token manager
            self.token_manager = TokenManager(
                model=self.config.model_name,
                deployment_id=self.config.deployment_id,
                config=self.config,
                metrics_collector=self.metrics
            )
            self.logger.info("Token manager initialized")

            # 6. Initialize AI handler with response parser
            self.ai_handler = AIInteractionHandler(
                config=self.config,
                cache=self.cache,
                token_manager=self.token_manager,
                response_parser=self.response_parser
            )
            self.logger.info("AI handler initialized")

            # 7. Initialize repository handler if path provided
            if base_path:
                self.repo_handler = RepositoryHandler(repo_path=base_path)
                await self.repo_handler.__aenter__()
                self.logger.info(f"Repository handler initialized with path: {base_path}")

            # 8. Initialize system monitor
            self.system_monitor = SystemMonitor(
                check_interval=60,
                token_manager=self.token_manager
            )
            await self.system_monitor.start()
            self.logger.info("System monitor started")

            self._initialized = True
            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            await self.cleanup()
            raise ConfigurationError(f"Failed to initialize components: {str(e)}") from e

    async def cleanup(self) -> None:
        """Clean up all resources safely."""
        cleanup_errors = []
        
        # List of (component, name, cleanup_method)
        components = [
            (self.system_monitor, "System Monitor", 
             lambda x: x.stop() if hasattr(x, 'stop') else None),
            
            (self.repo_handler, "Repository Handler", 
             lambda x: x.__aexit__(None, None, None) if hasattr(x, '__aexit__') else None),
            
            (self.ai_handler, "AI Handler", 
             lambda x: x.close() if hasattr(x, 'close') else None),
            
            (self.token_manager, "Token Manager", 
             None),
            
            (self.cache, "Cache", 
             lambda x: x.close() if hasattr(x, 'close') else None),
            
            (self.metrics, "Metrics Collector", 
             lambda x: x.close() if hasattr(x, 'close') else None),
            
            (self.response_parser, "Response Parser",
             None)
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
        Process a single file with improved error handling and response parsing.

        Args:
            file_path (Path): Path to the Python file to process
            output_base (Path): Base directory for output documentation

        Returns:
            Optional[Tuple[str, str]]: Tuple of (updated_code, documentation) if successful, None otherwise

        Raises:
            RuntimeError: If DocumentationGenerator is not initialized
            FileNotFoundError: If the input file is not found
            ValueError: If the input file is not a Python file
            DocumentationError: If documentation generation fails
        """
        if not self._initialized:
            raise RuntimeError("DocumentationGenerator not initialized")

        start_time = datetime.now()

        try:
            # Create normalized paths
            file_path = Path(file_path).resolve()
            output_base = Path(output_base).resolve()

            # Validate input file
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if not file_path.suffix == '.py':
                raise ValueError(f"Not a Python file: {file_path}")

            # Read and validate source code
            try:
                try:
                    source_code = file_path.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    source_code = file_path.read_text(encoding='latin-1')
                    self.logger.warning(f"Used latin-1 fallback encoding for {file_path}")

                if not source_code.strip():
                    raise ValueError("Empty source code")

                # Validate syntax before processing
                try:
                    ast.parse(source_code)
                except SyntaxError as e:
                    self.logger.error(f"Syntax error in {file_path}: {str(e)}")
                    if self.metrics:
                        await self.metrics.track_operation(
                            operation_type="documentation_generation",
                            success=False,
                            duration=(datetime.now() - start_time).total_seconds(),
                            error=f"Syntax error: {str(e)}",
                            metadata={"file_path": str(file_path)}
                        )
                    return None

                # Process with AI handler
                if not self.ai_handler:
                    raise RuntimeError("AI handler not initialized")

                # Generate cache key based on file content
                cache_key = f"doc:{file_path.stem}:{hash(source_code.encode())}"
                
                # Process code with AI handler
                result = await self.ai_handler.process_code(
                    source_code=source_code,
                    cache_key=cache_key
                )

                if not result:
                    raise DocumentationError("AI processing failed")

                updated_code, ai_docs = result

                # Create documentation context
                context = DocumentationContext(
                    source_code=updated_code,
                    module_path=file_path,
                    include_source=True,
                    metadata={
                        "file_path": str(file_path),
                        "module_name": file_path.stem,
                        "creation_time": datetime.now().isoformat()
                    },
                    ai_generated=ai_docs
                )

                # Generate documentation
                doc_manager = DocStringManager(
                    context=context,
                    ai_handler=self.ai_handler,
                    response_parser=self.response_parser
                )

                documentation = await doc_manager.generate_documentation()

                # Handle output directory and file creation
                try:
                    # Create base docs directory
                    output_dir = output_base / 'docs'
                    output_dir.mkdir(parents=True, exist_ok=True)
                    self.logger.debug(f"Created/verified output directory: {output_dir}")

                    # Calculate relative path to maintain directory structure
                    try:
                        relative_path = file_path.relative_to(output_base)
                    except ValueError:
                        # If file_path is not relative to output_base, use just the filename
                        relative_path = file_path.name

                    # Create output path
                    output_path = output_dir / relative_path.with_suffix('.md')
                    self.logger.debug(f"Calculated output path: {output_path}")

                    # Ensure parent directories exist
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Write documentation file
                    try:
                        output_path.write_text(documentation, encoding='utf-8')
                        self.logger.info(f"Successfully wrote documentation to: {output_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to write documentation file: {e}")
                        raise DocumentationError(f"Failed to write documentation: {e}")

                    # Track metrics if enabled
                    if self.metrics:
                        await self.metrics.track_operation(
                            operation_type="documentation_generation",
                            success=True,
                            duration=(datetime.now() - start_time).total_seconds(),
                            metadata={
                                "file_path": str(file_path),
                                "output_path": str(output_path),
                                "file_size": len(documentation)
                            }
                        )

                    return updated_code, documentation

                except Exception as e:
                    self.logger.error(f"Error handling output files: {e}")
                    if self.metrics:
                        await self.metrics.track_operation(
                            operation_type="documentation_generation",
                            success=False,
                            duration=(datetime.now() - start_time).total_seconds(),
                            error=str(e),
                            metadata={"file_path": str(file_path)}
                        )
                    raise DocumentationError(f"Failed to handle output files: {e}")

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {str(e)}")
                if self.metrics:
                    await self.metrics.track_operation(
                        operation_type="documentation_generation",
                        success=False,
                        duration=(datetime.now() - start_time).total_seconds(),
                        error=str(e),
                        metadata={"file_path": str(file_path)}
                    )
                return None

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            if self.metrics:
                await self.metrics.track_operation(
                    operation_type="documentation_generation",
                    success=False,
                    duration=(datetime.now() - start_time).total_seconds(),
                    error=str(e),
                    metadata={"file_path": str(file_path)}
                )
            return None
        
    async def process_repository(self, repo_path_or_url: str) -> int:
        """Process an entire repository."""
        try:
            if not self._initialized:
                raise RuntimeError("DocumentationGenerator not initialized")

            start_time = datetime.now()
            processed_files = 0
            failed_files = []
            is_url = urlparse(repo_path_or_url).scheme != ''

            try:
                # Handle repository setup
                if is_url:
                    self.logger.info(f"Cloning repository from URL: {repo_path_or_url}")
                    if not self.repo_handler:
                        self.repo_handler = RepositoryHandler(repo_path=Path(tempfile.mkdtemp()))
                    repo_path = await self.repo_handler.clone_repository(repo_path_or_url)
                else:
                    repo_path = Path(repo_path_or_url).resolve()
                    if not repo_path.exists():
                        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
                    if not self.repo_handler:
                        self.repo_handler = RepositoryHandler(repo_path=repo_path)

                self.logger.info(f"Starting repository processing: {repo_path}")
                python_files = self.repo_handler.get_python_files()
                
                if not python_files:
                    self.logger.warning("No Python files found")
                    return 0

                # Process files
                with tqdm(python_files, desc="Processing files") as progress:
                    for file_path in progress:
                        try:
                            relative_path = file_path.relative_to(repo_path)
                            if self.ai_handler and self.ai_handler.context:
                                # Use pathlib's parts to create module name
                                module_name = '.'.join(relative_path.with_suffix('').parts)
                                self.ai_handler.context.module_name = module_name
                            result = await self.process_file(file_path, repo_path)

                            if result:
                                processed_files += 1
                            else:
                                failed_files.append((file_path, "Processing failed"))
                        except Exception as e:
                            self.logger.error(f"Error processing {file_path}: {e}")
                            failed_files.append((file_path, str(e)))
                        
                        progress.set_postfix(
                            processed=processed_files,
                            failed=len(failed_files)
                        )

                # Log summary
                processing_time = (datetime.now() - start_time).total_seconds()
                self.logger.info("\nProcessing Summary:")
                self.logger.info(f"Total files: {len(python_files)}")
                self.logger.info(f"Successfully processed: {processed_files}")
                self.logger.info(f"Failed: {len(failed_files)}")
                self.logger.info(f"Total processing time: {processing_time:.2f} seconds")

                if failed_files:
                    self.logger.error("\nFailed files:")
                    for file_path, error in failed_files:
                        self.logger.error(f"- {file_path}: {error}")

                return 0 if processed_files > 0 else 1

            finally:
                if is_url and self.repo_handler:
                    await self.repo_handler.cleanup()
                    self.logger.info("Repository cleanup completed")

        except Exception as e:
            self.logger.error(f"Repository processing failed: {e}")
            return 1


async def main(args: argparse.Namespace) -> int:
    """Main application entry point."""
    generator = DocumentationGenerator()

    try:
        await generator.initialize()
        
        if args.repository:
            return await generator.process_repository(args.repository)
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
        await generator.cleanup()

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate documentation for Python files"
    )
    parser.add_argument(
        '--repository',
        type=str,
        help='Local path or Git URL of the repository to process'
    )
    parser.add_argument(
        '--files',
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
