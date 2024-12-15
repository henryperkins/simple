"""
Main module for running the AI documentation generation process.
"""
# Standard library imports
import argparse
import ast
import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

# Third party imports
import autopep8

# Initialize core console and logging first
from core.config import Config
from core.console import print_error, print_info, print_success, setup_live_layout, stop_live_layout
from core.dependency_injection import Injector, setup_dependencies
from core.logger import LoggerSetup
from core.monitoring import SystemMonitor

# Configure logging
logger = LoggerSetup.get_logger(__name__)

# Register global exception handler
sys.excepthook = LoggerSetup.handle_exception

# Import dependency injection after logging is configured
# Import additional components
from core.docs import DocumentationOrchestrator
# Local application imports
from core.exceptions import ConfigurationError, DocumentationError
from utils import RepositoryManager, get_logger, read_file_safe_async


class DocumentationGenerator:
    """
    A class responsible for generating documentation from source code files and repositories.
    This class handles the generation of documentation for Python source code files and repositories,
    with support for both local and remote repositories. It includes features such as syntax analysis,
    indentation fixing, and metrics collection.

    Attributes:
        logger: A logging instance for tracking operations.
        config (Config): Configuration settings for the documentation generator.
        correlation_id: Unique identifier for tracking operations across the system.
        metrics_collector: Component for collecting and tracking metrics.
        system_monitor (SystemMonitor): Monitor for system operations and health.
        repo_manager (Optional[RepositoryManager]): Manager for repository operations.
        doc_orchestrator (DocumentationOrchestrator): Orchestrator for documentation generation.
        ai_service (Any): Service for AI-related operations.

    Example:
        ```python
        config = Config()
        doc_gen = DocumentationGenerator(config)
        await doc_gen.initialize()
        success = await doc_gen.process_repository("https://github.com/user/repo")
        await doc_gen.cleanup()
        ```

    Raises:
        ConfigurationError: If initialization fails.
        DocumentationError: If documentation generation fails.
        RuntimeError: If system operations fail.
        ValueError: If invalid input is provided.
        IOError: If file operations fail.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the documentation generator with dependency injection.

        Args:
            config: Configuration object to use for initialization.
        """
        self.logger = get_logger()
        self.config = config
        self.correlation_id = Injector.get("correlation_id")
        self.metrics_collector = Injector.get("metrics_collector")
        self.system_monitor = SystemMonitor(
            token_manager=Injector.get("token_manager"),
            metrics_collector=self.metrics_collector,
            correlation_id=self.correlation_id,
        )
        self.repo_manager: Optional[RepositoryManager] = None
        self.doc_orchestrator: DocumentationOrchestrator = Injector.get(
            "doc_orchestrator"
        )
        self.ai_service: Any = Injector.get("ai_service")

    async def initialize(self) -> None:
        """Start systems that require asynchronous setup."""
        try:
            print_info(
                f"Initializing system components with correlation ID: {self.correlation_id}"
            )
            if hasattr(self, "system_monitor"):
                await self.system_monitor.start()
            print_info(
                f"All components initialized successfully with correlation ID: {self.correlation_id}"
            )
        except (RuntimeError, ValueError) as init_error:
            await self.cleanup()
            error_msg = f"Initialization failed: {init_error}"
            raise ConfigurationError(error_msg) from init_error

    async def process_file(self, file_path: Path, output_path: Path, fix_indentation: bool = False) -> bool:
        """Process a single file and generate documentation."""
        try:
            print_info(f"Processing file: {file_path}")
            source_code = await read_file_safe_async(file_path)
            start_time = asyncio.get_event_loop().time()

            if fix_indentation:
                source_code = self._fix_indentation(source_code)

            if not self.analyze_syntax(source_code, file_path):
                print_info(
                    f"Skipping file due to syntax errors: {file_path} with correlation ID: {self.correlation_id}"
                )
                return False

            await self.doc_orchestrator.generate_module_documentation(
                file_path, output_path.parent, source_code=source_code
            )
            success = True
        except DocumentationError as e:
            print_error(f"Failed to generate documentation for {file_path}: {e}")
            raise DocumentationError(f"Error processing file {file_path}") from e
        except (FileNotFoundError, ValueError, IOError) as process_error:
            print_error(f"Error processing file: {process_error}")
            return False

        processing_time = asyncio.get_event_loop().time() - start_time
        await self.metrics_collector.track_operation(
            operation_type="file_processing",
            success=success,
            duration=processing_time,
            metadata={"file_path": str(file_path)},
        )

        print_info(
            f"Finished processing file: {file_path} with correlation ID: {self.correlation_id}"
        )
        return success

    def _fix_indentation(self, source_code: str) -> str:
        """Fix inconsistent indentation using autopep8."""
        try:
            return autopep8.fix_code(source_code)  # type: ignore
        except ImportError:
            print_info("autopep8 not installed. Skipping indentation fix.")
            return source_code

    def analyze_syntax(self, source_code: str, file_path: Path) -> bool:
        """Analyze the syntax of the given source code."""
        try:
            ast.parse(source_code)
            return True
        except SyntaxError as e:
            print_error(
                f"Syntax error in {file_path}: {e}",
                correlation_id=self.correlation_id,
            )
            return False

    async def process_repository(self, repo_path: str, output_dir: Path = Path("docs"), fix_indentation: bool = False) -> bool:
        """Process a repository for documentation."""
        start_time = asyncio.get_event_loop().time()
        success = False
        local_path: Path | None = None

        try:
            print_info(
                f"Starting repository processing: {repo_path} with correlation ID: {self.correlation_id}"
            )
            repo_path = repo_path.strip()

            if self._is_url(repo_path):
                local_path = await self._clone_repository(repo_path)
            else:
                local_path = Path(repo_path)

            if not local_path or not local_path.exists():
                print_error(
                    f"Repository path not found: {local_path or repo_path}"
                )
                return False

            self.doc_orchestrator.code_extractor.context.base_path = local_path
            self.repo_manager = RepositoryManager(local_path)

            # Process each Python file in the repository
            python_files = local_path.rglob("*.py")
            total_files = len(list(python_files))
            processed_files = 0
            skipped_files = 0

            python_files = local_path.rglob("*.py")
            for file_path in python_files:
                output_file = output_dir / (file_path.stem + ".md")
                source_code = await read_file_safe_async(file_path)  # Ensure source code is read
                if source_code:  # Check if source code is not empty
                    if await self.process_file(file_path, output_file, fix_indentation):
                        processed_files += 1
                    else:
                        skipped_files += 1
                else:
                    print_error(f"Source code is missing for file: {file_path}")
                    skipped_files += 1

            success = True
        except (FileNotFoundError, ValueError, IOError) as repo_error:
            print_error(f"Error processing repository {repo_path}: {repo_error}")
        except asyncio.CancelledError:
            print_error("Operation was cancelled.")
        finally:
            processing_time = asyncio.get_event_loop().time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="repository_processing",
                success=success,
                duration=processing_time,
                metadata={
                    "repo_path": str(repo_path),
                    "total_files": total_files,
                    "processed_files": processed_files,
                    "skipped_files": skipped_files,
                },
            )

            print_info(
                f"Finished repository processing: {repo_path} with correlation ID: {self.correlation_id}"
            )
            print_info(
                f"Processed {processed_files} files, skipped {skipped_files} files out of {total_files} total files."
            )

        return success

    def _is_url(self, path: str) -> bool:
        """Check if the given path is a URL."""
        return path.startswith(("http://", "https://", "git@", "ssh://", "ftp://"))

    async def _clone_repository(self, repo_url: str) -> Path | None:
        """Clone a repository and return its local path."""
        try:
            print_info(
                f"Cloning repository: {repo_url} with correlation ID: {self.correlation_id}"
            )
            local_path = Path(".") / repo_url.split("/")[-1].replace(".git", "")
            if local_path.exists():
                print_info(f"Repository already exists at {local_path}")
                return local_path

            process = await asyncio.create_subprocess_exec(
                "git", "clone", repo_url, str(local_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()

            if process.returncode != 0:
                print_error(f"Error cloning repository {repo_url}: {stderr.decode().strip()}")
                return None

            return local_path
        except Exception as e:
            print_error(f"Error cloning repository {repo_url}: {e}")
            return None

    async def cleanup(self) -> None:
        """Cleanup resources used by the DocumentationGenerator."""
        try:
            print_info(f"Starting cleanup process with correlation ID: {self.correlation_id}")
            if hasattr(self, "ai_service") and self.ai_service:
                await self.ai_service.close()
            if hasattr(self, "metrics_collector") and self.metrics_collector:
                await self.metrics_collector.close()
            if hasattr(self, "system_monitor") and self.system_monitor:
                await self.system_monitor.stop()
            print_info(
                f"Cleanup completed successfully with correlation ID: {self.correlation_id}"
            )
        except (RuntimeError, ValueError, IOError) as cleanup_error:
            print_error(f"Error during cleanup: {cleanup_error}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate documentation for Python files or repositories."
    )
    parser.add_argument(
        "--repository",
        type=str,
        help="URL or local path of the repository to process",
    )
    parser.add_argument("--files", nargs="+", help="Python files to process")
    parser.add_argument(
        "--output",
        type=str,
        default="docs",
        help="Output directory for documentation (default: docs)",
    )
    parser.add_argument(
        "--fix-indentation",
        action="store_true",
        help="Enable indentation fixing using autopep8",
    )
    parser.add_argument(
        "--live-layout",
        action="store_true",
        help="Enable live layout using rich",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> int:
    """Main entry point for the documentation generator."""
    doc_generator: DocumentationGenerator | None = None
    try:
        correlation_id = str(uuid.uuid4())
        setup_dependencies(Config(), correlation_id)

        if args.live_layout:
            setup_live_layout()

        doc_generator = DocumentationGenerator(config=Injector.get("config"))
        await doc_generator.initialize()

        if args.repository:
            print_info(f"Processing repository: {args.repository}")
            success = await doc_generator.process_repository(
                args.repository, Path(args.output), args.fix_indentation
            )
            print_success(f"Repository documentation generated successfully: {success}")

        if args.files:
            for file_path in args.files:
                output_path = Path(args.output) / (Path(file_path).stem + ".md")
                success = await doc_generator.process_file(Path(file_path), output_path, args.fix_indentation)
                print_success(
                    f"Documentation generated successfully for {file_path}: {success}"
                )

    except (RuntimeError, ValueError, IOError) as unexpected_error:
        print_error(f"Unexpected error: {unexpected_error}")
        return 1
    except KeyError as ke:
        print_error(f"Dependency injection error: {ke}")
        return 1
    except asyncio.CancelledError:
        print_error("Operation was cancelled.")
        return 1
    finally:
        if doc_generator:
            await doc_generator.cleanup()
        if args.live_layout:
            stop_live_layout()
        print_info("Exiting documentation generation")

    return 0


if __name__ == "__main__":
    cli_args = parse_arguments()
    print_info(f"Command-line arguments: {cli_args}")
    exit_code = asyncio.run(main(cli_args))
    sys.exit(exit_code)
