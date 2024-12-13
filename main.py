"""Main module for running the AI documentation generation process."""

# Standard library imports
import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional
import ast  # Ensure ast is imported
import uuid
import logging  # Add this import
from core.config import Config
from core.dependency_injection import setup_dependencies, Injector

# Third party imports
import autopep8
import git

# Initialize core console and logging first
from core.console import (
    DEFAULT_CONSOLE,
    print_info,
    print_error,
    print_success,
    setup_logging  # Ensure setup_logging is imported
)
from core.logger import LoggerSetup
from core.config import Config
from core.monitoring import SystemMonitor  # Ensure SystemMonitor is imported

# Configure logging with console
setup_logging(logging.INFO)  # Use setup_logging to ensure consistent format
logger = LoggerSetup.get_logger(__name__)

# Register global exception handler
sys.excepthook = LoggerSetup.handle_exception

# Import dependency injection after logging is configured  
from core.dependency_injection import setup_dependencies, Injector

# Local application imports
from core.exceptions import ConfigurationError, DocumentationError
from utils import (
    ensure_directory,
    read_file_safe_async,
    RepositoryManager,
    get_logger
)

# Import additional components
from core.ai_service import AIService
from core.extraction.code_extractor import CodeExtractor
from core.docstring_processor import DocstringProcessor
from core.docs import DocumentationOrchestrator
from core.markdown_generator import MarkdownGenerator
from core.prompt_manager import PromptManager
from core.response_parsing import ResponseParsingService

class DocumentationGenerator:
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
        self.repo_manager = None
        self.doc_orchestrator = Injector.get("doc_orchestrator")  # Add this line

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

    async def process_file(self, file_path: Path, output_path: Path) -> bool:
        """Process a single file and generate documentation."""
        try:
            print_info(f"Processing file: {file_path}")
            source_code = await read_file_safe_async(file_path)
            start_time = asyncio.get_event_loop().time()

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
            return autopep8.fix_code(source_code)
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

    async def process_repository(self, repo_path: str, output_dir: Path = Path("docs")) -> bool:
        """Process a repository for documentation."""
        start_time = asyncio.get_event_loop().time()
        success = False
        local_path: Optional[Path] = None

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
            for file_path in python_files:
                output_file = output_dir / (file_path.stem + ".md")
                success = await self.process_file(file_path, output_file)
                if not success:
                    print_error(f"Failed to process file: {file_path}")

            success = True
        except (FileNotFoundError, ValueError, IOError) as repo_error:
            print_error(f"Error processing repository {repo_path}: {repo_error}")
        finally:
            processing_time = asyncio.get_event_loop().time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="repository_processing",
                success=success,
                duration=processing_time,
                metadata={"repo_path": str(repo_path)},
            )

            print_info(
                f"Finished repository processing: {repo_path} with correlation ID: {self.correlation_id}"
            )

        return success

    def _is_url(self, path: str) -> bool:
        """Check if the given path is a URL."""
        return path.startswith(("http://", "https://", "git@", "ssh://", "ftp://"))

    async def _clone_repository(self, repo_url: str) -> Optional[Path]:
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
            stdout, stderr = await process.communicate()

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

def parse_arguments():
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
    return parser.parse_args()

async def main(args):
    """Main entry point for the documentation generator."""
    doc_generator = None  # Ensure doc_generator is defined
    try:
        correlation_id = str(uuid.uuid4())
        setup_dependencies(Config(), correlation_id)

        doc_generator = DocumentationGenerator(config=Injector.get("config"))
        await doc_generator.initialize()

        if args.repository:
            print_info(f"Processing repository: {args.repository}")
            success = await doc_generator.process_repository(
                args.repository, Path(args.output)
            )
            print_success(f"Repository documentation generated successfully: {success}")

        if args.files:
            for file_path in args.files:
                output_path = Path(args.output) / (Path(file_path).stem + ".md")
                success = await doc_generator.process_file(Path(file_path), output_path)
                print_success(
                    f"Documentation generated successfully for {file_path}: {success}"
                )

    except (RuntimeError, ValueError, IOError) as unexpected_error:
        print_error(f"Unexpected error: {unexpected_error}")
        return 1
    except KeyError as ke:
        print_error(f"Dependency injection error: {ke}")
        return 1
    finally:
        if doc_generator:
            await doc_generator.cleanup()
        print_info("Exiting documentation generation")

    return 0

if __name__ == "__main__":
    try:
        cli_args = parse_arguments()
        print_info(f"Command-line arguments: {cli_args}")
        exit_code = asyncio.run(main(cli_args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print_info("Documentation generation interrupted by user")
        sys.exit(1)
    except (RuntimeError, ValueError, IOError) as run_error:
        print_error(f"Failed to run documentation generator: {run_error}")
        sys.exit(1)
    finally:
        LoggerSetup.shutdown()