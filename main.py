"""Main module for running the AI documentation generation process."""
import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, Union

import uuid
import ast
import autopep8
import git

from core.ai_service import AIService
from core.config import Config
from core.docs import DocumentationOrchestrator
from core.exceptions import ConfigurationError, DocumentationError
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.metrics_collector import MetricsCollector
from core.monitoring import SystemMonitor
from core.types.base import Injector, MetricData, DocstringData
from core.docstring_processor import DocstringProcessor
from core.response_parsing import ResponseParsingService
from api.token_management import TokenManager
from utils import RepositoryManager

from hello import hello
from core.console import (
    print_info,
    print_error,
    print_success,
    setup_logging
)

from utils import (
    ensure_directory,
    read_file_safe,
    is_url,
    RepositoryManager
)

# Setup dependencies
from core.dependency_injection import setup_dependencies
setup_dependencies()

# Configure logger globally with dynamic settings
LOG_DIR = "logs"  # This could be set via an environment variable or command-line argument
LoggerSetup.configure(level="DEBUG", log_dir=LOG_DIR)

# Set global exception handler
sys.excepthook = LoggerSetup.handle_exception


class DocumentationGenerator:
    """Main documentation generation coordinator with monitoring."""

    def __init__(self, config: Config) -> None:
        """
        Initialize the documentation generator with dependency injection.

        Args:
            config: Configuration object to use for initialization.
        """
        self.config = config
        self.correlation_id = str(uuid.uuid4())
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__),
            correlation_id=self.correlation_id
        )

        # Initialize core components with dependency injection
        self.ai_service = Injector.get('ai_service')
        self.doc_orchestrator = DocumentationOrchestrator(
            ai_service=self.ai_service,
            correlation_id=self.correlation_id
        )
        self.metrics_collector = MetricsCollector(
            correlation_id=self.correlation_id)
        self.system_monitor = SystemMonitor(
            token_manager=self.ai_service.token_manager,
            metrics_collector=self.metrics_collector,
            correlation_id=self.correlation_id
        )
        self.repo_manager = None

    async def initialize(self) -> None:
        """Start systems that require asynchronous setup."""
        try:
            print_info("Initializing system components", correlation_id=self.correlation_id)
            if hasattr(self, 'system_monitor'):
                await self.system_monitor.start()
            print_info("All components initialized successfully", correlation_id=self.correlation_id)
        except (RuntimeError, ValueError) as init_error:
            error_msg = f"Initialization failed: {init_error}"
            print_error(error_msg, correlation_id=self.correlation_id)
            await self.cleanup()
            raise ConfigurationError(error_msg) from init_error

    async def process_file(self, file_path: Path, output_path: Path) -> bool:
        """Process a single file and generate documentation."""
        try:
            print_info(f"Processing file: {file_path}", correlation_id=self.correlation_id)
            start_time = asyncio.get_event_loop().time()

            source_code = read_file_safe(file_path)
            source_code = self._fix_indentation(source_code)

            # Analyze syntax before processing
            if not self.analyze_syntax(source_code, file_path):
                print_info(f"Skipping file due to syntax errors: {file_path}", correlation_id=self.correlation_id)
                return False

            try:
                await self.doc_orchestrator.generate_module_documentation(
                    file_path,
                    output_path.parent,
                    source_code=source_code
                )
                success = True
            except DocumentationError as e:
                print_error(f"Failed to generate documentation for {file_path}: {e}", correlation_id=self.correlation_id)
                success = False
            except Exception as e:
                display_error(e, {"file_path": str(file_path)}, correlation_id=self.correlation_id)
                success = False

            processing_time = asyncio.get_event_loop().time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="file_processing",
                success=success,
                metadata={"file_path": str(file_path)},
                duration=processing_time
            )

            print_info(f"Finished processing file: {file_path}", correlation_id=self.correlation_id)
            return success

        except (FileNotFoundError, ValueError, IOError) as process_error:
            print_error(f"Error processing file: {process_error}", correlation_id=self.correlation_id)
            return False

    def _fix_indentation(self, source_code: str) -> str:
        """Fix inconsistent indentation using autopep8."""
        try:
            return autopep8.fix_code(source_code)
        except ImportError:
            print_info("autopep8 not installed. Skipping indentation fix.", correlation_id=self.correlation_id)
            return source_code

    def analyze_syntax(self, source_code: str, file_path: Path) -> bool:
        """Analyze the syntax of the given source code."""
        try:
            ast.parse(source_code)
            return True
        except SyntaxError as e:
            print_error(f"Syntax error in {file_path}: {e}", correlation_id=self.correlation_id)
            return False

    async def process_repository(self, repo_path: str, output_dir: Path = Path("docs")) -> bool:
        """Process a repository for documentation."""
        start_time = asyncio.get_event_loop().time()
        success = False
        local_path: Optional[Path] = None

        try:
            print_info(f"Starting repository processing: {repo_path}", correlation_id=self.correlation_id)
            repo_path = repo_path.strip()

            if is_url(repo_path):
                local_path = await self._clone_repository(repo_path)
            else:
                local_path = Path(repo_path)

            if not local_path or not local_path.exists():
                raise FileNotFoundError(f"Repository path not found: {local_path or repo_path}")

            if not self.repo_manager:
                self.repo_manager = RepositoryManager(local_path)

            self.doc_orchestrator.code_extractor.context.base_path = local_path
            success = await self._process_local_repository(local_path, output_dir)

        except (FileNotFoundError, ValueError, IOError) as repo_error:
            print_error(f"Error processing repository {repo_path}: {repo_error}", correlation_id=self.correlation_id)
            success = False
        finally:
            processing_time = asyncio.get_event_loop().time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="repository_processing",
                success=success,
                duration=processing_time,
                metadata={"repo_path": str(repo_path)}
            )
            print_info(f"Finished repository processing: {repo_path}", correlation_id=self.correlation_id)

    def _is_url(self, path: Union[str, Path]) -> bool:
        """Check if the path is a URL."""
        path_str = str(path)
        return path_str.startswith(('http://', 'https://', 'git@', 'ssh://', 'ftp://'))

    async def _clone_repository(self, repo_url: str) -> Path:
        """Clone a repository and return its local path."""
        print_info(f"Cloning repository: {repo_url}", correlation_id=self.correlation_id)
        try:
            if not self.repo_manager:
                self.repo_manager = RepositoryManager(Path('.'))
            repo_path = await self.repo_manager.clone_repository(repo_url)
            print_info(f"Successfully cloned repository to {repo_path}", correlation_id=self.correlation_id)
            return repo_path
        except (git.GitCommandError, ValueError, IOError) as clone_error:
            display_error(clone_error, {"repo_url": repo_url}, correlation_id=self.correlation_id)
            raise DocumentationError(f"Repository cloning failed: {clone_error}") from clone_error

    async def _process_local_repository(self, repo_path: Path, output_dir: Path) -> bool:
        """Process a local repository."""
        try:
            print_info(f"Processing local repository: {repo_path}", correlation_id=self.correlation_id)
            output_dir = ensure_directory(output_dir)
            python_files = repo_path.rglob("*.py")

            for file_path in python_files:
                output_file = output_dir / (file_path.stem + ".md")
                success = await self.process_file(file_path, output_file)
                if not success:
                    print_error(f"Failed to process file: {file_path}", correlation_id=self.correlation_id)

            print_info(f"Finished processing local repository: {repo_path}", correlation_id=self.correlation_id)
            return True

        except (FileNotFoundError, ValueError, IOError) as local_repo_error:
            print_error(f"Error processing local repository: {local_repo_error}", correlation_id=self.correlation_id)
            return False

    async def display_metrics(self) -> None:
        """Display collected metrics and system performance metrics."""
        try:
            print_info("Displaying metrics", correlation_id=self.correlation_id)
            collected_metrics = self.metrics_collector.get_metrics()
            system_metrics = self.system_monitor.get_metrics()

            print("\n=== System Performance Metrics ===")
            print(f"CPU Usage: {system_metrics.get('cpu', {}).get('percent', 0)}%")
            print(f"Memory Usage: {system_metrics.get('memory', {}).get('percent', 0)}%")
            print(f"Status: {system_metrics.get('status', 'unknown')}")
            print("-" * 40)

        except (KeyError, ValueError, IOError) as display_error:
            print_error(f"Error displaying metrics: {display_error}", correlation_id=self.correlation_id)

    async def cleanup(self) -> None:
        """Cleanup resources used by the DocumentationGenerator."""
        try:
            print_info("Starting cleanup process", correlation_id=self.correlation_id)
            if hasattr(self, 'ai_service') and self.ai_service:
                await self.ai_service.close()
            if hasattr(self, 'metrics_collector') and self.metrics_collector:
                await self.metrics_collector.close()
            if hasattr(self, 'system_monitor') and self.system_monitor:
                await self.system_monitor.stop()
            print_info("Cleanup completed successfully", correlation_id=self.correlation_id)
        except (RuntimeError, ValueError, IOError) as cleanup_error:
            print_error(f"Error during cleanup: {cleanup_error}", correlation_id=self.correlation_id)


async def main(args: argparse.Namespace) -> int:
    """Main function to manage documentation generation process."""
    exit_code = 1
    doc_generator: Optional[DocumentationGenerator] = None

    try:
        config = Config()
        setup_logging(config.app.log_level)
        print_info("Starting documentation generation")
        doc_generator = DocumentationGenerator(config)
        await doc_generator.initialize()

        if args.repository:
            print_info(f"Processing repository: {args.repository}")
            success = await doc_generator.process_repository(
                args.repository,
                Path(args.output)
            )
            print_success(f"Repository documentation generated successfully: {success}")

        if args.files:
            for file in args.files:
                print_info(f"Processing file: {file}")
                file_path = Path(file)
                output_path = Path(args.output) / (file_path.stem + ".md")
                success = await doc_generator.process_file(file_path, output_path)
                print_success(f"Documentation generated successfully for {file}: {success}")

        await doc_generator.display_metrics()
        exit_code = 0

    except DocumentationError as de:
        print_error(f"Documentation generation failed: {de}")
    except (RuntimeError, ValueError, IOError) as unexpected_error:
        print_error(f"Unexpected error: {unexpected_error}")
    finally:
        if doc_generator:
            await doc_generator.cleanup()
        print_info("Exiting documentation generation")

    return exit_code


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
    parser.add_argument(
        "--files",
        nargs="+",
        help="Python files to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs",
        help="Output directory for documentation (default: docs)",
    )
    return parser.parse_args()


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
