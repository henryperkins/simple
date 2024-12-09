"""
Main documentation generation coordinator with monitoring.
"""

from core.ai_service import AIService
from core.config import Config
from core.docs import DocumentationOrchestrator
from core.exceptions import ConfigurationError, DocumentationError
from core.logger import LoggerSetup, CorrelationLoggerAdapter, log_error, log_info
from core.metrics_collector import MetricsCollector
from core.monitoring import SystemMonitor
from core.types.base import Injector, MetricData, DocstringData
import uuid

from utils import (
    ensure_directory,
    read_file_safe,
    RepositoryManager
)

# Register dependencies
Injector.register('metric_calculator', lambda element: MetricData())
Injector.register('docstring_parser', lambda docstring: DocstringData(summary=docstring))

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional, Union

# Configure logger globally with dynamic settings
LOG_DIR = "logs"  # This could be set via an environment variable or command-line argument
LoggerSetup.configure(level="DEBUG", log_dir=log_dir)

# Set global exception handler
sys.excepthook = LoggerSetup.handle_exception

class DocumentationGenerator:
    """Main documentation generation coordinator with monitoring."""

    def __init__(self) -> None:
        """Initialize the documentation generator."""
        self.config = Config()
        self.correlation_id = str(uuid.uuid4())
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__), 
            correlation_id=self.correlation_id
        )

        # Initialize core components
        self.metrics_collector = MetricsCollector(correlation_id=self.correlation_id)
        self.ai_service = AIService(config=self.config.ai, correlation_id=self.correlation_id)
        self.doc_orchestrator = DocumentationOrchestrator(
            ai_service=self.ai_service,
            correlation_id=self.correlation_id
        )
        self.system_monitor = SystemMonitor(
            token_manager=self.ai_service.token_manager,
            metrics_collector=self.metrics_collector,
            correlation_id=self.correlation_id
        )
        self.repo_manager = None

    async def initialize(self) -> None:
        """Start systems that require asynchronous setup."""
        try:
            self.logger.info("Initializing system components")
            await self.system_monitor.start()
            self.logger.info("All components initialized successfully")
        except Exception as init_error:
            error_msg = f"Initialization failed: {init_error}"
            self.logger.error(error_msg, exc_info=True)
            await self.cleanup()
            raise ConfigurationError(error_msg) from init_error

    async def process_file(self, file_path: Path, output_path: Path) -> bool:
        """Process a single file and generate documentation."""
        try:
            self.logger.info(f"Processing file: {file_path}")
            start_time: float = asyncio.get_event_loop().time()
            
            source_code: str = read_file_safe(file_path)
            source_code = self._fix_indentation(source_code)

            try:
                await self.doc_orchestrator.generate_module_documentation(
                    file_path,
                    output_path.parent,
                    source_code=source_code
                )
                success = True
            except DocumentationError as e:
                self.logger.error(f"Failed to generate documentation for {file_path}: {e}")
                success = False
            except Exception as e:
                self.logger.error(f"Unexpected error processing file {file_path}: {e}", exc_info=True)
                success = False

            processing_time: float = asyncio.get_event_loop().time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="file_processing",
                success=success,
                duration=processing_time,
                metadata={"file_path": str(file_path)}
            )
            
            self.logger.info(f"Finished processing file: {file_path}")
            return success

        except Exception as process_error:
            self.logger.error(f"Error processing file: {process_error}", exc_info=True)
            return False

    def _fix_indentation(self, source_code: str) -> str:
        """Fix inconsistent indentation using autopep8."""
        try:
            import autopep8
            return autopep8.fix_code(source_code)
        except ImportError:
            self.logger.warning("autopep8 not installed. Skipping indentation fix.")
            return source_code

    async def process_repository(self, repo_path: str, output_dir: Path = Path("docs")) -> bool:
        """Process a repository for documentation."""
        start_time = asyncio.get_event_loop().time()
        success = False
        local_path: Optional[Path] = None
        
        try:
            self.logger.info(f"Starting repository processing: {repo_path}")
            repo_path = repo_path.strip()

            if self._is_url(repo_path):
                local_path = await self._clone_repository(repo_path)
            else:
                local_path = Path(repo_path)

            if not local_path or not local_path.exists():
                raise FileNotFoundError(f"Repository path not found: {local_path or repo_path}")

            if not self.repo_manager:
                self.repo_manager = RepositoryManager(local_path)

            self.doc_orchestrator.code_extractor.context.base_path = local_path
            success = await self._process_local_repository(local_path, output_dir)

        except Exception as repo_error:
            self.logger.error(f"Error processing repository {repo_path}: {repo_error}", exc_info=True)
            success = False
        finally:
            processing_time: float = asyncio.get_event_loop().time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="repository_processing",
                success=success,
                duration=processing_time,
                metadata={"repo_path": str(repo_path)}
            )
            self.logger.info(f"Finished repository processing: {repo_path}")
            return success

    def _is_url(self, path: Union[str, Path]) -> bool:
        """Check if the path is a URL."""
        path_str = str(path)
        return path_str.startswith(('http://', 'https://', 'git@', 'ssh://', 'ftp://'))

    async def _clone_repository(self, repo_url: str) -> Path:
        """Clone a repository and return its local path."""
        self.logger.info(f"Cloning repository: {repo_url}")
        try:
            if not self.repo_manager:
                self.repo_manager = RepositoryManager(Path('.'))
            repo_path = await self.repo_manager.clone_repository(repo_url)
            self.logger.info(f"Successfully cloned repository to {repo_path}")
            return repo_path
        except Exception as clone_error:
            self.logger.error(f"Failed to clone repository: {clone_error}", exc_info=True)
            raise DocumentationError(f"Repository cloning failed: {clone_error}") from clone_error

    async def _process_local_repository(self, repo_path: Path, output_dir: Path) -> bool:
        """Process a local repository."""
        try:
            self.logger.info(f"Processing local repository: {repo_path}")
            output_dir = ensure_directory(output_dir)
            python_files = repo_path.rglob("*.py")

            for file_path in python_files:
                output_file = output_dir / (file_path.stem + ".md")
                success = await self.process_file(file_path, output_file)
                if not success:
                    self.logger.error(f"Failed to process file: {file_path}")

            self.logger.info(f"Finished processing local repository: {repo_path}")
            return True

        except Exception as local_repo_error:
            self.logger.error(f"Error processing local repository: {local_repo_error}", exc_info=True)
            return False

    async def display_metrics(self) -> None:
        """Display collected metrics and system performance metrics."""
        try:
            self.logger.info("Displaying metrics")
            collected_metrics = self.metrics_collector.metrics_history
            system_metrics = self.system_monitor.get_metrics()

            print("\n=== Documentation Generation Metrics ===")
            for module, metrics_list in collected_metrics.items():
                for metric in metrics_list:
                    print(f"Module: {module}")
                    print(f"Timestamp: {metric['timestamp']}")
                    print(f"Metrics: {metric['metrics']}")
                    print("-" * 40)

            print("\n=== System Performance Metrics ===")
            print(f"CPU Usage: {system_metrics.get('cpu', {}).get('percent', 0)}%")
            print(f"Memory Usage: {system_metrics.get('memory', {}).get('percent', 0)}%")
            print(f"Status: {system_metrics.get('status', 'unknown')}")
            print("-" * 40)

        except Exception as display_error:
            self.logger.error(f"Error displaying metrics: {display_error}", exc_info=True)

    async def cleanup(self) -> None:
        """Cleanup resources used by the DocumentationGenerator."""
        try:
            self.logger.info("Starting cleanup process")
            if self.ai_service:
                await self.ai_service.close()
            if self.metrics_collector:
                await self.metrics_collector.close()
            if self.system_monitor:
                await self.system_monitor.stop()
            self.logger.info("Cleanup completed successfully")
        except Exception as cleanup_error:
            self.logger.error(f"Error during cleanup: {cleanup_error}", exc_info=True)

async def main(args: argparse.Namespace) -> int:
    """Main function to manage documentation generation process."""
    exit_code = 1
    doc_generator: Optional[DocumentationGenerator] = None

    try:
        log_info("Starting documentation generation")
        doc_generator = DocumentationGenerator()
        await doc_generator.initialize()

        if args.repository:
            log_info(f"Processing repository: {args.repository}")
            success = await doc_generator.process_repository(
                args.repository,
                Path(args.output)
            )
            log_info("Repository documentation generated successfully" if success else "Failed to generate repository documentation")

        if args.files:
            for file in args.files:
                log_info(f"Processing file: {file}")
                file_path = Path(file)
                output_path = Path(args.output) / (file_path.stem + ".md")
                success = await doc_generator.process_file(file_path, output_path)
                log_info(f"Documentation for {file} generated successfully" if success else f"Failed to generate documentation for {file}")

        await doc_generator.display_metrics()
        exit_code = 0

    except DocumentationError as de:
        log_error(f"Documentation generation failed: {de}")
    except Exception as unexpected_error:
        log_error(f"Unexpected error: {unexpected_error}")
    finally:
        if doc_generator:
            await doc_generator.cleanup()
        log_info("Exiting documentation generation")
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
        args = parse_arguments()
        log_info(f"Command-line arguments: {args}")
        exit_code = asyncio.run(main(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log_info("Documentation generation interrupted by user")
        sys.exit(1)
    except Exception as run_error:
        log_error(f"Failed to run documentation generator: {run_error}")
        sys.exit(1)
    finally:
        LoggerSetup.shutdown()
