"""
Main documentation generation coordinator with monitoring.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from core.ai_service import AIService
from core.config import Config
from core.docs import DocumentationOrchestrator
from core.logger import LoggerSetup, CorrelationLoggerAdapter, log_error, log_info
from core.metrics_collector import MetricsCollector
from core.monitoring import SystemMonitor
from utils import (
    ensure_directory,
    read_file_safe,
    RepositoryManager
)
from exceptions import ConfigurationError, DocumentationError
import uuid

# Configure logger globally with dynamic settings
log_dir = "logs"  # This could be set via an environment variable or command-line argument
LoggerSetup.configure(level="DEBUG", log_dir=log_dir)

# Optionally, set the global exception handler
sys.excepthook = LoggerSetup.handle_exception

class DocumentationGenerator:
    """
    Main documentation generation coordinator with monitoring.
    Manages the overall documentation generation process while monitoring
    system resources and collecting performance metrics.
    """

    def __init__(self) -> None:
        """Initialize the documentation generator."""
        self.config = Config()
        base_logger = LoggerSetup.get_logger(__name__)
        correlation_id = str(uuid.uuid4())
        self.logger = CorrelationLoggerAdapter(base_logger, correlation_id=correlation_id)

        # Initialize core components
        self.ai_service = AIService(config=self.config.ai)
        self.doc_orchestrator = DocumentationOrchestrator(ai_service=self.ai_service)
        self.system_monitor = SystemMonitor()
        self.metrics_collector = MetricsCollector()

    async def initialize(self) -> None:
        """Start systems that require asynchronous setup."""
        try:
            self.logger.info("Initializing system components", extra={'correlation_id': self.logger.correlation_id})
            await self.system_monitor.start()
            self.logger.info("All components initialized successfully", extra={'correlation_id': self.logger.correlation_id})
        except Exception as e:
            error_msg = f"Initialization failed: {e}"
            self.logger.error(error_msg, exc_info=True, extra={'correlation_id': self.logger.correlation_id})
            await self.cleanup()
            raise ConfigurationError(error_msg) from e

    async def process_file(self, file_path: Path, output_path: Path) -> bool:
        """Process a single file and generate documentation."""
        try:
            self.logger.info(f"Processing file: {file_path}", extra={'correlation_id': self.logger.correlation_id})

            # Track operation metrics
            start_time = asyncio.get_event_loop().time()

            source_code = read_file_safe(file_path)
            source_code = self._fix_indentation(source_code)

            # Pass the fixed source_code to the orchestrator
            await self.doc_orchestrator.generate_module_documentation(
                file_path,
                output_path.parent,
                source_code=source_code  # Pass the updated source code
            )

            # Record metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="file_processing",
                success=True,
                duration=processing_time,
                metadata={"file_path": str(file_path)}
            )

            self.logger.info(f"Successfully processed file: {file_path}", extra={'correlation_id': self.logger.correlation_id})
            return True

        except Exception as e:
            self.logger.error(f"Error processing file: {e}", exc_info=True, extra={'correlation_id': self.logger.correlation_id})

            # Record failure metrics
            await self.metrics_collector.track_operation(
                operation_type="file_processing",
                success=False,
                duration=0,
                error=str(e),
                metadata={"file_path": str(file_path)}
            )

            return False

    def _fix_indentation(self, source_code: str) -> str:
        """Fix inconsistent indentation using autopep8."""
        import autopep8
        return autopep8.fix_code(source_code)

    async def process_repository(self, repo_path: str, output_dir: Path = Path("docs")) -> bool:
        """Process a repository for documentation."""
        try:
            self.logger.info(f"Starting repository processing: {repo_path}", extra={'correlation_id': self.logger.correlation_id})
            if self._is_url(repo_path):
                self.logger.info(f"Processing repository from URL: {repo_path}", extra={'correlation_id': self.logger.correlation_id})
                repo_path = await self._clone_repository(repo_path)
            else:
                self.logger.info(f"Processing local repository: {repo_path}", extra={'correlation_id': self.logger.correlation_id})
                repo_path = Path(repo_path)

            if not repo_path.exists():
                raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

            start_time = asyncio.get_event_loop().time()

            # Process repository
            success = await self._process_local_repository(repo_path, output_dir)

            # Record metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="repository_processing",
                success=success,
                duration=processing_time,
                metadata={"repo_path": str(repo_path)}
            )

            self.logger.info(f"Repository processing completed: {repo_path}", extra={'correlation_id': self.logger.correlation_id})
            return success

        except Exception as e:
            self.logger.error(f"Repository processing failed: {e}", exc_info=True, extra={'correlation_id': self.logger.correlation_id})

            # Record failure metrics
            await self.metrics_collector.track_operation(
                operation_type="repository_processing",
                success=False,
                duration=0,
                error=str(e),
                metadata={"repo_path": repo_path}
            )

            return False

    def _is_url(self, path: str) -> bool:
        """Simple check to determine if the path is a URL."""
        return path.startswith(('http://', 'https://', 'git@', 'ssh://', 'ftp://'))

    async def _process_local_repository(self, repo_path: Path, output_dir: Path) -> bool:
        """Process a local repository for documentation."""
        try:
            output_dir = ensure_directory(output_dir)
            python_files = list(repo_path.rglob("*.py"))

            for file_path in python_files:
                output_file = output_dir / (file_path.stem + ".md")
                success = await self.process_file(file_path, output_file)
                if not success:
                    self.logger.error(f"Failed to process file: {file_path}", extra={'correlation_id': self.logger.correlation_id})

            return True

        except Exception as e:
            self.logger.error(f"Error processing local repository: {e}", exc_info=True, extra={'correlation_id': self.logger.correlation_id})
            return False

    async def display_metrics(self) -> None:
        """Display collected metrics and system performance metrics."""
        try:
            self.logger.info("Displaying metrics", extra={'correlation_id': self.logger.correlation_id})
            collected_metrics = self.metrics_collector.get_metrics()
            system_metrics = self.system_monitor.get_metrics()

            print("\n=== Documentation Generation Metrics ===")
            for metric in collected_metrics['operations']:
                print(f"Operation: {metric['operation_type']}")
                print(f"Success: {metric['success']}")
                print(f"Duration: {metric['duration']:.2f} seconds")
                print(f"Timestamp: {metric['timestamp']}")
                if 'error' in metric:
                    print(f"Error: {metric['error']}")
                print("-" * 40)

            print("\n=== System Performance Metrics ===")
            print(f"CPU Usage: {system_metrics.get('cpu', {}).get('percent', 0)}%")
            print(f"Memory Usage: {system_metrics.get('memory', {}).get('percent', 0)}%")
            print(f"Status: {system_metrics.get('status', 'unknown')}")
            print("-" * 40)

        except Exception as e:
            self.logger.error(f"Error displaying metrics: {e}", exc_info=True, extra={'correlation_id': self.logger.correlation_id})

    async def cleanup(self) -> None:
        """Cleanup resources used by the DocumentationGenerator."""
        try:
            self.logger.info("Starting cleanup process", extra={'correlation_id': self.logger.correlation_id})
            if self.ai_service:
                await self.ai_service.close()
            if self.metrics_collector:
                await self.metrics_collector.close()
            if self.system_monitor:
                await self.system_monitor.stop()

            self.logger.info("Cleanup completed successfully", extra={'correlation_id': self.logger.correlation_id})
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True, extra={'correlation_id': self.logger.correlation_id})

    async def _clone_repository(self, repo_url: str) -> Path:
        """Clone a repository and return its local path."""
        self.logger.info(f"Cloning repository from URL: {repo_url}", extra={'correlation_id': self.logger.correlation_id})
        repo_manager = RepositoryManager(Path('.'))
        cloned_path = await repo_manager.clone_repository(repo_url)
        self.logger.info(f"Repository cloned to: {cloned_path}", extra={'correlation_id': self.logger.correlation_id})
        return cloned_path


async def main(args: argparse.Namespace) -> int:
    """Main function to manage documentation generation process."""
    log_info("Starting main documentation generation process")
    doc_generator = DocumentationGenerator()
    try:
        await doc_generator.initialize()

        if args.repository:
            log_info(f"Processing repository: {args.repository}")
            success = await doc_generator.process_repository(
                args.repository,
                Path(args.output)
            )
            if success:
                log_info("Repository documentation generated successfully")
            else:
                log_error("Failed to generate repository documentation")

        if args.files:
            for file in args.files:
                log_info(f"Processing file: {file}")
                file_path = Path(file)
                output_path = Path(args.output) / (file_path.stem + ".md")
                success = await doc_generator.process_file(file_path, output_path)
                if success:
                    log_info(f"Documentation for {file} generated successfully")
                else:
                    log_error(f"Failed to generate documentation for {file}")

        await doc_generator.display_metrics()
        return 0

    except DocumentationError as de:
        log_error(f"Documentation generation failed: {de}")
        return 1
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        return 1
    finally:
        await doc_generator.cleanup()


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
        exit(exit_code)
    except KeyboardInterrupt:
        log_info("Documentation generation interrupted by user")
        exit(1)
    except Exception as e:
        log_error(f"Failed to run documentation generator: {e}")
        exit(1)
    finally:
        LoggerSetup.shutdown()