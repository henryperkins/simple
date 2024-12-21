import argparse
import ast
import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any

import autopep8
from rich.progress import Progress

from core.config import Config
from core.console import (
    display_metrics,
    print_debug,
    print_error,
    print_info,
    print_phase_header,
    print_section_break,
    print_status,
    print_success,
)
from core.dependency_injection import Injector, setup_dependencies
from core.logger import LoggerSetup
from core.monitoring import SystemMonitor
from core.docs import DocumentationOrchestrator
from core.docstring_processor import DocstringProcessor
from core.exceptions import ConfigurationError, DocumentationError
from utils import RepositoryManager, fetch_dependency, log_and_raise_error

# Configure logging
logger = LoggerSetup.get_logger(__name__)

# Register global exception handler
sys.excepthook = LoggerSetup.handle_exception

class DocumentationGenerator:
    """
    A class responsible for generating documentation from source code files and
    repositories.
    This class handles the generation of documentation for Python source code
    files and repositories,
    with support for both local and remote repositories. It includes features
    such as syntax analysis,
    indentation fixing, and metrics collection.

    Attributes:
        logger: A logging instance for tracking operations.
        config (Config): Configuration settings for the documentation generator.
        correlation_id: Unique identifier for tracking operations across the
        system.
        metrics_collector: Component for collecting and tracking metrics.
        system_monitor (SystemMonitor): Monitor for system operations and health.
        repo_manager (Optional[RepositoryManager]): Manager for repository
        operations.
        doc_orchestrator (DocumentationOrchestrator): Orchestrator for
        documentation generation.
        ai_service (Any): Service for AI-related operations.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the documentation generator with dependency injection.

        Args:
            config: Configuration object to use for initialization.
        """
        self.logger = fetch_dependency("logger")
        self.config = config
        self.correlation_id = fetch_dependency("correlation_id")
        self.metrics_collector = fetch_dependency("metrics_collector")
        self.system_monitor = SystemMonitor(
            token_manager=fetch_dependency("token_manager"),
            metrics_collector=self.metrics_collector,
            correlation_id=self.correlation_id,
        )
        self.repo_manager: RepositoryManager | None = None
        self.doc_orchestrator: DocumentationOrchestrator = fetch_dependency(
            "doc_orchestrator"
        )
        self.docstring_processor: DocstringProcessor = fetch_dependency(
            "docstring_processor"
        )
        self.ai_service: Any = fetch_dependency("ai_service")
        self.read_file_safe_async = fetch_dependency("read_file_safe_async")

    async def initialize(self) -> None:
        """Start systems that require asynchronous setup."""
        try:
            self.logger.debug(
                "Initializing system components with correlation ID: "
                f"{self.correlation_id}"
            )
            if hasattr(self, "system_monitor"):
                await self.system_monitor.start()
            print_info(
                "All components initialized successfully with correlation ID: "
                f"{self.correlation_id}"
            )
        except (RuntimeError, ValueError) as init_error:
            log_and_raise_error(
                self.logger,
                init_error,
                ConfigurationError,
                "Initialization failed",
                self.correlation_id,
            )

    async def process_file(
        self, file_path: Path, output_path: Path, fix_indentation: bool = False
    ) -> bool:
        """
        Process a single file and generate documentation.

        Args:
            file_path: Path to the source file.
            output_path: Path to store the generated documentation.
            fix_indentation: Whether to auto-fix indentation before processing.

        Returns:
            True if the file was successfully processed, False otherwise.
        """
        try:
            print_section_break()

            print_phase_header(f"📄 Processing File: {file_path}")

            # Validate file type
            if file_path.suffix != ".py":
                print_info(f"⏩ Skipping non-Python file: {file_path}")
                return False

            # Read source code
            source_code = await self.read_file_safe_async(file_path)
            if not source_code or not source_code.strip():
                print_info(
                    f"⚠️ Skipping empty or invalid source file: {file_path}"
                )
                return False

            # Optionally fix indentation
            if fix_indentation:
                print_info(f"🧹 Fixing indentation for: {file_path}")
                source_code = self._fix_indentation(source_code)

            # Validate syntax
            if not self.analyze_syntax(source_code, file_path):
                print_info(f"❌ Skipping file with syntax errors: {file_path}")
                return False

            # Generate documentation
            print_status(f"✍️ Generating documentation for: {file_path}")
            await self.doc_orchestrator.generate_module_documentation(
                file_path, output_path, source_code
            )
            print_success(f"✅ Successfully processed file: {file_path}")
            return True
        except DocumentationError as e:
            log_and_raise_error(
                self.logger,
                e,
                DocumentationError,
                "Error processing file",
                self.correlation_id,
            )
            return False
        finally:
            print_section_break()

    def _fix_indentation(self, source_code: str) -> str:
        """Fix inconsistent indentation using autopep8."""
        try:
            return autopep8.fix_code(source_code)
        except Exception as e:
            print_info(f"Error fixing indentation with autopep8: {e}")
            print_info("Skipping indentation fix.")
            return source_code

    def analyze_syntax(self, source_code: str, file_path: Path) -> bool:
        """Analyze the syntax of the given source code."""
        try:
            ast.parse(source_code)
            return True
        except SyntaxError as e:
            log_and_raise_error(
                self.logger,
                e,
                DocumentationError,
                f"Syntax error in {file_path}",
                self.correlation_id,
            )
            return False

    async def process_repository(
        self,
        repo_path: str,
        output_dir: Path = Path("docs"),
        fix_indentation: bool = False,
    ) -> bool:
        """Process a repository for documentation."""
        start_time = asyncio.get_running_loop().time()
        success = False
        local_path: Path | None = None
        total_files = 0
        processed_files = 0
        skipped_files = 0

        try:
            print_section_break()
            print_info(
                "🚀 Starting Documentation Generation for Repository: "
                f"{repo_path} 🚀"
            )
            print_info(f"Output Directory: {output_dir}")
            print_section_break()

            repo_path = repo_path.strip()

            if self._is_url(repo_path):
                print_info(f"Cloning repository from: {repo_path}")
                local_path = await self._clone_repository(repo_path)
                if not local_path:
                    print_error(f"Failed to clone repository: {repo_path}")
                    return False
                print_success(
                    f"Repository successfully cloned to: {local_path}"
                )
            else:
                local_path = Path(repo_path)

            if not local_path.exists():
                print_error(f"Repository path not found: {local_path}")
                return False

            # base_path is now set to the directory where the repo was cloned
            base_path = local_path

            self.doc_orchestrator.code_extractor.context.base_path = base_path
            self.repo_manager = RepositoryManager(base_path)

            python_files = [
                file
                for file in base_path.rglob("*.py")
                if file.suffix == ".py"
            ]
            total_files = len(python_files)

            print_status(
                "Preparing for Documentation Generation",
                {"Files Found": total_files},
            )

            print_section_break()
            print_info("🔨 Starting Documentation of Python Files 🔨")
            print_section_break()

            progress_ = Progress()
            try:
                with progress_:
                    task = progress_.add_task(
                        "Processing Files", total=total_files
                    )
                    for i, file_path in enumerate(python_files, 1):
                        # Use output_dir to create an output path that mirrors
                        # the structure of the cloned repo
                        relative_path = file_path.relative_to(base_path)
                        output_file = output_dir / relative_path.with_suffix(
                            ".md"
                        )

                        # Ensure the output directory for this file exists
                        output_file.parent.mkdir(parents=True, exist_ok=True)

                        source_code = await self.read_file_safe_async(
                            file_path
                        )
                        if source_code and not source_code.isspace():
                            print_status(
                                "Processing file ({i}/{total_files}): "
                                f"{file_path.name}"
                            )
                            if await self.process_file(
                                file_path,
                                output_file.parent,
                                fix_indentation,
                            ):
                                processed_files += 1
                                progress_.update(
                                    task, advance=1
                                )  # Update progress after successful processing
                            else:
                                skipped_files += 1
                                progress_.update(
                                    task, advance=1
                                )  # Update progress even if skipped
            except KeyboardInterrupt:
                print_error(
                    "🔥 Operation interrupted during file processing."
                )
                raise
            finally:
                progress_.stop()

            print_section_break()
            print_section_break()
            print_info("📊 Repository Processing Summary 📊")

            metrics = self.metrics_collector.get_metrics()

            # Consolidate metrics display
            print_info("Aggregated Metrics Summary:")
            if isinstance(metrics, dict):
                display_metrics(
                    {
                        "Total Files": total_files,
                        "Successfully Processed": processed_files,
                        "Skipped Files": skipped_files,
                        "Total Lines of Code": metrics.get(
                            "total_lines_of_code", 0
                        ),
                        "Maintainability Index": metrics.get(
                            "maintainability_index", 0
                        ),
                        "Total Classes": len(
                            metrics.get("current_metrics", {}).get(
                                "classes", []
                            )
                        ),
                        "Total Functions": len(
                            metrics.get("current_metrics", {}).get(
                                "functions", []
                            )
                        ),
                        "Average Cyclomatic Complexity": metrics.get(
                            "current_metrics", {}
                        ).get("cyclomatic_complexity", 0),
                        "Average Maintainability Index": metrics.get(
                            "current_metrics", {}
                        ).get("maintainability_index", 0.0),
                    }
                )
            else:
                print_info("No metrics available to display.")
            print_section_break()

            success = True

        except (FileNotFoundError, ValueError, IOError) as repo_error:
            log_and_raise_error(
                self.logger,
                repo_error,
                DocumentationError,
                f"Error processing repository {repo_path}",
                self.correlation_id,
            )
            if hasattr(self, "metrics_collector") and self.metrics_collector:
                await self.metrics_collector.close()
            if hasattr(self, "system_monitor") and self.system_monitor:
                await self.system_monitor.stop()
        except Exception as cleanup_error:
            print_error(
                f"Error during cleanup after interruption: {cleanup_error}"
            )
        finally:
            if (
                "progress_" in locals()
            ):  # Ensure progress exists before stopping
                progress_.stop()  # Stop the progress bar explicitly
            processing_time = asyncio.get_running_loop().time() - start_time
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
            print_section_break()
            if success:
                print_success(
                    "✅ Successfully Generated Documentation for Repository: "
                    f"{repo_path} ✅"
                )
            else:
                print_error(
                    "❌ Documentation Generation Failed for Repository: "
                    f"{repo_path} ❌"
                )
            print_info(
                "Processed {processed_files} files, skipped {skipped_files} "
                f"files out of {total_files} total files."
            )
            print(f"Total processing time: {processing_time:.2f} seconds")
            print_section_break()

        return success

    def _is_url(self, path: str) -> bool:
        """Check if the given path is a URL."""
        return path.startswith(("http://", "https://", "git@", "ssh://", "ftp://"))

    async def _clone_repository(self, repo_url: str) -> Path | None:
        """Clone a repository and return its local path."""
        try:
            print_info(
                "Cloning repository: {repo_url} with correlation ID: "
                f"{self.correlation_id}"
            )
            local_path = (
                Path(".")
                / "docs"
                / repo_url.split("/")[-1].replace(".git", "")
            )
            if local_path.exists():
                print_info(f"Repository already exists at {local_path}")
                return local_path

            process = await asyncio.create_subprocess_exec(
                "git",
                "clone",
                repo_url,
                str(local_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()

            if process.returncode != 0:
                print_error(
                    "Error cloning repository {repo_url}: "
                    f"{stderr.decode().strip()}"
                )
                return None

            return local_path
        except Exception as e:
            print_error(f"Error cloning repository {repo_url}: {e}")
            return None  # Ensure None is returned on error

    async def _cleanup_resources(self) -> None:
        """Cleanup resources used by the DocumentationGenerator."""
        try:
            print_info(
                "Starting cleanup process with correlation ID: "
                f"{self.correlation_id}"
            )
            if hasattr(self, "ai_service") and self.ai_service:
                try:
                    if (
                        hasattr(self.ai_service, "client_session")
                        and self.ai_service.client_session
                    ):
                        if not self.ai_service.client_session.closed:
                            await self.ai_service.client_session.close()
                    await self.ai_service.close()
                except Exception as e:
                    print_error(f"Error closing AI service: {e}")
            if hasattr(self, "metrics_collector") and self.metrics_collector:
                await self.metrics_collector.close()
            if hasattr(self, "system_monitor") and self.system_monitor:
                await self.system_monitor.stop()
            print_info(
                "Cleanup completed successfully with correlation ID: "
                f"{self.correlation_id}"
            )
        except (RuntimeError, ValueError, IOError) as cleanup_error:
            print_error(f"Error during cleanup: {cleanup_error}")

    async def cleanup(self) -> None:
        """Call the cleanup resources function."""
        await self._cleanup_resources()

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
    return parser.parse_args()

async def main(args: argparse.Namespace) -> int:
    """Main entry point for the documentation generator."""
    doc_generator: DocumentationGenerator | None = None
    total_files = 0
    processed_files = 0
    skipped_files = 0
    processing_time = 0.0
    try:
        correlation_id = str(uuid.uuid4())
        config = Config()
        print_section_break()
        print_phase_header("Initialization")
        print_info("Initializing system components...")
        print_info("Configuration Summary:")
        print(f"  AI Model: {config.ai.model}")
        print(f"  Deployment: {config.ai.deployment}")
        print(f"  Max Tokens: {config.ai.max_tokens}")
        print(f"  Temperature: {config.ai.temperature}")
        print(f"  Output Directory: {args.output}")
        print_section_break()
        await setup_dependencies(config, correlation_id)

        doc_generator = DocumentationGenerator(
            config=Injector.get("config")
        )
        await doc_generator.initialize()

        if args.repository:
            print_info(f"Processing repository: {args.repository}")
            try:
                success = await doc_generator.process_repository(
                    args.repository,
                    Path(args.output),
                    args.fix_indentation,
                )
            except KeyboardInterrupt:
                logger.info(
                    "Keyboard interrupt detected. Shutting down gracefully..."
                )
                return 130
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred: {e}", exc_info=True
                )
                return 1
        metrics = doc_generator.metrics_collector.get_metrics()
        processed_files = len(
            [
                op
                for op in metrics.get("operations", [])
                if op.get("operation_type") == "file_processing"
                and op.get("success")
            ]
        )
        print_success(
            "Repository documentation generated successfully: "
            f"{success}"
        )
        print_info(f"Processed {processed_files} files.")

        if args.files:
            for file_path in args.files:
                output_path = Path(args.output) / (
                    Path(file_path).stem + ".md"
                )
                success = await doc_generator.process_file(
                    Path(file_path), output_path, args.fix_indentation
                )
                print_success(
                    "Documentation generated successfully for "
                    f"{file_path}: {success}"
                )
    except ConfigurationError as e:
        log_and_raise_error(
            logger, e, ConfigurationError, "Configuration error"
        )
        return 1
    except (RuntimeError, ValueError, IOError) as unexpected_error:
        log_and_raise_error(
            logger,
            unexpected_error,
            Exception,
            "Unexpected error occurred",
            details={
                "Suggested Fix": "Check the logs for more details or "
                "retry the operation."
            },
        )
        return 1
    except KeyError as ke:
        log_and_raise_error(
            logger,
            ke,
            Exception,
            "Dependency injection error",
        )
        return 1
    except KeyboardInterrupt:
        # Gracefully handle user interruptions
        print_error(
            "🔥 Operation Interrupted: The script was stopped by the user."
        )
        try:
            if doc_generator:
                await doc_generator.cleanup()  # Ensure cleanup is awaited
        except Exception as cleanup_error:
            print_error(
                f"Error during cleanup after interruption: {cleanup_error}"
            )
        finally:
            print_success("✅ Cleanup completed. Exiting.")
        return 130  # Standard exit code for terminated by Ctrl+C

    finally:
        try:
            if doc_generator:
                print_info("Info: Starting cleanup process...")
                await doc_generator.cleanup()
        except Exception as cleanup_error:
            print_error(f"Error during cleanup: {cleanup_error}")

        print_section_break()

        # Display final token usage summary and other metrics only after
        # initialization and processing
        if doc_generator and doc_generator.metrics_collector:
            metrics = doc_generator.metrics_collector.get_metrics()
            total_files_processed = len(metrics.get("history", {}))

            # Initialize counters
            total_classes_extracted = 0
            total_functions_extracted = 0 
            total_variables_extracted = 0
            total_constants_extracted = 0
            total_cyclomatic_complexity = 0
            maintainability_sum = 0.0
            maintainability_count = 0
            total_lines_of_code = 0

            # Process metrics history
            history = metrics.get("history", {})
            if history:
                for module_metrics in history.values():
                    if isinstance(module_metrics, list):
                        for entry in module_metrics:
                            if isinstance(entry, dict):
                                # Accumulate metrics
                                total_classes_extracted += entry.get("total_classes", 0)
                                total_functions_extracted += entry.get("total_functions", 0)
                                total_variables_extracted += len(entry.get("variables", []))
                                total_constants_extracted += len(entry.get("constants", []))
                                total_cyclomatic_complexity += entry.get("cyclomatic_complexity", 0)
                                
                                # Handle maintainability index
                                if "maintainability_index" in entry:
                                    maintainability_sum += entry["maintainability_index"]
                                    maintainability_count += 1
                                
                                # Get lines of code from metrics dictionary
                                if "metrics" in entry:
                                    total_lines_of_code += entry["metrics"].get("lines_of_code", 0)

            # Calculate averages
            average_cyclomatic_complexity = (
                total_cyclomatic_complexity / total_functions_extracted 
                if total_functions_extracted > 0 
                else 0.0
            )
            
            average_maintainability_index = (
                maintainability_sum / maintainability_count 
                if maintainability_count > 0 
                else 0.0
            )

            # Display aggregated metrics
            aggregated_metrics = {
                "Total Files Processed": total_files_processed,
                "Total Classes Extracted": total_classes_extracted,
                "Total Functions Extracted": total_functions_extracted,
                "Total Variables Extracted": total_variables_extracted,
                "Total Constants Extracted": total_constants_extracted,
                "Average Cyclomatic Complexity": average_cyclomatic_complexity,
                "Average Maintainability Index": average_maintainability_index,
                "Total Lines of Code": total_lines_of_code,
            }
            display_metrics(aggregated_metrics, title="Aggregated Statistics")

            # Display token usage metrics
            token_metrics = doc_generator.metrics_collector.get_aggregated_token_usage()
            print_section_break()
            print_info("📊 Token Usage Summary 📊")
            display_metrics({
                "Total Prompt Tokens": token_metrics.get("total_prompt_tokens", 0),
                "Total Completion Tokens": token_metrics.get("total_completion_tokens", 0),
                "Total Tokens": token_metrics.get("total_tokens", 0),
                "Estimated Cost": f"${token_metrics.get('total_cost', 0):.2f}",
            })
            print_section_break()

            total_lines_of_code = (
                sum(
                    entry.get("lines_of_code", 0)
                    for module_metrics in metrics.get("history", {}).values()
                    for entry in module_metrics if isinstance(entry, dict)
                )
                if metrics.get("history")
                else 0
            )

            aggregated_metrics = {
                "Total Files Processed": total_files_processed,
                "Total Classes Extracted": total_classes_extracted,
                "Total Functions Extracted": total_functions_extracted,
                "Total Variables Extracted": total_variables_extracted,
                "Total Constants Extracted": total_constants_extracted,
                "Average Cyclomatic Complexity": average_cyclomatic_complexity,
                "Average Maintainability Index": average_maintainability_index,
                "Total Lines of Code": total_lines_of_code,
            }
            display_metrics(
                aggregated_metrics, title="Aggregated Statistics"
            )

            token_metrics = (
                doc_generator.metrics_collector.get_aggregated_token_usage()
            )
            print_section_break()
            print_info("📊 Token Usage Summary 📊")
            display_metrics(
                {
                    "Total Prompt Tokens": token_metrics.get(
                        "total_prompt_tokens", 0
                    ),
                    "Total Completion Tokens": token_metrics.get(
                        "total_completion_tokens", 0
                    ),
                    "Total Tokens": token_metrics.get("total_tokens", 0),
                    "Estimated Cost": f"${token_metrics.get('total_cost', 0):.2f}",
                }
            )
            print_section_break()
        print_section_break()
        print_info("📊 Final Summary 📊")
        print_status(
            "Repository Processing Summary",
            {
                "Total Files": total_files,
                "Successfully Processed": processed_files,
                "Skipped Files": skipped_files,
                "Total Processing Time (seconds)": f"{processing_time:.2f}",
            },
        )
        print_section_break()
        # Add a concise high-level summary at the end
        print_section_break()
        print_info("Exiting documentation generation")
        print_section_break()
        print_info("📊 Final Summary:")
        print_status(
            "Repository Processing Summary",
            {
                "Total Files": total_files,
                "Successfully Processed": processed_files,
                "Skipped": skipped_files,
                "Total Processing Time (seconds)": f"{processing_time:.2f}",
            },
        )
        print_section_break()

    return 0

if __name__ == "__main__":
    cli_args = parse_arguments()
    # Initialize configuration
    config = Config()

    # Add verbosity level for detailed logs
    if config.app.verbose:
        print_debug(f"Command-line arguments: {cli_args}")
    try:
        EXIT_CODE = asyncio.run(main(cli_args))
    except KeyboardInterrupt:
        # Gracefully handle user interruptions
        print_error(
            "🔥 Operation Interrupted: The script was stopped by the user."
        )
        EXIT_CODE = 130  # Standard exit code for terminated by Ctrl+C
    sys.exit(EXIT_CODE)
