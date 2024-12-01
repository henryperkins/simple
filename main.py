import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
import argparse
import ast
from dotenv import load_dotenv
from tqdm import tqdm
import git  # Import git module

from core.logger import LoggerSetup
from core.cache import Cache
from core.monitoring import SystemMonitor
from core.metrics_collector import MetricsCollector
from core.config import AzureOpenAIConfig
from api.token_management import TokenManager
from ai_interaction import AIInteractionHandler
from core.docs import DocumentationContext, DocStringManager
from core.code_extraction import CodeExtractor
from repository_handler import RepositoryHandler
from exceptions import (
    ConfigurationError,
    ExtractionError
)

load_dotenv()

# Configure logging
logger = LoggerSetup.get_logger(__name__)  # Initialize logger

class DocumentationGenerator:
    """
    Documentation Generator Class

    Handles the initialization, processing, and cleanup of components for generating documentation.
    """

    def __init__(self) -> None:
        """
        Initialize the documentation generator.
        """
        self.logger = LoggerSetup.get_logger(__name__)
        self.cache: Optional[Cache] = None
        self.metrics: Optional[MetricsCollector] = None
        self.token_manager: Optional[TokenManager] = None
        self.ai_handler: Optional[AIInteractionHandler] = None
        self.repo_handler: Optional[RepositoryHandler] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.config = AzureOpenAIConfig.from_env()

    async def initialize(self) -> None:
        """
        Initialize all components.

        Raises:
            ConfigurationError: If initialization fails.
        """
        try:
            if self.config.cache_enabled:
                self.cache = Cache(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password,
                    enabled=self.config.cache_enabled
                )
                logger.info("Cache initialized successfully")

            self.metrics = MetricsCollector()
            logger.info("Metrics collector initialized")

            self.token_manager = TokenManager(
                model=self.config.model_name,
                deployment_name=self.config.deployment_name,
                config=self.config,
                metrics_collector=self.metrics
            )
            logger.info("Token manager initialized")

            self.ai_handler = AIInteractionHandler(
                cache=self.cache,
                metrics_collector=self.metrics,
                token_manager=self.token_manager,
                config=self.config
            )
            logger.info("AI interaction handler initialized")

            self.repo_handler = RepositoryHandler()
            logger.info("Repository handler initialized")

            self.system_monitor = SystemMonitor(token_manager=self.token_manager)
            await self.system_monitor.start()
            logger.info("System monitor started")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise ConfigurationError(f"Failed to initialize components: {str(e)}") from e

    async def cleanup(self) -> None:
        """
        Clean up and close all resources.
        """
        try:
            if self.system_monitor:
                await self.system_monitor.stop()
                logger.info("System monitor stopped")

            for component, name in [
                (self.ai_handler, "AI Handler"),
                (self.repo_handler, "Repository Handler"),
                (self.metrics, "Metrics Collector"),
                (self.cache, "Cache"),
                (self.token_manager, "Token Manager")
            ]:
                if component and hasattr(component, 'close'):
                    try:
                        await component.close()
                        logger.info(f"{name} closed successfully")
                    except Exception as e:
                        logger.error(f"Error closing {name}: {e}")

            logger.info("Cleanup completed successfully")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    async def process_file(self, file_path: Path) -> Optional[Tuple[str, str]]:
        """
        Process a single file and generate documentation.

        Args:
            file_path (Path): The path to the file to process.

        Returns:
            Optional[Tuple[str, str]]: Tuple of (updated_code, documentation) or None if processing fails.

        Raises:
            Exception: If processing fails.
        """
        try:
            self.logger.info(f"Processing file: {file_path}")
            source_code = file_path.read_text(encoding='utf-8')
            
            if not source_code.strip():
                self.logger.warning(f"Empty source code in {file_path}")
                return None
    
            # Process with AI handler
            self.logger.debug("Generating AI documentation...")
            result = await self.ai_handler.process_code(
                source_code,
                cache_key=f"doc:{file_path.stem}:{hash(source_code.encode())}"
            )
    
            if result:
                self.logger.debug("AI documentation generated successfully")
                updated_code, ai_docs = result
    
                # Create documentation context
                context = DocumentationContext(
                    source_code=updated_code,
                    module_path=file_path,
                    include_source=True,
                    metadata={
                        "ai_generated": ai_docs,
                        "file_path": str(file_path),
                        "module_name": file_path.stem,
                        "changes": await self._get_recent_changes(file_path)
                    }
                )
    
                # Generate documentation
                self.logger.debug("Generating markdown documentation...")
                doc_manager = DocStringManager(context)
                documentation = await doc_manager.generate_documentation()
    
                # Create output directory
                docs_dir = Path("docs")
                docs_dir.mkdir(parents=True, exist_ok=True)
    
                # Save documentation
                doc_path = docs_dir / f"{file_path.stem}.md"
                doc_path.write_text(documentation, encoding='utf-8')
                self.logger.info(f"Documentation saved to: {doc_path}")
    
                return updated_code, documentation
    
            else:
                self.logger.error(f"AI documentation generation failed for {file_path}")
                return None
    
        except Exception as e:
            self.logger.error(f"Failed to process {file_path}: {str(e)}")
            return None

    async def _get_recent_changes(self, file_path: Path) -> List[str]:
        """
        Get recent changes for a file from git history.

        Args:
            file_path (Path): The path to the file.

        Returns:
            List[str]: List of recent changes.

        Raises:
            Exception: If retrieving changes fails.
        """
        try:
            if not self.repo_handler:
                return ["No change history available - not a git repository"]

            changes = []
            repo = git.Repo(file_path.parent, search_parent_directories=True)
            commits = list(repo.iter_commits(paths=str(file_path), max_count=5))

            for commit in commits:
                commit_date = datetime.fromtimestamp(commit.committed_date)
                date_str = commit_date.strftime("%Y-%m-%d")
                message = commit.message.split('\n')[0][:100]
                changes.append(f"[{date_str}] {message}")

            return changes if changes else ["No recent changes recorded"]

        except Exception as e:
            logger.warning(f"Failed to get change history: {e}")
            return ["Could not retrieve change history"]

    async def process_files(self, file_paths: List[Path]) -> None:
        """
        Process multiple Python files for documentation generation.

        Args:
            file_paths (List[Path]): List of file paths to process.
        """
        stats = {
            'total': len(file_paths),
            'processed': 0,
            'errors': {
                'syntax': [],
                'extraction': [],
                'empty': [],
                'other': []
            }
        }

        for file_path in tqdm(file_paths, desc="Processing files", unit="file"):
            try:
                source_code = file_path.read_text(encoding='utf-8')
                if not source_code.strip():
                    logger.warning(f"Empty source code in {file_path}")
                    stats['errors']['empty'].append((file_path, "Empty file"))
                    continue

                try:
                    result = await self.process_file(file_path)
                    if result:
                        stats['processed'] += 1
                    else:
                        logger.error(f"Processing failed for {file_path}")
                        stats['errors']['other'].append((file_path, "Processing failed"))
                except SyntaxError as e:
                    logger.error(f"Syntax error in {file_path}: {str(e)}")
                    stats['errors']['syntax'].append((file_path, str(e)))
                except ExtractionError as e:
                    logger.error(f"Extraction error in {file_path}: {str(e)}")
                    stats['errors']['extraction'].append((file_path, str(e)))
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    stats['errors']['other'].append((file_path, str(e)))

            except Exception as e:
                logger.error(f"Unexpected error reading {file_path}: {str(e)}")
                stats['errors']['other'].append((file_path, str(e)))

        logger.info("\nProcessing Summary:")
        logger.info(f"Total files: {stats['total']}")
        logger.info(f"Successfully processed: {stats['processed']}")

        if stats['processed'] == 0:
            logger.warning("\nWARNING: No files were successfully processed!")

        if any(stats['errors'].values()):
            logger.info("\nErrors by category:")
            for category, errors in stats['errors'].items():
                if errors:
                    logger.info(f"\n{category.upper()} Errors ({len(errors)}):")
                    for file_path, error in errors:
                        logger.info(f"- {file_path}: {error}")

        if stats['errors']['syntax']:
            logger.info("\nSuggestion: Review syntax errors in the files listed above.")
        if stats['errors']['extraction']:
            logger.info("\nSuggestion: Check the extraction logic for possible issues.")


async def process_repository(args: argparse.Namespace) -> int:
    """
    Process repository for documentation generation.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    generator = DocumentationGenerator()

    try:
        await generator.initialize()

        if not generator.repo_handler:
            logger.error("Repository handler not initialized")
            return 1

        repo_path = generator.repo_handler.clone_repository(args.repository)
        logger.info(f"Repository cloned to: {repo_path}")

        python_files = generator.repo_handler.get_python_files()
        if not python_files:
            logger.warning("No Python files found in repository")
            return 0

        stats = {
            'total': len(python_files),
            'processed': 0,
            'errors': {
                'syntax': [],
                'extraction': [],
                'empty': [],
                'other': []
            }
        }

        for file_path in tqdm(python_files, desc="Processing files"):
            try:
                source_code = file_path.read_text(encoding='utf-8')

                if not source_code.strip():
                    logger.warning(f"Empty source code in {file_path}")
                    stats['errors']['empty'].append((file_path, "Empty file"))
                    continue

                result = await generator.process_file(file_path)

                if result:
                    updated_code, documentation = result

                    # Calculate relative path from repository root
                    relative_path = normalize_path(file_path, repo_path)

                    # Create output directory preserving structure
                    output_dir = Path("docs") / relative_path.parent
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Write documentation to markdown file
                    doc_path = output_dir / f"{file_path.stem}.md"
                    doc_path.write_text(documentation, encoding='utf-8')

                    stats['processed'] += 1
                else:
                    logger.error(f"Failed to process {file_path}")
                    stats['errors']['other'].append((file_path, "Processing failed"))

            except SyntaxError as e:
                logger.error(f"Syntax error in {file_path}: {str(e)}")
                stats['errors']['syntax'].append((file_path, str(e)))
            except ExtractionError as e:
                logger.error(f"Extraction error in {file_path}: {str(e)}")
                stats['errors']['extraction'].append((file_path, str(e)))
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                stats['errors']['other'].append((file_path, str(e)))

        logger.info("\nProcessing Summary:")
        logger.info(f"Total files: {stats['total']}")
        logger.info(f"Successfully processed: {stats['processed']}")

        if stats['processed'] == 0:
            logger.warning("\nWARNING: No files were successfully processed!")

        if any(stats['errors'].values()):
            logger.info("\nErrors by category:")
            for category, errors in stats['errors'].items():
                if errors:
                    logger.info(f"\n{category.upper()} Errors ({len(errors)}):")
                    for file_path, error in errors:
                        logger.info(f"- {file_path}: {error}")

        return 0 if stats['processed'] > 0 else 1

    except Exception as e:
        logger.error(f"Repository processing failed: {str(e)}")
        return 1
    finally:
        await generator.cleanup()

async def main(args: argparse.Namespace) -> int:
    """
    Main application entry point for processing local files.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    generator = DocumentationGenerator()

    try:
        await generator.initialize()

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
        argparse.Namespace: Parsed command line arguments.
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

def normalize_path(file_path: Path, base_path: Optional[Path] = None) -> Path:
    """
    Normalize file path for documentation output.

    Args:
        file_path (Path): The file path to normalize.
        base_path (Optional[Path]): The base path to use for normalization.

    Returns:
        Path: The normalized file path.
    """
    if base_path and file_path.is_absolute():
        try:
            return file_path.relative_to(base_path)
        except ValueError:
            return file_path
    return file_path

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
