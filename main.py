import os
import asyncio
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import argparse
import ast
import git
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm  # Add tqdm for progress bar
import logging  # Add logging

load_dotenv()

from core.logger import LoggerSetup
from core.cache import Cache
from core.monitoring import SystemMonitor
from core.metrics_collector import MetricsCollector
from core.config import AzureOpenAIConfig
from api.token_management import TokenManager
from ai_interaction import AIInteractionHandler
from core.docs import DocumentationContext, DocStringManager
from core.markdown_generator import MarkdownGenerator, MarkdownConfig
from repository_handler import RepositoryHandler
from exceptions import (
    ConfigurationError,
    ProcessingError,
    ValidationError
)

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO
logger = LoggerSetup.get_logger(__name__)


class DocumentationGenerator:
    def __init__(self):
        """Initialize the documentation generator."""
        self.cache: Optional[Cache] = None
        self.metrics: Optional[MetricsCollector] = None
        self.token_manager: Optional[TokenManager] = None
        self.ai_handler: Optional[AIInteractionHandler] = None
        self.repo_handler: Optional[RepositoryHandler] = None
        self.system_monitor: Optional[SystemMonitor] = None  # Add system monitor
        self.config = AzureOpenAIConfig.from_env()

    async def initialize(self) -> None:
        """Initialize all components."""
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
                metrics_collector=self.metrics  # Pass metrics collector
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

            self.system_monitor = SystemMonitor(token_manager=self.token_manager)  # Initialize system monitor
            await self.system_monitor.start()  # Start the system monitor
            logger.info("System monitor started")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise ConfigurationError(f"Failed to initialize components: {str(e)}") from e

    async def cleanup(self) -> None:
        """Clean up and close all resources."""
        try:
            if self.system_monitor:  # Stop system monitor first
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
        """Process a single file and generate documentation."""
        try:
            source_code = file_path.read_text(encoding='utf-8')
            if not source_code.strip():
                logger.error("Empty source code provided")
                return None

            result = await self.ai_handler.process_code(
                source_code,
                cache_key=f"doc:{file_path.stem}:{hash(file_path.read_bytes())}"
            )

            if not result:
                logger.error(f"AI documentation generation failed for {file_path}")
                return None

            updated_code, ai_docs = result

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

            doc_manager = DocStringManager(context)
            documentation = await doc_manager.generate_documentation()

            docs_dir = Path("docs")
            docs_dir.mkdir(parents=True, exist_ok=True)

            relative_path = file_path.relative_to(file_path.parent)
            output_dir = docs_dir / relative_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            doc_path = output_dir / f"{file_path.stem}.md"
            doc_path.write_text(documentation, encoding='utf-8')

            logger.info(f"Documentation saved to: {doc_path}")
            return updated_code, documentation

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return None

    async def _get_recent_changes(self, file_path: Path) -> List[str]:
        """Get recent changes for a file from git history."""
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
        """Process multiple Python files for documentation generation."""
        for file_path in tqdm(file_paths, desc="Processing files", unit="file"):
            try:
                result = await self.process_file(file_path)
                if result:
                    updated_code, documentation = result
                    file_path.write_text(updated_code, encoding='utf-8')
                else:
                    logger.warning(f"No results generated for {file_path}")

            except ValidationError as ve:
                logger.error(f"Validation error for file {file_path}: {str(ve)}")
            except ProcessingError as pe:
                logger.error(f"Processing error for file {file_path}: {str(pe)}")
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {str(e)}")


async def process_repository(args: argparse.Namespace) -> int:
    """Process repository for documentation generation."""
    generator = DocumentationGenerator()

    try:
        await generator.initialize()

        if generator.repo_handler:
            repo_path = generator.repo_handler.clone_repository(args.repository)
            logger.info(f"Repository cloned to: {repo_path}")

            python_files = generator.repo_handler.get_python_files()
            if not python_files:
                logger.warning("No Python files found in repository")
                return 0

            await generator.process_files([Path(f) for f in python_files])
            logger.info("Documentation generation completed successfully")
            return 0
        else:
            logger.error("Repository handler not initialized")
            return 1

    except ConfigurationError as ce:
        logger.error(f"Configuration error: {str(ce)}")
        return 1
    except Exception as e:
        logger.error(f"Documentation generation failed: {str(e)}")
        return 1
    finally:
        await generator.cleanup()


async def main(args: argparse.Namespace) -> int:
    """Main application entry point for processing local files."""
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
    """Parse command line arguments."""
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