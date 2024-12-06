import os
import sys
import shutil
import tempfile
import asyncio
import stat
import re
from pathlib import Path
from typing import Optional, List, Tuple, Set, Any, Dict
import git
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError
from core.logger import LoggerSetup
from core.utils import FileUtils, GitUtils

logger = LoggerSetup.get_logger(__name__)


class RepositoryHandler:
    """
    Handles operations related to git repositories, including cloning, file retrieval,
    and cleanup.

    Attributes:
        repo_path (Path): Path to the repository.
        repo_handler (Optional[Any]): Optional repository handler.
        ai_handler (Optional[Any]): Optional AI handler.
        token_manager (Optional[Any]): Optional token manager.
        metrics (Optional[Any]): Optional metrics collector.
        cache (Optional[Any]): Optional cache.
        system_monitor (Optional[Any]): Optional system monitor.
        logger (Logger): Logger instance for logging.
        repo (Optional[Repo]): Git repository instance.
        _lock (asyncio.Lock): Asynchronous lock for thread safety.
    """

    def __init__(
        self,
        repo_path: Path,
        repo_handler: Optional[Any] = None,
        ai_handler: Optional[Any] = None,
        token_manager: Optional[Any] = None,
        metrics: Optional[Any] = None,
        cache: Optional[Any] = None,
        system_monitor: Optional[Any] = None,
    ):
        """
        Initialize the RepositoryHandler.

        Args:
            repo_path (Path): Path to the repository.
            repo_handler (Optional[Any]): Optional repository handler.
            ai_handler (Optional[Any]): Optional AI handler.
            token_manager (Optional[Any]): Optional token manager.
            metrics (Optional[Any]): Optional metrics collector.
            cache (Optional[Any]): Optional cache.
            system_monitor (Optional[Any]): Optional system monitor.
        """
        self.repo_path = repo_path
        self.repo_handler = repo_handler
        self.ai_handler = ai_handler
        self.token_manager = token_manager
        self.metrics = metrics
        self.cache = cache
        self.system_monitor = system_monitor
        self.logger = LoggerSetup.get_logger(__name__)
        self.repo = None
        self._lock = asyncio.Lock()  # Initialize the lock

    async def __aenter__(self):
        """
        Asynchronous context manager entry.

        Returns:
            RepositoryHandler: The instance of the handler.
        """
        self.logger.info("Entering RepositoryHandler context")
        self.initialize()
        return self

    async def __aexit__(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        """Asynchronous context manager exit with proper handling."""
        self.logger.info("Exiting RepositoryHandler context")
        cleanup_exception = None
        try:
            if hasattr(self, "temp_dir") and self.temp_dir:
                await GitUtils.cleanup_git_directory(Path(self.temp_dir))
            await self.cleanup()
        except Exception as e:
            self.logger.error(f"Error during repository cleanup: {e}")
            cleanup_exception = e

        if exc_type:
            # If there was an exception in the block, we need to re-raise it.
            if cleanup_exception:
                # If there was also a cleanup exception, we can log or handle both
                raise Exception("Exception during execution and cleanup") from exc_val
            else:
                # Re-raise the original exception
                raise exc_val
        elif cleanup_exception:
            # If there was only a cleanup exception, raise it
            raise cleanup_exception

    def initialize(self) -> None:
        """
        Initialize the repository handler.

        Raises:
            InvalidGitRepositoryError: If the repository path is not a valid git repository
            Exception: If initialization fails
        """
        try:
            if Path(self.repo_path / ".git").exists():
                self.repo = git.Repo(self.repo_path)
                self.logger.info("Repository initialized at %s", self.repo_path)
            else:
                # Don't initialize git repo if it doesn't exist yet - will be created during clone
                self.logger.debug(
                    "No git repository found at %s, will be created during clone",
                    self.repo_path,
                )
                self.repo = None

        except Exception as e:
            self.logger.error("Failed to initialize repository: %s", str(e))
            raise

    async def clone_repository(self, repo_url: str) -> Path:
        """Clone a repository from URL."""
        try:
            clone_dir = self.repo_path / Path(repo_url).stem
            if clone_dir.exists():
                self.logger.info(f"Repository exists at: {clone_dir}")
                # Verify repository structure
                if not self._verify_repository(clone_dir):
                    self.logger.warning(f"Invalid repository structure at {clone_dir}, re-cloning")
                    shutil.rmtree(clone_dir)
                else:
                    return clone_dir

            self.logger.info(f"Cloning repository from {repo_url} to {clone_dir}")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, Repo.clone_from, repo_url, clone_dir)
            
            # Verify cloned repository
            if not self._verify_repository(clone_dir):
                raise GitCommandError("clone", "Invalid repository structure after cloning")
                
            self.logger.info(f"Repository cloned successfully to {clone_dir}")
            return clone_dir

        except Exception as e:
            self.logger.error(f"Failed to clone repository: {e}")
            raise

    def _verify_repository(self, path: Path) -> bool:
        """Verify repository structure and content."""
        try:
            # Check if it's a git repository
            if not (path / ".git").exists():
                self.logger.error(f"No .git directory found in {path}")
                return False

            # Log repository structure
            self.logger.debug("Repository structure:")
            for item in path.rglob("*"):
                self.logger.debug(f"  {item}")

            return True
        except Exception as e:
            self.logger.error(f"Error verifying repository: {e}")
            return False

    def get_python_files(self, exclude_patterns: Optional[Set[str]] = None) -> List[Path]:
        """Retrieve all Python files in the repository, excluding specified patterns."""
        python_files = GitUtils.get_python_files(self.repo_path, exclude_patterns)
        self.logger.info(f"Found {len(python_files)} Python files in the repository")
        return python_files

    def get_file_content(self, file_path: Path) -> Tuple[str, str]:
        """Retrieve the content of the specified file."""
        content = FileUtils.read_file_safe(file_path)
        return content, file_path.name

    async def cleanup(self) -> None:
        """
        Clean up resources.

        Raises:
            Exception: If an error occurs during cleanup.
        """
        exceptions = []
        if self.system_monitor:
            try:
                await self.system_monitor.stop()
                self.logger.info("System monitor stopped")
            except Exception as e:
                self.logger.error(f"Error stopping system monitor: {e}")
                exceptions.append(e)

        if self.repo_handler:
            try:
                await self.repo_handler.__aexit__(None, None, None)
                self.logger.info("Repository handler cleaned up")
            except Exception as e:
                self.logger.error(f"Error cleaning up repository handler: {e}")
                exceptions.append(e)

        if self.ai_handler:
            try:
                await self.ai_handler.close()
                self.logger.info("AI handler cleaned up")
            except Exception as e:
                self.logger.error(f"Error closing AI handler: {e}")
                exceptions.append(e)

        if self.token_manager:
            try:
                await self.token_manager.close()
                self.logger.info("Token manager cleaned up")
            except Exception as e:
                self.logger.error(f"Error closing token manager: {e}")
                exceptions.append(e)

        if self.metrics:
            try:
                await self.metrics.close()
                self.logger.info("Metrics collector cleaned up")
            except Exception as e:
                self.logger.error(f"Error closing metrics collector: {e}")
                exceptions.append(e)

        if self.cache:
            try:
                await self.cache.close()
                self.logger.info("Cache cleaned up")
            except Exception as e:
                self.logger.error(f"Error closing cache: {e}")
                exceptions.append(e)

        # Optionally, raise an aggregated exception if needed
        if exceptions:
            raise Exception("One or more errors occurred during cleanup.")

    async def get_file_history(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        This method retrieves the commit history for a specified file in the git repository.
        Each commit record includes the commit hash (first 8 characters), author name,
        commit date in ISO format, and commit message.

            file_path (Path): Path to the file whose history should be retrieved.

            List[Dict[str, Any]]: A list of dictionaries, each containing commit details:
                - hash (str): First 8 characters of commit hash
                - author (str): Name of the commit author
                - date (str): Commit date and time in ISO format
                - message (str): Commit message

            Exception: If an error occurs while accessing git history. In this case,
                      the error is logged and an empty list is returned.

        Example:
            >>> history = await repo_handler.get_file_history(Path('src/main.py'))
            >>> print(history[0])
            {
                'hash': 'a1b2c3d4',
                'author': 'John Doe',
                'date': '2023-01-01T12:00:00+00:00',
                'message': 'Initial commit'
            }
        """
        if not self.repo:
            return []

        try:
            history = []
            # Convert Path to string for git operations
            str_path = str(file_path)
            for commit in self.repo.iter_commits(paths=str_path):
                history.append(
                    {
                        "hash": commit.hexsha[:8],
                        "author": commit.author.name,
                        "date": commit.committed_datetime.isoformat(),
                        "message": commit.message.strip(),
                    }
                )
            return history
        except Exception as e:
            self.logger.error("Error getting file history: %s", str(e))
            return []

    def _get_git_progress(self):
        """
        Create a progress handler for git operations.

        Returns:
            Callable: Progress callback function for git operations.
        """

        def progress_callback(
            op_code: int,
            cur_count: str | float,
            max_count: str | float | None = None,
            message: str = "",
        ):
            if max_count:
                percentage = float(cur_count) / float(max_count) * 100
                self.logger.debug("Git progress: %.1f%% %s", percentage, message or "")
            else:
                self.logger.debug("Git progress: %s", message or "")

        return progress_callback
