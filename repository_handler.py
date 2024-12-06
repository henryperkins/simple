import os
import sys
import shutil
import tempfile
import asyncio
import stat
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
        """
        Clone a git repository to a temporary directory.

        Args:
            repo_url (str): URL of the git repository to clone.

        Returns:
            Path: Path to the cloned repository.

        Raises:
            GitCommandError: If cloning the repository fails.
            ValueError: If the repository URL is invalid.
            Exception: If an error occurs during cloning.
        """
        async with self._lock:
            try:
                # Validate URL
                if not self.validate_repository(repo_url):
                    raise ValueError(f"Invalid git repository URL: {repo_url}")

                # Clone repository
                logger.info(f"Cloning repository from {repo_url}")
                self.repo = git.Repo.clone_from(
                    repo_url,
                    str(self.repo_path),  # Convert Path to string for git operations
                    progress=self._get_git_progress(),
                )

                logger.info(f"Repository cloned to {self.repo_path}")
                return self.repo_path

            except GitCommandError as e:
                error_msg = f"Failed to clone repository: {e}"
                logger.error(error_msg)
                await self.cleanup()
                raise GitCommandError(e.command, e.status, e.stderr)
            except Exception as e:
                error_msg = f"Error cloning repository: {e}"
                logger.error(error_msg)
                await self.cleanup()
                raise

    def get_python_files(
        self, exclude_patterns: Optional[Set[str]] = None
    ) -> List[Path]:
        """
        Get all Python files from the repository.

        Args:
            exclude_patterns (Optional[Set[str]]): Set of patterns to exclude from the search.

        Returns:
            List[Path]: List of Python file paths.

        Raises:
            ValueError: If the repository path is not set.
            Exception: If an error occurs while finding Python files.
        """
        if not self.repo_path:
            raise ValueError("Repository path not set")

        exclude_patterns = exclude_patterns or {
            "*/venv/*",
            "*/env/*",
            "*/build/*",
            "*/dist/*",
            "*/.git/*",
            "*/__pycache__/*",
            "*/migrations/*",
        }

        try:
            python_files = FileUtils.filter_files(
                self.repo_path, pattern="*.py", exclude_patterns=exclude_patterns
            )
            logger.info(f"Found {len(python_files)} Python files")
            return python_files
        except Exception as e:
            logger.error(f"Error finding Python files: {e}")
            return []

    def get_file_content(self, file_path: Path) -> Tuple[str, str]:
        """
        Get the content of a file and its relative path.

        Args:
            file_path (Path): Path to the file.

        Returns:
            Tuple[str, str]: Tuple containing the file content and its relative path.

        Raises:
            FileNotFoundError: If the file is not found.
            Exception: If an error occurs while reading the file.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            content = FileUtils.read_file_safe(file_path)
            relative_path = str(file_path.relative_to(self.repo_path))
            return content, relative_path
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return "", str(file_path)

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

    def validate_repository(self, path: str) -> bool:
        """
        Validate if the given path is a valid git repository URL.

        Args:
            path (str): URL or path to the git repository.

        Returns:
            bool: True if the path is a valid git repository, False otherwise.

        Raises:
            Exception: If an error occurs during validation.
        """
        try:
            return GitUtils.is_valid_git_url(str(path))
        except Exception as e:
            self.logger.error("Error validating git URL: %s", str(e))
            return False

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
