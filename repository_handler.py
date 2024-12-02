"""
Repository handling module for cloning and managing git repositories.

This module provides functionality to clone, manage, and interact with git repositories.
It includes methods for cloning repositories, retrieving Python files, getting file content,
cleaning up resources, and more.

Usage Example:
    ```python
    from repository_handler import RepositoryHandler
    from pathlib import Path

    async def main():
        repo_path = Path("/path/to/repo")
        handler = RepositoryHandler(repo_path)
        await handler.clone_repository("https://github.com/user/repo.git")
        python_files = handler.get_python_files()
        for file in python_files:
            content, relative_path = handler.get_file_content(file)
            print(f"File: {relative_path}\nContent:\n{content}")

    import asyncio
    asyncio.run(main())
    ```

Key Classes and Functions:
- RepositoryHandler: Main class for handling repository operations.
- clone_repository: Clone a git repository to a temporary directory.
- get_python_files: Get all Python files from the repository.
- get_file_content: Get the content of a file and its relative path.
- cleanup: Clean up resources.
- get_file_history: Get git history for a specific file.
- validate_repository: Validate if the given path is a valid git repository.
"""

import os
import sys
import shutil
import tempfile
import asyncio
import stat
from pathlib import Path
from typing import Optional, List, Tuple, Set, Any, Dict
from urllib.parse import urlparse
import git
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError
from core.logger import LoggerSetup

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

    def __init__(self, repo_path: Path, repo_handler: Optional[Any] = None, ai_handler: Optional[Any] = None,
                 token_manager: Optional[Any] = None, metrics: Optional[Any] = None,
                 cache: Optional[Any] = None, system_monitor: Optional[Any] = None):
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
        """
        Asynchronous context manager exit with proper argument handling.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.

        Raises:
            Exception: If an error occurs during cleanup.
        """
        self.logger.info("Exiting RepositoryHandler context")
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir:
                await self._cleanup_git_directory(Path(self.temp_dir))
            await self.cleanup()
        except Exception as e:
            self.logger.error(f"Error during repository cleanup: {e}")
            raise

    def initialize(self) -> None:
        """
        Initialize the repository handler.

        Raises:
            InvalidGitRepositoryError: If the repository path is not a valid git repository.
            Exception: If initialization fails.
        """
        try:
            self.repo = git.Repo(self.repo_path)
            self.logger.info("Repository initialized at %s", self.repo_path)
        except git.exc.InvalidGitRepositoryError:
            self.logger.error("Invalid Git repository at %s", self.repo_path)
            raise
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
                if not self._is_valid_git_url(repo_url):
                    raise ValueError(f"Invalid git repository URL: {repo_url}")

                # Create temporary directory as string
                self.temp_dir = tempfile.mkdtemp()
                logger.info(f"Created temporary directory: {self.temp_dir}")

                # Clone repository
                logger.info(f"Cloning repository from {repo_url}")
                self.repo = git.Repo.clone_from(
                    repo_url, 
                    self.temp_dir,
                    progress=self._get_git_progress()
                )
                
                # Convert to Path for return
                self.repo_path = Path(self.temp_dir)
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

    def get_python_files(self, exclude_patterns: Optional[Set[str]] = None) -> List[Path]:
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
            '*/venv/*', '*/env/*', '*/build/*', '*/dist/*',
            '*/.git/*', '*/__pycache__/*', '*/migrations/*'
        }

        python_files = []
        try:
            for path in self.repo_path.rglob('*.py'):
                # Convert to relative path for pattern matching
                try:
                    rel_path = path.relative_to(self.repo_path)
                    
                    # Check if path matches any exclude pattern
                    if not any(rel_path.match(pattern) for pattern in exclude_patterns):
                        python_files.append(path)
                except Exception as e:
                    logger.error(f"Error processing path {path}: {e}")
                    continue

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
            # Try UTF-8 first
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to latin-1
            logger.warning(f"Failed UTF-8 decode for {file_path}, trying latin-1")
            content = file_path.read_text(encoding='latin-1')

        try:
            relative_path = str(file_path.relative_to(self.repo_path))
            return content, relative_path
        except ValueError as e:
            logger.error(f"Error calculating relative path: {e}")
            return content, str(file_path)

    async def cleanup(self) -> None:
        """
        Clean up resources.

        Raises:
            Exception: If an error occurs during cleanup.
        """
        try:
            if self.system_monitor:
                await self.system_monitor.stop()
                self.logger.info("System monitor stopped")

            if self.repo_handler:
                await self.repo_handler.__aexit__(None, None, None)
                self.logger.info("Repository handler cleaned up")

            if self.ai_handler:
                await self.ai_handler.close()
                self.logger.info("AI handler cleaned up")

            if self.token_manager:
                await self.token_manager.close()
                self.logger.info("Token manager cleaned up")

            if self.metrics:
                await self.metrics.close()
                self.logger.info("Metrics collector cleaned up")

            if self.cache:
                await self.cache.close()
                self.logger.info("Cache cleaned up")

        except Exception as e:
            self.logger.error("Cleanup failed: %s", str(e))

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
                history.append({
                    'hash': commit.hexsha[:8],
                    'author': commit.author.name,
                    'date': commit.committed_datetime.isoformat(),
                    'message': commit.message.strip(),
                })
            return history
        except Exception as e:
            self.logger.error("Error getting file history: %s", str(e))
            return []

    def validate_repository(self, path: Path) -> bool:
        """
        Validate if the given path is a valid git repository.

        Args:
            path (Path): Path to the repository.

        Returns:
            bool: True if the path is a valid git repository, False otherwise.

        Raises:
            Exception: If an error occurs during validation.
        """
        try:
            _ = git.Repo(path).git_dir
            return True
        except git.exc.InvalidGitRepositoryError:
            return False
        except Exception as e:
            self.logger.error("Error validating git URL: %s", str(e))
            return False

    def _get_git_progress(self) -> git.RemoteProgress:
        """
        Create a progress handler for git operations.

        Returns:
            git.RemoteProgress: Progress handler for git operations.
        """
        logger = self.logger  # Capture logger from outer scope
        
        class GitProgress(git.RemoteProgress):
            def __init__(self):
                super().__init__()
                self.logger = logger  # Assign logger to instance

            def update(self, op_code, cur_count, max_count=None, message=''):
                if max_count:
                    percentage = cur_count / max_count * 100
                    self.logger.debug('Git progress: %.1f%% %s', percentage, message or "")
                else:
                    self.logger.debug('Git progress: %s', message or "")

        return GitProgress()

    async def _cleanup_git_directory(self, path: Path) -> None:
        """
        Safely clean up a Git repository directory.

        Args:
            path (Path): Path to the directory to clean up.

        Raises:
            Exception: If an error occurs during cleanup.
        """
        try:
            # Kill any running Git processes on Windows
            if sys.platform == 'win32':
                os.system('taskkill /F /IM git.exe 2>NUL')
            
            await asyncio.sleep(1)
            
            def handle_rm_error(func, path, exc_info):
                """Handle errors during rmtree."""
                try:
                    if path.exists():
                        os.chmod(str(path), stat.S_IWRITE)
                        func(str(path))
                except Exception as e:
                    logger.error(f"Error removing path {path}: {e}")

            # Attempt cleanup with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if path.exists():
                        # Convert Path to string for shutil operations
                        shutil.rmtree(str(path), onerror=handle_rm_error)
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.error(f"Could not remove directory after {max_retries} attempts: {path}")
                except Exception as e:
                    logger.error(f"Error removing directory {path}: {e}")
                    break
                        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _is_valid_git_url(self, url: str) -> bool:
        """
        Validate if the URL is a valid git repository URL.

        Args:
            url (str): URL to validate.

        Returns:
            bool: True if the URL is valid, False otherwise.

        Raises:
            Exception: If an error occurs during validation.
        """
        try:
            result = urlparse(url)
            
            # Check basic URL structure
            if not all([result.scheme, result.netloc]):
                return False

            # Check for valid git URL patterns
            valid_schemes = {'http', 'https', 'git', 'ssh'}
            if result.scheme not in valid_schemes:
                return False

            # Check for common git hosting domains or .git extension
            common_domains = {
                'github.com', 'gitlab.com', 'bitbucket.org',
                'dev.azure.com'
            }
            
            domain = result.netloc.lower()
            if not any(domain.endswith(d) for d in common_domains):
                if not url.endswith('.git'):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating git URL: {e}")
            return False
