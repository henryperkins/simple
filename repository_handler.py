"""
Repository handling module for cloning and managing git repositories.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple
from urllib.parse import urlparse
import git
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class RepositoryHandler:
    """Handles git repository operations and file management."""

    def __init__(self):
        """Initialize the repository handler."""
        self.temp_dir: Optional[str] = None
        self.repo_path: Optional[Path] = None

    def clone_repository(self, repo_url: str) -> Path:
        """
        Clone a git repository to a temporary directory.

        Args:
            repo_url: URL of the git repository to clone

        Returns:
            Path: Path to the cloned repository

        Raises:
            ValueError: If the URL is invalid
            git.GitCommandError: If cloning fails
        """
        # Validate URL
        if not self._is_valid_git_url(repo_url):
            raise ValueError(f"Invalid git repository URL: {repo_url}")

        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp()
            self.repo_path = Path(self.temp_dir)

            # Clone repository
            logger.info(f"Cloning repository from {repo_url}")
            git.Repo.clone_from(repo_url, self.temp_dir)
            logger.info(f"Repository cloned to {self.temp_dir}")

            return self.repo_path

        except git.GitCommandError as e:
            logger.error(f"Failed to clone repository: {e}")
            self.cleanup()
            raise

    def get_python_files(self) -> List[Path]:
        """
        Get all Python files from the repository.

        Returns:
            List[Path]: List of paths to Python files

        Raises:
            ValueError: If repository path is not set
        """
        if not self.repo_path:
            raise ValueError("Repository path not set")

        python_files = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)

        logger.info(f"Found {len(python_files)} Python files")
        return python_files

    def get_file_content(self, file_path: Path) -> Tuple[str, str]:
        """
        Get the content of a file and its relative path.

        Args:
            file_path: Path to the file

        Returns:
            Tuple[str, str]: File content and relative path

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        relative_path = str(file_path.relative_to(self.repo_path))
        return content, relative_path

    def cleanup(self) -> None:
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up directory: {e}")

    def _is_valid_git_url(self, url: str) -> bool:
        """
        Validate if the URL is a valid git repository URL.

        Args:
            url: URL to validate

        Returns:
            bool: True if URL is valid, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    async def __aenter__(self) -> 'RepositoryHandler':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self.cleanup()