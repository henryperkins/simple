"""  
Logging configuration and utilities.  
Provides consistent logging across the application.  
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
from logging.handlers import RotatingFileHandler


class LoggerSetup:
    """Configures and manages application logging."""

    @staticmethod
    def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        """
        Get a configured logger instance.

        Args:
            name: Logger name
            level: Logging level

        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger(name)

        if not logger.handlers:
            logger.setLevel(level)

            # Create formatters
            console_formatter = logging.Formatter("%(levelname)s: %(message)s")
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            # File handler
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            file_handler = RotatingFileHandler(
                log_dir / f"{name}.log", maxBytes=1024 * 1024, backupCount=5  # 1MB
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        return logger

    @staticmethod
    def configure(level: str, format_str: str, log_dir: Optional[str] = None) -> None:
        """
        Configure global logging settings.

        Args:
            level: Logging level
            format_str: Log format string
            log_dir: Optional log directory
        """
        numeric_level = getattr(logging, level.upper(), logging.INFO)

        # Create handlers
        handlers = []

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(format_str))
        handlers.append(console)

        # File handler
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                log_path / f"app_{datetime.now():%Y%m%d}.log",
                maxBytes=1024 * 1024,  # 1MB
                backupCount=5,
            )
            file_handler.setFormatter(logging.Formatter(format_str))
            handlers.append(file_handler)

        # Configure root logger
        logging.basicConfig(level=numeric_level, format=format_str, handlers=handlers)


def log_error(msg: str, *args: Any, **kwargs: Any) -> None:
    """
    Log an error message.

    Args:
        msg: The message to log
        args: Additional positional arguments for the logger
        kwargs: Additional keyword arguments for the logger
    """
    logger = logging.getLogger(__name__)
    logger.error(msg, *args, **kwargs)


def log_debug(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a debug message."""
    logger = logging.getLogger(__name__)
    logger.debug(msg, *args, **kwargs)


def log_info(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log an info message."""
    logger = logging.getLogger(__name__)
    logger.info(msg, *args, **kwargs)


def log_warning(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a warning message."""
    logger = logging.getLogger(__name__)
    logger.warning(msg, *args, **kwargs)
