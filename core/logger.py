"""Enhanced logging configuration with structured output."""

import logging
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler
from rich.console import Console
from contextvars import ContextVar
from core.console import (
    console as rich_console,
)  # Use the default console from console.py

# Context variable for the correlation ID
correlation_id_var = ContextVar("correlation_id", default=None)


def set_correlation_id(correlation_id: str):
    """Set the correlation ID in the context."""
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Retrieve the correlation ID from the context or return None if not set."""
    return correlation_id_var.get()


class CorrelationLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that includes correlation ID in log messages."""

    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})
        # Use the context variable to get the current correlation ID
        self.correlation_id_var = correlation_id_var

    def process(self, msg, kwargs):
        # Ensure 'extra' in kwargs
        kwargs["extra"] = kwargs.get("extra", {})
        # Include correlation ID in 'extra'
        correlation_id = self.correlation_id_var.get()
        kwargs["extra"]["correlation_id"] = correlation_id or "N/A"
        return msg, kwargs


class SanitizedLogFormatter(logging.Formatter):
    """Custom formatter to sanitize and format log records."""

    def format(self, record):
        # Ensure correlation_id and sanitized_info fields exist
        record.correlation_id = get_correlation_id() or "N/A"
        record.sanitized_info = getattr(
            record, "sanitized_info", {"info": "[Sanitized]"}
        )

        # Sanitize the message and arguments
        record.msg = self._sanitize(record.msg)
        if record.args:
            record.args = tuple(self._sanitize(arg) for arg in record.args)

        # Format the timestamp
        if self.usesTime():
            record.asctime = datetime.fromtimestamp(record.created).isoformat() + "Z"

        return super().format(record)

    def _sanitize(self, item: Any) -> Any:
        """Sanitize sensitive information from logs."""
        if isinstance(item, dict):
            return {k: self._sanitize(v) for k, v in item.items()}
        elif isinstance(item, (list, tuple)) and not isinstance(item, str):
            return [self._sanitize(it) for it in item]
        elif isinstance(item, str):
            # Sanitize file paths and secrets
            item = re.sub(r"(/[a-zA-Z0-9_\-./]+)", "[SANITIZED_PATH]", item)
            item = re.sub(
                r"(secret_key|password|token)=[^&\s]+", r"\1=[REDACTED]", item
            )
            return item
        return item


class LoggerSetup:
    """Configures and manages application logging."""

    _loggers: Dict[str, logging.Logger] = {}
    _log_dir: Path = Path("logs")  # Set a default or use a placeholder
    _file_logging_enabled: bool = True
    _configured: bool = False
    _default_level: int = logging.INFO
    _default_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    _max_bytes: int = 10 * 1024 * 1024  # 10MB
    _backup_count: int = 5

    @classmethod
    def configure(
        cls, console_instance: Optional[Console] = None, config: Optional[Any] = None
    ) -> None:
        """Configure global logging settings."""
        if cls._configured:
            return

        # Use 'config' if provided, otherwise use default settings
        if config is not None:
            cls._default_level = getattr(
                logging, config.app.log_level.upper(), logging.INFO
            )
            cls._log_dir = Path(config.app.log_dir)
        else:
            # Use default level and log_dir if not configured
            cls._default_level = logging.INFO
            cls._log_dir = Path("logs")

        cls._configured = True

        # Use the provided console instance or the default one
        console_instance = console_instance or rich_console

        # Configure the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(cls._default_level)

        # Console handler
        console_handler = RichHandler(
            console=console_instance,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
        )
        console_formatter = logging.Formatter(cls._default_format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # File handler (if enabled)
        if cls._file_logging_enabled:
            try:
                cls._log_dir.mkdir(parents=True, exist_ok=True)
                file_handler = RotatingFileHandler(
                    cls._log_dir / "app.log",
                    maxBytes=cls._max_bytes,
                    backupCount=cls._backup_count,
                )
                file_formatter = SanitizedLogFormatter(
                    fmt='{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", '
                    '"message": "%(message)s", "correlation_id": "%(correlation_id)s", '
                    '"sanitized_info": %(sanitized_info)s}',
                    datefmt="%Y-%m-%dT%H:%M:%S",
                )
                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)
            except Exception as e:
                console_instance.print(
                    f"Failed to set up file handler: {e}", style="bold red"
                )

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """Get a configured logger instance."""
        if not cls._configured:
            cls.configure()

        if name is None:
            name = __name__

        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)

        # Ensure the logger has the correct methods
        if not hasattr(logger, "isEnabledFor"):
            logger.isEnabledFor = (
                lambda level: True
            )  # Dummy method to avoid AttributeError

        cls._loggers[name] = logger
        return cls._get_correlation_logger_adapter(logger)

    @classmethod
    def _get_correlation_logger_adapter(cls, logger):
        return CorrelationLoggerAdapter(logger)

    @classmethod
    def shutdown(cls) -> None:
        """Cleanup logging handlers and close files."""
        for logger in cls._loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        cls._loggers.clear()
        logging.shutdown()

    @classmethod
    def handle_exception(
        cls, exc_type: type, exc_value: BaseException, exc_traceback: Any
    ) -> None:
        """Global exception handler."""
        if not issubclass(exc_type, KeyboardInterrupt):
            logger = cls.get_logger("global")
            logger.critical(
                "Unhandled exception",
                exc_info=(exc_type, exc_value, exc_traceback),
                extra={
                    "correlation_id": get_correlation_id() or "N/A",
                    "sanitized_info": {},
                },
            )
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
