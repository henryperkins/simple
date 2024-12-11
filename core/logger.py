"""
Enhanced Logging Configuration and Utilities.
Provides structured, contextual, and robust logging across the application.
"""

import logging
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, MutableMapping, Optional
from logging.handlers import RotatingFileHandler
import uuid
from collections.abc import Mapping, Sequence


class SanitizedLogFormatter(logging.Formatter):
    """Custom formatter to sanitize and format log records."""

    def format(self, record):
        # Ensure 'correlation_id' and 'sanitized_info' fields are present with default values
        record.correlation_id = getattr(record, 'correlation_id', "N/A")
        record.sanitized_info = getattr(
            record, 'sanitized_info', {"info": "[Sanitized]"})

        # Sanitize the message and arguments recursively
        record.msg = self._sanitize(record.msg)
        if record.args:
            record.args = tuple(self._sanitize(arg) for arg in record.args)

        # Format the timestamp in ISO8601 format
        if self.usesTime():
            record.asctime = datetime.fromtimestamp(
                record.created).isoformat() + 'Z'

        # Now format the message using the parent class
        return super().format(record)

    def _sanitize(self, item: Any) -> Any:  # Generalized sanitization
        if isinstance(item, Mapping):
            return {k: self._sanitize(v) for k, v in item.items()}
        elif isinstance(item, Sequence) and not isinstance(item, str):
            return [self._sanitize(it) for it in item]
        elif isinstance(item, str):
            # Example sanitization: Redact file paths and secrets (customize as needed)
            item = re.sub(r'(/[a-zA-Z0-9_\-./]+)',
                          '[SANITIZED_PATH]', item)  # File paths
            item = re.sub(
                # Secrets
                r'(secret_key|password|token)=[^&\s]+', r'\1=[REDACTED]', item)
            return item
        return item


class LoggerSetup:
    """Configures and manages application logging."""

    _loggers: Dict[str, logging.Logger] = {}
    _log_dir: Path = Path("logs")
    _file_logging_enabled: bool = True
    _configured: bool = False
    _default_level: int = logging.INFO
    _default_format: str = "%(levelname)s: %(message)s"
    _max_bytes: int = 10 * 1024 * 1024  # 10MB default
    _backup_count: int = 5

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

        if not logger.hasHandlers():

            # Correlation ID handling: generate or retrieve from logger's existing context
            # retrieve existing context from logger object if it exists
            extra = getattr(logger, '_extra_context', {})

            correlation_id = extra.get('correlation_id')
            if not correlation_id:
                correlation_id = str(uuid.uuid4())
                extra['correlation_id'] = correlation_id
                logger._extra_context = extra  # Attach context to logger

            logger.setLevel(cls._default_level)

            # Console handler (using extra now)
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(cls._default_format)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            # File handler (enhanced error handling and extra usage)
            if cls._file_logging_enabled:
                try:
                    cls._log_dir.mkdir(parents=True, exist_ok=True)
                    file_handler = RotatingFileHandler(
                        cls._log_dir / f"{name}.log",
                        maxBytes=cls._max_bytes,
                        backupCount=cls._backup_count
                    )
                    sanitized_formatter = SanitizedLogFormatter(
                        fmt='{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", '
                            '"message": "%(message)s", "correlation_id": "%(correlation_id)s", '
                            # no quotes around sanitized_info because the .format method handles dicts
                            '"sanitized_info": %(sanitized_info)s}',
                        datefmt='%Y-%m-%dT%H:%M:%S'
                    )
                    file_handler.setFormatter(sanitized_formatter)
                    logger.addHandler(file_handler)
                except Exception as e:
                    # Log the exception to console or elsewhere
                    log_error(
                        f"Failed to set up file handler: {e}", exc_info=True)
                    # Fallback mechanism (e.g., write to a temporary file) if needed.

        cls._loggers[name] = logger
        return logger

    @classmethod
    def configure(
        cls,
        level: str = "INFO",
        format_str: Optional[str] = None,
        log_dir: Optional[str] = None,
        file_logging_enabled: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB default
        backup_count: int = 5,
    ) -> None:
        """Configure global logging settings."""
        if cls._configured:
            return

        cls._default_level = getattr(logging, level.upper(), logging.INFO)
        cls._default_format = format_str or cls._default_format
        cls._file_logging_enabled = file_logging_enabled
        cls._max_bytes = max_bytes
        cls._backup_count = backup_count

        if log_dir:
            cls._log_dir = Path(log_dir)

        cls._configured = True

    @classmethod
    def shutdown(cls) -> None:
        """Cleanup logging handlers and close files."""
        for logger in cls._loggers.values():
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
        cls._loggers.clear()
        logging.shutdown()

    @classmethod
    def handle_exception(cls, exc_type: type, exc_value: BaseException, exc_traceback: Any) -> None:
        """Global exception handler."""
        if not issubclass(exc_type, KeyboardInterrupt):
            logger = cls.get_logger("global")
            logger.critical(
                "Unhandled exception",
                exc_info=(exc_type, exc_value, exc_traceback),
                extra={'correlation_id': 'N/A', 'sanitized_info': {}}
            )
        # Call the default excepthook if needed
        sys.__excepthook__(exc_type, exc_value, exc_traceback)


class CorrelationLoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter to add a correlation ID to logs."""

    def __init__(self, logger: logging.Logger, correlation_id: Optional[str] = None):
        super().__init__(logger, {})
        self.correlation_id = correlation_id if correlation_id is not None else "N/A"

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> tuple[str, MutableMapping[str, Any]]:
        # Avoid mutating the original kwargs
        extra = kwargs.get('extra', {})
        extra['correlation_id'] = self.correlation_id
        kwargs['extra'] = extra
        return msg, kwargs

# Module-level utility functions (optional)


def log_error(msg: str, *args: Any, exc_info: bool = True, **kwargs: Any) -> None:
    """Log an error message at module level."""
    logger = LoggerSetup.get_logger()
    logger.error(msg, *args, exc_info=exc_info, **kwargs)


def log_debug(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a debug message at module level."""
    logger = LoggerSetup.get_logger()
    logger.debug(msg, *args, **kwargs)


def log_info(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log an info message at module level."""
    logger = LoggerSetup.get_logger()
    logger.info(msg, *args, **kwargs)


def log_warning(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a warning message at module level."""
    logger = LoggerSetup.get_logger()
    logger.warning(msg, *args, **kwargs)


__all__ = [
    'LoggerSetup',
    'CorrelationLoggerAdapter',
    'log_error',
    'log_debug',
    'log_info',
    'log_warning'
]

# Optional: Set up the root logger if needed
# LoggerSetup.configure()

# Optionally, set the global exception handler
# sys.excepthook = LoggerSetup.handle_exception
