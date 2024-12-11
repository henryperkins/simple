"""Enhanced logging configuration with structured output."""
import logging
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, MutableMapping, Optional, Union
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler
from contextvars import ContextVar
from functools import wraps
from core.console import console

# Context variable for the correlation ID
correlation_id_var = ContextVar('correlation_id')

def set_correlation_id(correlation_id: str):
    """Set the correlation ID in the context."""
    correlation_id_var.set(correlation_id)

def get_correlation_id() -> str:
    """Retrieve the correlation ID from the context or return 'N/A' if not set."""
    return correlation_id_var.get('N/A')

class SanitizedLogFormatter(logging.Formatter):
    """Custom formatter to sanitize and format log records."""

    def format(self, record):
        # Ensure correlation_id and sanitized_info fields exist
        record.correlation_id = get_correlation_id()
        record.sanitized_info = getattr(record, 'sanitized_info', {"info": "[Sanitized]"})

        # Sanitize the message and arguments
        record.msg = self._sanitize(record.msg)
        if record.args:
            record.args = tuple(self._sanitize(arg) for arg in record.args)

        # Format the timestamp
        if self.usesTime():
            record.asctime = datetime.fromtimestamp(record.created).isoformat() + 'Z'

        return super().format(record)

    def _sanitize(self, item: Any) -> Any:
        """Sanitize sensitive information from logs."""
        if isinstance(item, dict):
            return {k: self._sanitize(v) for k, v in item.items()}
        elif isinstance(item, (list, tuple)) and not isinstance(item, str):
            return [self._sanitize(it) for it in item]
        elif isinstance(item, str):
            # Sanitize file paths and secrets
            item = re.sub(r'(/[a-zA-Z0-9_\-./]+)', '[SANITIZED_PATH]', item)
            item = re.sub(r'(secret_key|password|token)=[^&\s]+', r'\1=[REDACTED]', item)
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
    _max_bytes: int = 10 * 1024 * 1024  # 10MB
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

        if not logger.handlers:
            # Console handler
            console_handler = RichHandler(console=console)
            console_formatter = logging.Formatter(cls._default_format)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            # File handler
            if cls._file_logging_enabled:
                try:
                    cls._log_dir.mkdir(parents=True, exist_ok=True)
                    file_handler = RotatingFileHandler(
                        cls._log_dir / f"{name}.log",
                        maxBytes=cls._max_bytes,
                        backupCount=cls._backup_count
                    )
                    file_formatter = SanitizedLogFormatter(
                        fmt='{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", '
                            '"message": "%(message)s", "correlation_id": "%(correlation_id)s", '
                            '"sanitized_info": %(sanitized_info)s}',
                        datefmt='%Y-%m-%dT%H:%M:%S'
                    )
                    file_handler.setFormatter(file_formatter)
                    logger.addHandler(file_handler)
                except Exception as e:
                    console.print(f"Failed to set up file handler: {e}", style="red")

        # Ensure the logger has the correct methods
        if not hasattr(logger, 'isEnabledFor'):
            logger.isEnabledFor = lambda level: True  # Dummy method to avoid AttributeError

        cls._loggers[name] = logger
        return CorrelationLoggerAdapter(logger, extra={'correlation_id': get_correlation_id()})

    @classmethod
    def configure(
        cls,
        level: str = "INFO",
        format_str: Optional[str] = None,
        log_dir: Optional[str] = None,
        file_logging_enabled: bool = True,
        max_bytes: int = 10 * 1024 * 1024,
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
            for handler in logger.handlers[:]:
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
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

class CorrelationLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds correlation ID to logs."""

    def __init__(self, logger, extra=None, correlation_id=None):
        if extra is None:
            extra = {}
        if correlation_id is not None:
            set_correlation_id(correlation_id)
        super().__init__(logger, extra={'correlation_id': get_correlation_id()})

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> tuple[str, MutableMapping[str, Any]]:
        extra = kwargs.get('extra', {})
        extra['correlation_id'] = get_correlation_id()
        kwargs['extra'] = extra
        return msg, kwargs

# Utility functions
def log_error(msg: str, *args: Any, exc_info: bool = True, **kwargs: Any) -> None:
    """Log an error message."""
    logger = LoggerSetup.get_logger()
    logger.error(msg, *args, exc_info=exc_info, **kwargs)

def log_debug(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a debug message."""
    logger = LoggerSetup.get_logger()
    logger.debug(msg, *args, **kwargs)

def log_info(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log an info message."""
    logger = LoggerSetup.get_logger()
    logger.info(msg, *args, **kwargs)

def log_warning(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a warning message."""
    logger = LoggerSetup.get_logger()
    logger.warning(msg, *args, **kwargs)

def handle_error(func):
    """Decorator to handle common exceptions with logging."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger = LoggerSetup.get_logger()
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise
    return wrapper
