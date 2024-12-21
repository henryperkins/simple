"""Enhanced logging configuration with structured output."""

import logging
from logging import LogRecord, Logger
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Any, TypeVar, Generic, cast, Optional
from collections.abc import Mapping, MutableMapping
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar
from types import TracebackType

T = TypeVar("T", bound=Logger)

# Context variable for the correlation ID
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID in the context."""
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> str | None:
    """Retrieve the correlation ID from the context or return None if not set."""
    return correlation_id_var.get()


class CorrelationLoggerAdapter(logging.LoggerAdapter[T], Generic[T]):
    """Logger adapter that includes correlation ID in log messages."""

    RESERVED_KEYS = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "process",
        "processName",
    }

    def __init__(self, logger: T, extra: Mapping[str, Any] | None = None) -> None:
        super().__init__(logger, extra or {})
        self.correlation_id_var = correlation_id_var

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        extra = kwargs.get("extra", {})

        # Rename reserved keys in the extra dictionary
        sanitized_extra = {
            (f"{key}_extra" if key in self.RESERVED_KEYS else key): value
            for key, value in extra.items()
        }

        # Add the correlation ID to the sanitized extra dictionary
        correlation_id = self.correlation_id_var.get()
        sanitized_extra["correlation_id"] = correlation_id or "N/A"

        kwargs["extra"] = sanitized_extra
        return msg, kwargs


class SanitizedLogFormatter(logging.Formatter):
    """Custom formatter to sanitize and format log records."""

    def format(self, record: LogRecord) -> str:
        """Format log records with consistent formatting."""
        # Ensure correlation_id and sanitized_info fields exist
        setattr(record, "correlation_id", get_correlation_id() or "N/A")
        setattr(
            record,
            "sanitized_info",
            getattr(record, "sanitized_info", {"info": "[Sanitized]"}),
        )

        # Sanitize the message and arguments
        record.msg = self._sanitize(record.msg)
        if record.args:
            record.args = tuple(self._sanitize(arg) for arg in record.args)

        # Format the timestamp
        if self.usesTime():
            record.asctime = datetime.fromtimestamp(record.created).isoformat() + "Z"

        # Add section breaks for error and critical logs
        if record.levelno >= logging.ERROR:
            formatted = "\n" + "-" * 80 + "\n"
            formatted += f"ERROR ({record.correlation_id}):\n"
            formatted += "  " + super().format(record)
            formatted += "\n" + "-" * 80 + "\n"
            return formatted

        return super().format(record)

    def _sanitize(self, item: Any) -> Any:
        """Sanitize sensitive information from logs."""
        if isinstance(item, dict):
            return {str(k): self._sanitize(v) for k, v in item.items()}
        elif isinstance(item, (list, tuple)) and not isinstance(item, str):
            return [self._sanitize(x) for x in item]
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

    _loggers: dict[str, Logger] = {}
    _log_dir: Path = Path("logs")
    _file_logging_enabled: bool = True
    _configured: bool = False
    _default_level: int = logging.INFO
    _default_format: str = "%(message)s"
    _max_bytes: int = 10 * 1024 * 1024  # 10MB
    _backup_count: int = 5

    @classmethod
    def configure(cls, config: Any | None = None) -> None:
        """Configure global logging settings."""
        if cls._configured:
            return

        if config is not None:
            cls._default_level = getattr(
                logging, config.app.log_level.upper(), logging.INFO
            )
            cls._log_dir = Path(config.app.log_dir)
        else:
            cls._default_level = logging.INFO
            cls._log_dir = Path("logs")

        cls._configured = True

        # Configure the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(cls._default_level)

        # Remove all existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Console handler with simplified format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Set console level to INFO
        console_handler.setFormatter(logging.Formatter(cls._default_format))
        root_logger.addHandler(console_handler)

        # File handler with detailed JSON format (if enabled)
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
                file_handler.setLevel(logging.DEBUG)  # Capture all logs to file
                root_logger.addHandler(file_handler)
            except Exception as e:
                sys.stderr.write(f"Failed to set up file handler: {e}\n")

    _logged_messages = set()

    @classmethod
    def log_once(cls, logger, level, message):
        """Log a message only once."""
        if message not in cls._logged_messages:
            cls._logged_messages.add(message)
            logger.log(level, message)

    @classmethod
    def get_logger(
        cls, name: str | None = None, correlation_id: str | None = None
    ) -> Logger:
        """Get a configured logger instance with optional correlation ID."""
        if not cls._configured:
            cls.configure()

        if name is None:
            name = __name__

        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        if not hasattr(logger, "isEnabledFor"):
            logger.isEnabledFor = lambda level: True

        cls._loggers[name] = logger

        if correlation_id:
            set_correlation_id(correlation_id)

        return cast(Logger, cls._get_correlation_logger_adapter(logger))

    @classmethod
    def _get_correlation_logger_adapter(
        cls, logger: Logger
    ) -> CorrelationLoggerAdapter[Logger]:
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
        cls,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
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
