import logging
from logging import LogRecord
from contextvars import ContextVar
from typing import Any, TypeVar, Generic, cast, Optional
from collections.abc import Mapping, MutableMapping
from types import TracebackType
import re
from datetime import datetime
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

T = TypeVar("T", bound=logging.Logger)

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

def get_logger(name: str, correlation_id: Optional[str] = None) -> CorrelationLoggerAdapter:
    """Get a logger instance with optional correlation ID."""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        formatter = SanitizedLogFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if correlation_id:
        set_correlation_id(correlation_id)

    return CorrelationLoggerAdapter(logger, {"correlation_id": correlation_id})
