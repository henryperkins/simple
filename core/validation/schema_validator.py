"""Base schema validator class."""

from typing import Any, List, Dict
from jsonschema import validate, ValidationError

from core.logger import LoggerSetup


class SchemaValidator:
    """Base class for schema validation."""

    def __init__(self, logger_name: str, correlation_id: str | None = None):
        """Initialize the schema validator."""
        self.logger = LoggerSetup.get_logger(logger_name, correlation_id)
        self.correlation_id = correlation_id

    def validate_schema(
        self, instance: Any, schema: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Validate the instance against the schema."""
        validation_errors: List[str] = []
        try:
            validate(instance=instance, schema=schema)
            return True, validation_errors
        except ValidationError as e:
            validation_errors.append(str(e))
            self.logger.error(
                f"Validation error: {e}", extra={"correlation_id": self.correlation_id}
            )
            return False, validation_errors
        except Exception as e:
            validation_errors.append(f"Unexpected validation error: {str(e)}")
            self.logger.error(
                f"Unexpected validation error: {e}",
                exc_info=True,
                extra={"correlation_id": self.correlation_id},
            )
            return False, validation_errors