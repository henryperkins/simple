"""Docstring-specific validator class."""

from typing import Any, List, Dict
from jsonschema import validate, ValidationError

from core.logger import LoggerSetup
from core.types.base import DocstringSchema
from core.validation.schema_validator import SchemaValidator


class DocstringValidator(SchemaValidator):
    """Validator for docstring data."""

    def __init__(self, correlation_id: str | None = None):
        """Initialize the docstring validator."""
        super().__init__(
            logger_name=f"{__name__}.{self.__class__.__name__}",
            correlation_id=correlation_id,
        )
        self.docstring_schema = self._load_schema()

    def _load_schema(self) -> Dict[str, Any]:
        """Load the docstring schema."""
        from pathlib import Path
        import json

        schema_path = (
            Path(__file__).resolve().parent.parent.parent / "schemas" / "docstring_schema.json"
        )
        try:
            with schema_path.open("r") as f:
                return json.load(f)
        except FileNotFoundError as e:
            self.logger.error(
                f"Schema file not found: docstring_schema.json - {e}", exc_info=True
            )
            raise
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Error decoding JSON schema: docstring_schema.json - {e}",
                exc_info=True,
            )
            raise

    def validate_docstring(self, content: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate the content against the docstring schema."""
        if not self.docstring_schema:
            return False, ["Docstring schema not loaded"]
        schema = self.docstring_schema.get("schema", {})
        if not schema:
            return False, ["Invalid docstring schema structure"]
        return self.validate_schema(content, schema)
