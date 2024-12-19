"""Docstring-specific validator class."""

from typing import Any, List, Dict
from pathlib import Path
import json

from core.logger import LoggerSetup
from core.validation.schema_validator import SchemaValidator


class DocstringValidator(SchemaValidator):
    """Validator for docstring data."""

    def __init__(self, correlation_id: str | None = None):
        """Initialize the docstring validator."""
        super().__init__(
            logger_name=f"{__name__}.{self.__class__.__name__}",
            correlation_id=correlation_id,
        )
        self.docstring_schema = self.load_schema("docstring_schema.json")

    def validate_docstring(self, content: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate the content against the docstring schema."""
        if not self.docstring_schema:
            return False, ["Docstring schema not loaded"]
        schema = self.docstring_schema.get("schema", {})
        if not schema:
            return False, ["Invalid docstring schema structure"]
        return self.validate_schema(content, schema)
