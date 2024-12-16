"""Docstring data structures and types."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocstringData:
    """Unified data model for docstring information."""
    summary: str = "No summary available"
    description: str = "No description available"
    args: list[dict[str, Any]] = field(default_factory=list)
    returns: dict[str, str] = field(default_factory=lambda: {"type": "Any", "description": ""})
    raises: list[dict[str, str]] = field(default_factory=list)
    complexity: int = 1

    def validate(self) -> tuple[bool, list[str]]:
        """Validate the docstring data against schema."""
        try:
            # Import here to avoid circular imports
            from core.types.base import DocstringSchema
            DocstringSchema(
                summary=self.summary,
                description=self.description,
                args=self.args,
                returns=self.returns,
                raises=self.raises
            )
            return True, []
        except ValueError as e:
            return False, [str(e)]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "summary": self.summary,
            "description": self.description,
            "args": self.args,
            "returns": self.returns,
            "raises": self.raises,
            "complexity": self.complexity
        }
