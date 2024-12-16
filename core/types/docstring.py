from dataclasses import dataclass, field
from typing import Any, Dict, List, TypedDict

@dataclass
class DocstringData:
    """Docstring data structure."""
    summary: str
    description: str
    args: list[dict[str, Any]] = field(default_factory=list)
    returns: dict[str, str] = field(default_factory=lambda: {"type": "Any", "description": ""})
    raises: list[dict[str, str]] = field(default_factory=list)
    complexity: int = 1

    def validate(self) -> tuple[bool, list[str]]:
        """Validate docstring data."""
        errors = []
        if not self.summary:
            errors.append("Summary is required")
        if not self.description:
            errors.append("Description is required")
        return not errors, errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "description": self.description,
            "args": self.args,
            "returns": self.returns,
            "raises": self.raises,
            "complexity": self.complexity,
        }
