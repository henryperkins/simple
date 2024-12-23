from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional


@dataclass
class DocstringData:
    """Docstring data structure."""
    summary: str
    description: str
    args: List[Dict[str, Any]] = field(default_factory=list)
    returns: Dict[str, str] = field(default_factory=lambda: {"type": "Any", "description": ""})
    raises: List[Dict[str, str]] = field(default_factory=list)
    complexity: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "description": self.description,
            "args": self.args,
            "returns": self.returns,
            "raises": self.raises,
            "complexity": self.complexity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocstringData":
        """Create a DocstringData instance from a dictionary."""
        return cls(
            summary=data.get("summary", ""),
            description=data.get("description", ""),
            args=data.get("args", []),
            returns=data.get("returns", {"type": "Any", "description": ""}),
            raises=data.get("raises", []),
            complexity=data.get("complexity", 1),
        )