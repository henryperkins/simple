"""Shared type definitions to prevent circular imports."""

from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ProcessingResult:
    """Result of AI processing operation."""
    content: str
    usage: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    cached: bool = False
    processing_time: float = 0.0

@dataclass 
class DocstringData:
    """Common docstring data structure."""
    summary: str
    description: str
    args: list
    returns: dict
    raises: list
    complexity: int