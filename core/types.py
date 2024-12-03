"""
Core type definitions for documentation generation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DocstringData:
    """Structured representation of a docstring."""
    summary: str
    description: str
    args: List[Dict[str, Any]]
    returns: Dict[str, Any]
    raises: List[Dict[str, Any]]
    complexity: int = 1  # Default to 1 if not provided

@dataclass
class ParsedResponse:
    """Container for parsed response data."""
    content: Dict[str, Any]
    format_type: str
    parsing_time: float
    validation_success: bool
    errors: List[str]
    metadata: Dict[str, Any]

@dataclass
class ProcessingResult:
    """Result of AI processing operation."""
    content: str
    usage: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    is_cached: bool = False
    processing_time: float = 0.0

class DocumentationContext:
    def __init__(
        self,
        source_code: str,
        module_path: Optional[Path] = None,
        include_source: bool = False,
        ai_generated: Optional[str] = None,
        changes: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None  # Add this line
    ):
        self.source_code = source_code
        self.module_path = module_path
        self.include_source = include_source
        self.ai_generated = ai_generated
        self.changes = changes or []
        self.metadata = metadata or {}  # Initialize with an empty dict if not provided

class AIHandler(ABC):
    """Interface for AI processing."""
    
    @abstractmethod
    async def process_code(
        self,
        source_code: str,
        cache_key: Optional[str] = None,
        extracted_info: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[str, str]]:
        """
        Process code and generate documentation.

        Args:
            source_code: The source code to process
            cache_key: Optional cache key for storing results
            extracted_info: Optional pre-extracted code information

        Returns:
            Optional[Tuple[str, str]]: Tuple of (updated_code, documentation) or None if processing fails
        """
        pass