"""
Core type definitions for documentation generation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
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
    """Context for managing and generating documentation."""
    
    def __init__(
        self, 
        source_code: str, 
        module_path: Path, 
        include_source: bool = True, 
        metadata: Optional[Dict[str, Any]] = None, 
        ai_generated: Optional[Dict[str, Any]] = None,
        classes: Optional[List[Any]] = None,
        functions: Optional[List[Any]] = None,
        constants: Optional[List[Any]] = None,
        changes: Optional[List[Any]] = None
    ):
        self.source_code = source_code
        self.module_path = module_path
        self.include_source = include_source
        self.metadata = metadata or {}
        self.ai_generated = ai_generated or {}
        self.classes = classes or []
        self.functions = functions or []
        self.constants = constants or []
        self.changes = changes or []


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
            Optional[Tuple[str, str]]: Tuple of (updated_code, documentation) or
            None if processing fails
        """
        pass
