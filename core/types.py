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
    complexity: Optional[int] = None

@dataclass
class ProcessingResult:
    """Result of AI processing operation."""
    content: str
    usage: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    is_cached: bool = False
    processing_time: float = 0.0

@dataclass
class DocumentationContext:
    """Context for documentation generation."""
    source_code: str
    module_path: Optional[Path] = None
    include_source: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    ai_generated: Optional[str] = None
    metrics_enabled: bool = True  # Add this line

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