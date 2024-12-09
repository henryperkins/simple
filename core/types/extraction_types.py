"""Types for code extraction results."""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    docstring: Optional[str] = None
    methods: List[Dict[str, Any]] = None

@dataclass
class FunctionInfo:
    """Information about a function."""
    name: str
    docstring: Optional[str] = None
    params: List[Dict[str, Any]] = None
    returns: Dict[str, Any] = None

@dataclass
class ExtractionResult:
    """Result of code extraction."""
    source_code: str
    module_docstring: Optional[str]
    classes: List[ClassInfo]
    functions: List[FunctionInfo]
    file_path: str
