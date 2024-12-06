"""
Core type definitions for code analysis and documentation generation.
Provides dataclass definitions for structured data handling throughout the application.
"""

# Standard library imports
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Union

@dataclass
class BaseData:
    """Base class for data structures with common fields."""
    name: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocstringData:
    """Structured representation of a docstring."""
    summary: str
    args: List[Dict[str, Any]]
    returns: Dict[str, Any]
    raises: List[Dict[str, Any]]
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    complexity: int = 1


@dataclass
class TokenUsage:
    """Token usage statistics and cost calculation."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float


@dataclass
class ExtractedElement:
    """Base class for extracted code elements."""
    name: str
    lineno: int
    source: Optional[str] = None
    docstring: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    decorators: List[str] = field(default_factory=list)
    complexity_warnings: List[str] = field(default_factory=list)


@dataclass
class ExtractedArgument:
    """Represents a function argument."""
    name: str
    type: Optional[str] = None
    default_value: Optional[str] = None
    is_required: bool = True

@dataclass
class ExtractedFunction(ExtractedElement):
    """Represents an extracted function with its metadata."""
    name: str
    lineno: int
    source: Optional[str] = None
    docstring: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    decorators: List[str] = field(default_factory=list)
    complexity_warnings: List[str] = field(default_factory=list)
    raises: List[Dict[str, str]] = field(default_factory=list)
    body_summary: Optional[str] = None
    ast_node: Optional[ast.AST] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the ExtractedFunction instance to a dictionary."""
        return {
            "name": self.name,
            "lineno": self.lineno,
            "source": self.source,
            "docstring": self.docstring,
            "metrics": self.metrics,
            "dependencies": self.dependencies,
            "decorators": self.decorators,
            "complexity_warnings": self.complexity_warnings,
            "raises": [r.get('exception', 'Unknown') for r in self.raises],
            "body_summary": self.body_summary
            # ast_node intentionally excluded from dict conversion
        }

@dataclass
class ExtractedClass(ExtractedElement):
    """Represents extracted information about a class."""
    name: str
    docstring: str
    raises: List[Dict[str, str]] = field(default_factory=list)
    methods: List[ExtractedFunction] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    bases: List[str] = field(default_factory=list)
    lineno: int = 0
    source: str = ""
    dependencies: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    is_exception: bool = False
    decorators: List[str] = field(default_factory=list)
    instance_attributes: List[str] = field(default_factory=list)
    metaclass: Optional[str] = None
    complexity_warnings: List[str] = field(default_factory=list)
    ast_node: Optional[ast.ClassDef] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the ExtractedClass instance to a dictionary."""
        return {
            "name": self.name,
            "docstring": self.docstring,
            "raises": [r.get('exception', 'Unknown') for r in self.raises],
            "methods": [method.to_dict() for method in self.methods],
            "metrics": self.metrics,
            "bases": self.bases,
            "lineno": self.lineno,
            "source": self.source,
            "dependencies": self.dependencies,
            "attributes": self.attributes,
            "is_exception": self.is_exception,
            "decorators": self.decorators,
            "instance_attributes": self.instance_attributes,
            "metaclass": self.metaclass,
            "complexity_warnings": self.complexity_warnings
            # ast_node intentionally excluded from dict conversion
        }

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


@dataclass
class DocumentationContext:
    """Context for managing and generating documentation."""
    source_code: str
    module_path: Path
    include_source: bool = True
    metadata: Optional[Dict[str, Any]] = None
    ai_generated: Optional[Dict[str, Any]] = None
    classes: Optional[List[Any]] = None
    functions: Optional[List[Any]] = None
    constants: Optional[List[Any]] = None
    changes: Optional[List[Any]] = None

    def get_cache_key(self) -> str:
        """
        Generate a cache key for this documentation context.

        Returns:
            A unique hash combining the source code and metadata.
        """
        import hashlib

        key_parts = [
            self.source_code,
            str(self.module_path),
            str(self.metadata or {})
        ]
        combined = "|".join(key_parts).encode("utf-8")
        return hashlib.sha256(combined).hexdigest()


@dataclass
class ExtractionContext:
    """Context for code extraction operations."""
    metrics_enabled: bool = True
    module_name: Optional[str] = None
    include_private: bool = False
    include_magic: bool = False
    include_nested: bool = True
    include_source: bool = True
    max_line_length: int = 88
    ignore_decorators: Set[str] = field(default_factory=set)
    base_path: Optional[Path] = None
    source_code: Optional[str] = None

    def __init__(self):
        self.tree = None
        self.metrics_enabled = True
        self.include_private = False

@dataclass
class ExtractionResult:
    """Result of code extraction."""
    module_docstring: Dict[str, Any]
    classes: List[Any] = field(default_factory=list)
    functions: List[Any] = field(default_factory=list)
    variables: List[Dict[str, Any]] = field(default_factory=list)
    constants: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    maintainability_index: Optional[float] = None
    source_code: str = ""  # Added field for source code
    imports: List[Any] = field(default_factory=list)  # Added imports field

    def to_dict(self) -> Dict[str, Any]:
        """Convert extraction result to dictionary format."""
        return {
            'functions': self.functions if hasattr(self, 'functions') else [],
            'classes': self.classes if hasattr(self, 'classes') else [],
            'imports': self.imports if hasattr(self, 'imports') else [],
            'dependencies': self.dependencies if hasattr(self, 'dependencies') else []
        }
@dataclass
class DocumentationData:
    """Standardized documentation data structure."""
    module_info: Dict[str, str]
    ai_content: Dict[str, Any]
    docstring_data: DocstringData
    code_metadata: Dict[str, Any]
    source_code: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary containing all documentation data.
        """
        return {
            "module_info": self.module_info,
            "ai_content": self.ai_content,
            "docstring_data": {
                "summary": self.docstring_data.summary,
                "description": self.docstring_data.description,
                "args": self.docstring_data.args,
                "returns": self.docstring_data.returns,
                "raises": self.docstring_data.raises,
                "complexity": self.docstring_data.complexity,
            },
            "code_metadata": self.code_metadata,
            "source_code": self.source_code,
            "metrics": self.metrics,
        }