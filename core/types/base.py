"""Base type definitions for code extraction."""

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable, Mapping, Set
import ast
from typing import (
    Protocol,
    runtime_checkable,
    Any,
    TypeVar,
    cast,
    TypedDict,
    Union,
)

from core.dependency_injection import Injector


T = TypeVar('T')


@runtime_checkable
class DependencyAnalyzer(Protocol):
    """Interface for dependency analyzers."""
    def analyze_dependencies(self, node: ast.AST) -> dict[str, set[str]]:
        """Analyze dependencies of an AST node."""
        ...


try:
    from pydantic.v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field


class DocstringSchema(BaseModel):
    """Schema for validating docstring data."""
    summary: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    args: list[dict[str, Any]] = Field(default_factory=list)
    returns: dict[str, str] = Field(...)
    raises: list[dict[str, str]] = Field(default_factory=list)

    def validate_returns(self, self_param: Any, v: dict[str, str]) -> dict[str, str]:
        """Validate returns field."""
        if 'type' not in v or 'description' not in v:
            raise ValueError("Returns must contain 'type' and 'description'")
        return v


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


@dataclass
class ExtractionContext:
    """Context for code extraction operations."""
    module_name: str | None = None
    base_path: Path | None = None
    source_code: str | None = None  # Source code being processed
    include_private: bool = False
    include_nested: bool = False
    include_magic: bool = True  # Controls whether magic methods are included
    tree: ast.AST | None = None  # AST tree if already parsed
    _dependency_analyzer: DependencyAnalyzer | None = None  # Internal storage for lazy initialization

    @property
    def dependency_analyzer(self) -> DependencyAnalyzer | None:
        """Get the dependency analyzer, initializing it if needed."""
        if self._dependency_analyzer is None and self.module_name:
            from core.extraction.dependency_analyzer import DependencyAnalyzer as RealDependencyAnalyzer
            self._dependency_analyzer = RealDependencyAnalyzer(context=self, correlation_id=None)
        return self._dependency_analyzer

    @dependency_analyzer.setter
    def dependency_analyzer(self, value: DependencyAnalyzer | None) -> None:
        """Set the dependency analyzer."""
        self._dependency_analyzer = value


@dataclass
class ExtractionResult:
    """Holds the results of the code extraction process."""
    module_docstring: dict[str, Any] = field(default_factory=dict)
    classes: list[dict[str, Any]] = field(default_factory=list)
    functions: list[dict[str, Any]] = field(default_factory=list)
    variables: list[dict[str, Any]] = field(default_factory=list)
    constants: list[dict[str, Any]] = field(default_factory=list)
    dependencies: dict[str, set[str]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    source_code: str = ""
    module_name: str = ""
    file_path: str = ""


@dataclass
class DocumentationContext:
    """Context for documentation generation operations."""
    source_code: str
    module_path: Path | None = None
    include_source: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    classes: list[dict[str, Any]] = field(default_factory=list)
    functions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ProcessingResult:
    """Represents the result of a processing operation."""
    content: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    validation_status: bool = False
    validation_errors: list[str] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)


@dataclass
class MetricData:
    """Holds data for code metrics analysis."""
    module_name: str = ""
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    maintainability_index: float = 0.0
    halstead_metrics: dict[str, Any] = field(default_factory=dict)
    lines_of_code: int = 0
    total_functions: int = 0
    scanned_functions: int = 0
    function_scan_ratio: float = 0.0
    total_classes: int = 0
    scanned_classes: int = 0
    class_scan_ratio: float = 0.0
    complexity_graph: Any | None = None  # Placeholder for optional graph representation


@dataclass
class ParsedResponse:
    """Response from parsing operations."""
    content: dict[str, Any]
    format_type: str
    parsing_time: float
    validation_success: bool
    errors: list[str]
    metadata: dict[str, Any]
    markdown: str = ""


@dataclass
class TokenUsage:
    """Token usage statistics and cost calculation."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float


@dataclass
class ExtractedArgument:
    """Represents a function argument."""
    name: str
    type: str | None = None
    default_value: str | None = None
    is_required: bool = True
    description: str | None = None


@dataclass
class ExtractedElement:
    """Base class for extracted code elements."""
    name: str
    lineno: int
    source: str | None = None
    docstring: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    dependencies: dict[str, set[str]] = field(default_factory=dict)
    decorators: list[str] = field(default_factory=list)
    complexity_warnings: list[str] = field(default_factory=list)
    ast_node: ast.AST | None = None

    def get_docstring_info(self) -> DocstringData | None:
        """Retrieve or parse the docstring information."""
        if not hasattr(self, "_docstring_info"):
            from core.docstring_processor import DocstringProcessor
            if self.docstring is not None:
                processor = DocstringProcessor()
                self._docstring_info = processor.parse(self.docstring)
            else:
                self._docstring_info = DocstringData(
                    summary="No docstring available.",
                    description="No description available."
                )
        return self._docstring_info


@dataclass
class ExtractedFunction(ExtractedElement):
    """Represents an extracted function with its metadata."""
    args: list[ExtractedArgument] = field(default_factory=list)
    returns: dict[str, str] | None = None
    raises: list[dict[str, str]] = field(default_factory=list)
    body_summary: str | None = None
    is_async: bool = False
    is_method: bool = False
    parent_class: str | None = None


@dataclass
class ExtractedClass(ExtractedElement):
    """Represents a class extracted from code."""
    methods: list[ExtractedFunction] = field(default_factory=list)
    attributes: list[dict[str, Any]] = field(default_factory=list)
    instance_attributes: list[dict[str, Any]] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    metaclass: str | None = None
    is_exception: bool = False
    docstring_info: DocstringData | None = None


class DocstringDict(TypedDict, total=False):
    """Type definition for docstring dictionary."""
    summary: str
    description: str
    args: list[dict[str, Any]]
    returns: dict[str, str]
    raises: list[dict[str, str]]
    complexity: int


@dataclass
class DocumentationData:
    """Documentation data structure."""
    module_name: str
    module_path: Path
    module_summary: str
    source_code: str
    docstring_data: Union[DocstringData, DocstringDict]
    ai_content: dict[str, Any]
    code_metadata: dict[str, Any]
    glossary: dict[str, dict[str, str]] = field(default_factory=dict)
    changes: list[dict[str, Any]] = field(default_factory=list)
    complexity_scores: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    validation_status: bool = False
    validation_errors: list[str] = field(default_factory=list)
    docstring_parser: Callable[[str], DocstringData] | None = None
    metric_calculator: Callable[[str], dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        """Initialize dependencies."""
        if self.docstring_parser is None:
            self.docstring_parser = Injector.get("docstring_processor")
        if self.metric_calculator is None:
            self.metric_calculator = Injector.get("metrics_calculator")

        # Convert dict to DocstringData if needed
        if isinstance(self.docstring_data, dict):
            docstring_dict = self.docstring_data
            self.docstring_data = DocstringData(
                summary=str(docstring_dict.get("summary", "")),
                description=str(docstring_dict.get("description", "")),
                args=docstring_dict.get("args", []),
                returns=docstring_dict.get("returns", {}),
                raises=docstring_dict.get("raises", []),
                complexity=int(docstring_dict.get("complexity", 1))
            )

        # Ensure module summary is never None
        if not self.module_summary:
            ai_summary = self.ai_content.get("summary")
            self.module_summary = str(
                ai_summary if isinstance(ai_summary, str)
                else self.docstring_data.summary if isinstance(self.docstring_data, DocstringData)
                else "No module summary available."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert DocumentationData to a dictionary."""
        return {
            "module_name": self.module_name,
            "module_path": str(self.module_path),
            "module_summary": self.module_summary,
            "source_code": self.source_code,
            "docstring_data": self.docstring_data.to_dict() if isinstance(self.docstring_data, DocstringData) else self.docstring_data,
            "ai_content": self.ai_content,
            "code_metadata": self.code_metadata,
            "glossary": self.glossary,
            "changes": self.changes,
            "complexity_scores": self.complexity_scores,
            "metrics": self.metrics,
            "validation_status": self.validation_status,
            "validation_errors": self.validation_errors,
        }
