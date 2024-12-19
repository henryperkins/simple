from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import ast
from typing import (
    Protocol,
    runtime_checkable,
    Any,
    TypeVar,
    TypedDict,
    Union,
    Callable,
    Dict,
    List,
    Optional,
)

from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.types.docstring import DocstringData

T = TypeVar('T')


@runtime_checkable
class DependencyAnalyzer(Protocol):
    """Interface for dependency analyzers."""
    def analyze_dependencies(self, node: ast.AST) -> Dict[str, set[str]]:
        """Analyze dependencies of an AST node."""
        ...


# Attempt import for Pydantic v1 compatibility, else fallback
try:
    from pydantic.v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field


class DocstringSchema(BaseModel):
    """Schema for validating docstring data."""
    summary: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    args: List[Dict[str, Any]] = Field(default_factory=list)
    returns: Dict[str, str] = Field(...)
    raises: List[Dict[str, str]] = Field(default_factory=list)

    def validate_returns(self, self_param: Any, v: Dict[str, str]) -> Dict[str, str]:
        """Validate the 'returns' field content."""
        if 'type' not in v or 'description' not in v:
            raise ValueError("Returns must contain 'type' and 'description'")
        return v


@dataclass
class ExtractionContext:
    """Context for code extraction operations."""
    _source_code: Optional[str] = None
    module_name: Optional[str] = None
    base_path: Optional[Path] = None
    include_private: bool = False
    include_nested: bool = False
    include_magic: bool = True
    tree: Optional[ast.AST] = None
    _dependency_analyzer: Optional[DependencyAnalyzer] = None
    function_extractor: Optional[Any] = None
    docstring_processor: Optional[Any] = None
    logger: CorrelationLoggerAdapter = field(default_factory=lambda: CorrelationLoggerAdapter(LoggerSetup.get_logger(__name__)))
    metrics_collector: Optional[Any] = None
    strict_mode: bool = False
    config: Dict[str, Any] = field(default_factory=dict)

    def get_source_code(self) -> Optional[str]:
        """Get the source code of this instance."""
        return self._source_code

    def set_source_code(self, value: str, source: Optional[str] = None) -> None:
        """Set the source code with logging and validation."""
        if not value or not value.strip():
            raise ValueError(f"Source code cannot be empty or null for {source}")
        self._source_code = value
        self.logger.debug(f"Updated source code in context {type(self)}: {value[:50]}...")

    @property
    def dependency_analyzer(self) -> Optional[DependencyAnalyzer]:
        """Get or initialize the dependency analyzer."""
        if self._dependency_analyzer is None and self.module_name:
            from core.extraction.dependency_analyzer import DependencyAnalyzer as RealDependencyAnalyzer
            self._dependency_analyzer = RealDependencyAnalyzer(context=self, correlation_id=None)
        return self._dependency_analyzer

    @dependency_analyzer.setter
    def dependency_analyzer(self, value: Optional[DependencyAnalyzer]) -> None:
        """Set the dependency analyzer."""
        self._dependency_analyzer = value


@dataclass
class ExtractionResult:
    """Holds the results of the code extraction process."""
    source_code: str
    module_docstring: Dict[str, Any] = field(default_factory=dict)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    variables: List[Dict[str, Any]] = field(default_factory=list)
    constants: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: Dict[str, set[str]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    module_name: str = ""
    file_path: str = ""


@dataclass
class DocumentationContext:
    """Context for documentation generation operations."""
    source_code: str
    module_path: Optional[Path] = None
    include_source: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate required fields."""
        if not self.source_code or not self.source_code.strip():
            raise ValueError("source_code is required and cannot be empty")


@dataclass
class ProcessingResult:
    """Represents the result of a processing operation."""
    content: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    validation_status: bool = False
    validation_errors: List[str] = field(default_factory=list)
    schema_errors: List[str] = field(default_factory=list)


@dataclass
class MetricData:
    """Holds data for code metrics analysis."""
    module_name: str = ""
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    maintainability_index: float = 0.0
    halstead_metrics: Dict[str, Any] = field(default_factory=dict)
    lines_of_code: int = 0
    total_functions: int = 0
    scanned_functions: int = 0
    function_scan_ratio: float = 0.0
    total_classes: int = 0
    scanned_classes: int = 0
    class_scan_ratio: float = 0.0
    complexity_graph: Optional[Any] = None


@dataclass
class ParsedResponse:
    """Response from parsing operations."""
    content: Dict[str, Any]
    format_type: str
    parsing_time: float
    validation_success: bool
    errors: List[str]
    metadata: Dict[str, Any]
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
    type: Optional[str] = None
    default_value: Optional[str] = None
    is_required: bool = True
    description: Optional[str] = None


@dataclass
class ExtractedElement:
    """Base class for extracted code elements."""
    name: str
    lineno: int
    source: Optional[str] = None
    docstring: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, set[str]] = field(default_factory=dict)
    decorators: List[str] = field(default_factory=list)
    complexity_warnings: List[str] = field(default_factory=list)
    ast_node: Optional[ast.AST] = None

    def get_docstring_info(self) -> Optional[DocstringData]:
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
    def to_dict(self) -> Dict[str, Any]:
        """Convert ExtractedFunction to a dictionary."""
        return {
            "name": self.name,
            "lineno": self.lineno,
            "source": self.source,
            "docstring": self.docstring,
            "metrics": self.metrics,
            "dependencies": self.dependencies,
            "decorators": self.decorators,
            "complexity_warnings": self.complexity_warnings,
            "args": [arg.to_dict() if hasattr(arg, "to_dict") else asdict(arg) for arg in self.args],
            "returns": self.returns,
            "raises": self.raises,
            "body_summary": self.body_summary,
            "is_async": self.is_async,
            "is_method": self.is_method,
            "parent_class": self.parent_class,
        }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractedFunction:
        """Create an ExtractedFunction instance from a dictionary."""
        return cls(
            name=data.get("name", ""),
            lineno=data.get("lineno", 0),
            source=data.get("source"),
            docstring=data.get("docstring"),
            metrics=data.get("metrics", {}),
            dependencies=data.get("dependencies", {}),
            decorators=data.get("decorators", []),
            complexity_warnings=data.get("complexity_warnings", []),
            ast_node=data.get("ast_node"),
            args=[
                ExtractedArgument(**arg) if isinstance(arg, dict) else arg
                for arg in data.get("args", [])
            ],
            returns=data.get("returns"),
            raises=data.get("raises", []),
            body_summary=data.get("body_summary"),
            is_async=data.get("is_async", False),
            is_method=data.get("is_method", False),
            parent_class=data.get("parent_class"),
        )
    """Represents an extracted function with its metadata."""
    args: List[ExtractedArgument] = field(default_factory=list)
    returns: Optional[Dict[str, str]] = None
    raises: List[Dict[str, str]] = field(default_factory=list)
    body_summary: Optional[str] = None
    is_async: bool = False
    is_method: bool = False
    parent_class: Optional[str] = None


@dataclass
class ExtractedClass:
    name: str
    lineno: int
    source: Optional[str] = None
    docstring: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, set[str]] = field(default_factory=dict)
    decorators: List[str] = field(default_factory=list)
    complexity_warnings: List[str] = field(default_factory=list)
    ast_node: Optional[ast.AST] = None
    methods: List[Any] = field(default_factory=list)
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    instance_attributes: List[Dict[str, Any]] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)
    metaclass: Optional[str] = None
    is_exception: bool = False
    docstring_info: Any = None
    is_dataclass: bool = False
    is_abstract: bool = False
    abstract_methods: List[str] = field(default_factory=list)
    property_methods: List[Dict[str, Any]] = field(default_factory=list)
    class_variables: List[Dict[str, Any]] = field(default_factory=list)
    method_groups: Dict[str, List[str]] = field(default_factory=dict)
    inheritance_chain: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractedClass:
        """Create an ExtractedClass instance from a dictionary."""
        return cls(
            name=data.get("name", ""),
            lineno=data.get("lineno", 0),
            source=data.get("source"),
            docstring=data.get("docstring"),
            metrics=data.get("metrics", {}),
            dependencies=data.get("dependencies", {}),
            decorators=data.get("decorators", []),
            complexity_warnings=data.get("complexity_warnings", []),
            ast_node=data.get("ast_node"),
            methods=data.get("methods", []),
            attributes=data.get("attributes", []),
            instance_attributes=data.get("instance_attributes", []),
            bases=data.get("bases", []),
            metaclass=data.get("metaclass"),
            is_exception=data.get("is_exception", False),
            docstring_info=data.get("docstring_info"),
            is_dataclass=data.get("is_dataclass", False),
            is_abstract=data.get("is_abstract", False),
            abstract_methods=data.get("abstract_methods", []),
            property_methods=data.get("property_methods", []),
            class_variables=data.get("class_variables", []),
            method_groups=data.get("method_groups", {}),
            inheritance_chain=data.get("inheritance_chain", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "lineno": self.lineno,
            "source": self.source,
            "docstring": self.docstring,
            "metrics": self.metrics,
            "dependencies": self.dependencies,
            "decorators": self.decorators,
            "complexity_warnings": self.complexity_warnings,
            "methods": [method.to_dict() if hasattr(method, "to_dict") else asdict(method) for method in self.methods],
            "attributes": self.attributes,
            "instance_attributes": self.instance_attributes,
            "bases": self.bases,
            "metaclass": self.metaclass,
            "is_exception": self.is_exception,
            "docstring_info": self.docstring_info.to_dict() if hasattr(self.docstring_info, "to_dict") else self.docstring_info,
            "is_dataclass": self.is_dataclass,
            "is_abstract": self.is_abstract,
            "abstract_methods": self.abstract_methods,
            "property_methods": self.property_methods,
            "class_variables": self.class_variables,
            "method_groups": self.method_groups,
            "inheritance_chain": self.inheritance_chain,
        }


class DocstringDict(TypedDict, total=False):
    """Type definition for docstring dictionary."""
    summary: str
    description: str
    args: List[Dict[str, Any]]
    returns: Dict[str, str]
    raises: List[Dict[str, str]]
    complexity: int


@dataclass
class DocumentationData:
    """Documentation data structure."""
    module_name: str
    module_path: Path
    module_summary: str
    source_code: str
    docstring_data: Union[DocstringData, DocstringDict]
    ai_content: Dict[str, Any]
    code_metadata: Dict[str, Any]
    glossary: Dict[str, Dict[str, str]] = field(default_factory=dict)
    changes: List[Dict[str, Any]] = field(default_factory=list)
    complexity_scores: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    validation_status: bool = False
    validation_errors: List[str] = field(default_factory=list)
    docstring_parser: Optional[Callable[[str], DocstringData]] = None
    metric_calculator: Optional[Callable[[str], Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        """Initialize dependencies."""
        from core.dependency_injection import Injector  # Avoid circular imports

        if self.docstring_parser is None:
            self.docstring_parser = Injector.get("docstring_processor")
        if self.metric_calculator is None:
            self.metric_calculator = Injector.get("metrics_calculator")

        # Convert dict to DocstringData if needed
        if isinstance(self.docstring_data, dict):
            docstring_dict = self.docstring_data.copy()
            docstring_dict.pop('source_code', None)
            self.docstring_data = DocstringData(
                summary=str(docstring_dict.get("summary", "")),
                description=str(docstring_dict.get("description", "")),
                args=docstring_dict.get("args", []),
                returns=docstring_dict.get("returns", {"type": "Any", "description": ""}),
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

        if not self.source_code or not self.source_code.strip():
            raise ValueError("source_code is required and cannot be empty")

        if not isinstance(self.code_metadata, dict):
            self.code_metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert DocumentationData to a dictionary."""
        return {
            "module_name": self.module_name,
            "module_path": str(self.module_path),
            "module_summary": self.module_summary,
            "source_code": self.source_code,
            "docstring_data": (
                self.docstring_data.to_dict() if isinstance(self.docstring_data, DocstringData) else self.docstring_data
            ),
            "ai_content": self.ai_content,
            "code_metadata": self.code_metadata,
            "glossary": self.glossary,
            "changes": self.changes,
            "complexity_scores": self.complexity_scores,
            "metrics": self.metrics,
            "validation_status": self.validation_status,
            "validation_errors": self.validation_errors,
        }
