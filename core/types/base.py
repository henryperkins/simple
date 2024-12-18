"""Base type definitions for code extraction."""

from dataclasses import dataclass, field
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
    args: List[Dict[str, Any]] = Field(default_factory=list)
    returns: Dict[str, str] = Field(default_factory=lambda: {"type": "Any", "description": "No return value documented."})
    raises: List[Dict[str, str]] = Field(default_factory=list)

    def validate_returns(self, self_param: Any, v: Dict[str, str]) -> Dict[str, str]:
        """Validate returns field."""
        if 'type' not in v or 'description' not in v:
            raise ValueError("Returns must contain 'type' and 'description'")
        return v


@dataclass
class DocumentationContext:
    """Context for documentation generation operations."""
    source_code: str
    module_path: Path | None = None
    include_source: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    classes: list[dict[str, Any]] = field(default_factory=list)
    functions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate required fields and store source code."""
        if not self.source_code or not self.source_code.strip():
            raise ValueError("source_code is required and cannot be empty")
        self.metadata.setdefault("source_code", self.source_code)


@dataclass
class ExtractionContext:
    """Context for code extraction operations."""
    _source_code: str | None = None  # Default value for when setting
    module_name: str | None = None
    base_path: Path | None = None
    include_private: bool = False
    include_nested: bool = False
    include_magic: bool = True  # Controls whether magic methods are included
    tree: ast.AST | None = None  # AST tree if already parsed
    _dependency_analyzer: DependencyAnalyzer | None = None  # Internal storage for lazy initialization
    function_extractor: Any | None = None  # Add this line
    docstring_processor: Any | None = None  # Add this line
    logger: CorrelationLoggerAdapter = field(default_factory=lambda: CorrelationLoggerAdapter(LoggerSetup.get_logger(__name__)))
    metrics_collector: Any | None = None  # Add this line
    strict_mode: bool = False  # Add this line
    config: Dict[str, Any] = field(default_factory=dict)  # Add this line

    def get_source_code(self) -> str | None:
        """Get the source code of this instance"""
        return self._source_code

    def set_source_code(self, value: str, source: str | None = None) -> None:
        """Set the source code with logging and validation"""
        if not value or not value.strip():
            raise ValueError(f"Source code cannot be empty or null for {source}")
        self._source_code = value
        self.logger.debug(f"Updated source code in context {type(self)}: {value[:50]}...")

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
    source_code: str
    module_docstring: dict[str, Any] = field(default_factory=dict)
    classes: list[dict[str, Any]] = field(default_factory=list)
    functions: list[dict[str, Any]] = field(default_factory=list)
    variables: list[dict[str, Any]] = field(default_factory=list)
    constants: list[dict[str, Any]] = field(default_factory=list)
    dependencies: dict[str, set[str]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    module_name: str = ""
    file_path: str = ""


@dataclass
class ProcessingResult:
    """Represents the result of a processing operation."""
    content: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    validation_status: bool = False
    validation_errors: list[str] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)  # Fixed from default_factory.list


@dataclass
class MetricData:
    """Holds data for code metrics analysis."""
    module_name: str = ""
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    maintainability_index: float = 0.0
    halstead_metrics: dict[str, Any] = field(default_factory=dict)  # Fixed from default_factory.dict
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
    type: Optional[str] = None
    default_value: Optional[str] = None
    is_required: bool = True
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "type": self.type or "Any",
            "default_value": self.default_value,
            "is_required": self.is_required,
            "description": self.description or "",
        }


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
    complexity_warnings: list[str] = field(default_factory=list)  # Fixed from default_factory.list
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
class ExtractedFunction:
    """Represents an extracted function with its metadata."""

    name: str
    lineno: int
    source: Optional[str] = None
    docstring: Optional[str] = None
    args: List[ExtractedArgument] = field(default_factory=list)
    returns: Optional[Dict[str, str]] = None
    raises: List[Dict[str, str]] = field(default_factory=list)
    body_summary: Optional[str] = None
    is_async: bool = False
    is_method: bool = False
    parent_class: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    decorators: List[str] = field(default_factory=list)  # New field for decorators
    ast_node: Optional[ast.AST] = None  # Add ast_node field
    dependencies: Dict[str, set[str]] = field(default_factory=dict)  # Add dependencies field
    complexity_warnings: List[str] = field(default_factory=list)  # Add complexity_warnings field


    def to_dict(self) -> Dict[str, Any]:
        """Convert the ExtractedFunction instance to a dictionary."""
        return {
            "name": self.name,
            "lineno": self.lineno,
            "source": self.source,
            "docstring": self.docstring,
            "args": [arg.to_dict() for arg in self.args],
            "returns": self.returns,
            "raises": self.raises,
            "body_summary": self.body_summary,
            "is_async": self.is_async,
            "is_method": self.is_method,
            "parent_class": self.parent_class,
            "metrics": self.metrics,
            "decorators": self.decorators,  # Include decorators in the dictionary
            "ast_node": None,  # Exclude ast_node from serialization
            "dependencies": self.dependencies,  # Include dependencies in the dictionary
            "complexity_warnings": self.complexity_warnings,  # Include complexity_warnings in the dictionary
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedFunction":
        """Create an ExtractedFunction instance from a dictionary."""
        return cls(
            name=data.get("name", ""),
            lineno=data.get("lineno", 0),
            source=data.get("source"),
            docstring=data.get("docstring"),
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
            metrics=data.get("metrics", {}),
            decorators=data.get("decorators", []),  # Handle decorators
            ast_node=None,  # Ignore ast_node when creating from a dictionary
            dependencies=data.get("dependencies", {}),  # Handle dependencies
            complexity_warnings=data.get(
                "complexity_warnings", []
            ),  # Handle complexity_warnings
        )

    def get_docstring_info(self) -> Optional[DocstringData]:
        """Retrieve or parse the docstring information."""
        if self.docstring:
            # Simulate parsing the docstring into DocstringData
            return DocstringData(
                summary=self.docstring.split("\n")[0],
                description=self.docstring,
                args=[arg.to_dict() for arg in self.args],
                returns=self.returns or {"type": "Any", "description": ""},
                raises=self.raises,
                complexity=1,
            )
        return None


@dataclass
class ExtractedClass:
    """Represents a class extracted from code."""

    name: str
    lineno: int
    source: Optional[str] = None
    docstring: Optional[str] = None
    methods: List[ExtractedFunction] = field(default_factory=list)
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    instance_attributes: List[Dict[str, Any]] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)
    metaclass: Optional[str] = None
    is_exception: bool = False
    docstring_info: Optional[DocstringData] = None
    is_dataclass: bool = False
    is_abstract: bool = False
    abstract_methods: List[str] = field(default_factory=list)
    property_methods: List[Dict[str, Any]] = field(default_factory=list)
    class_variables: List[Dict[str, Any]] = field(default_factory=list)
    method_groups: Dict[str, List[str]] = field(default_factory=dict)
    inheritance_chain: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    decorators: List[str] = field(default_factory=list)  # New field for decorators
    ast_node: Optional[ast.AST] = None  # Add ast_node field
    dependencies: Dict[str, set[str]] = field(default_factory=dict)  # Add dependencies field
    complexity_warnings: List[str] = field(default_factory=list)  # Add complexity_warnings field


    def to_dict(self) -> Dict[str, Any]:
        """Convert the ExtractedClass instance to a dictionary."""
        return {
            "name": self.name,
            "lineno": self.lineno,
            "source": self.source,
            "docstring": self.docstring,
            "methods": [method.to_dict() for method in self.methods],
            "attributes": self.attributes,
            "instance_attributes": self.instance_attributes,
            "bases": self.bases,
            "metaclass": self.metaclass,
            "is_exception": self.is_exception,
            "docstring_info": (
                self.docstring_info.to_dict() if self.docstring_info else None
            ),
            "is_dataclass": self.is_dataclass,
            "is_abstract": self.is_abstract,
            "abstract_methods": self.abstract_methods,
            "property_methods": self.property_methods,
            "class_variables": self.class_variables,
            "method_groups": self.method_groups,
            "inheritance_chain": self.inheritance_chain,
            "metrics": self.metrics,
            "decorators": self.decorators,  # Include decorators in the dictionary
            "ast_node": None,  # Exclude ast_node from serialization
            "dependencies": self.dependencies,  # Include dependencies in the dictionary
            "complexity_warnings": self.complexity_warnings,  # Include complexity_warnings in the dictionary
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedClass":
        """Create an ExtractedClass instance from a dictionary."""
        return cls(
            name=data.get("name", ""),
            lineno=data.get("lineno", 0),
            source=data.get("source"),
            docstring=data.get("docstring"),
            methods=[
                ExtractedFunction.from_dict(m) if isinstance(m, dict) else m
                for m in data.get("methods", [])
            ],
            attributes=data.get("attributes", []),
            instance_attributes=data.get("instance_attributes", []),
            bases=data.get("bases", []),
            metaclass=data.get("metaclass"),
            is_exception=data.get("is_exception", False),
            docstring_info=(
                DocstringData.from_dict(data["docstring_info"])
                if data.get("docstring_info")
                else None
            ),
            is_dataclass=data.get("is_dataclass", False),
            is_abstract=data.get("is_abstract", False),
            abstract_methods=data.get("abstract_methods", []),
            property_methods=data.get("property_methods", []),
            class_variables=data.get("class_variables", []),
            method_groups=data.get("method_groups", {}),
            inheritance_chain=data.get("inheritance_chain", []),
            metrics=data.get("metrics", {}),
            decorators=data.get("decorators", []),  # Handle decorators
            ast_node=None,  # Ignore ast_node when creating from a dictionary
            dependencies=data.get("dependencies", {}),  # Handle dependencies
            complexity_warnings=data.get("complexity_warnings", []),  # Handle complexity_warnings

        )


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
    glossary: dict[str, dict[str, str]] = field(default_factory=dict)  # Fixed from default_factory.dict
    changes: list[dict[str, Any]] = field(default_factory=list)  # Fixed from default_factory.list
    complexity_scores: dict[str, float] = field(default_factory=dict)  # Fixed from default_factory.dict
    metrics: dict[str, Any] = field(default_factory=dict)  # Fixed from default_factory.dict
    validation_status: bool = False
    validation_errors: list[str] = field(default_factory=list)  # Fixed from default_factory.list
    docstring_parser: Callable[[str], DocstringData] | None = None
    metric_calculator: Callable[[str], dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        """Initialize dependencies."""
        from core.dependency_injection import Injector  # Import here to avoid circular imports

        if self.docstring_parser is None:
            self.docstring_parser = Injector.get("docstring_processor")
        if self.metric_calculator is None:
            self.metric_calculator = Injector.get("metrics_calculator")

        # Convert dict to DocstringData if needed
        if isinstance(self.docstring_data, dict):
            docstring_dict = self.docstring_data.copy()
            docstring_dict.pop('source_code', None)  # Remove source_code if present
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
        
        if not self.source_code or not self.source_code.strip():
            raise ValueError("source_code is required and cannot be empty")
        
        if not isinstance(self.code_metadata, dict):
            self.code_metadata = {}

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
