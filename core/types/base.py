"""Base type definitions for code extraction."""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Callable


class Injector:
    """Manages dependency injection for classes."""
    _dependencies: Dict[str, Any] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, name: str, dependency: Any) -> None:
        """Register a dependency with a name.

        Args:
            name: The name to register the dependency under
            dependency: The dependency instance to register
        """
        cls._dependencies[name] = dependency
        cls._initialized = True

    @classmethod
    def get(cls, name: str) -> Any:
        """Retrieve a dependency by name.

        Args:
            name: The name of the dependency to retrieve

        Returns:
            The registered dependency instance

        Raises:
            KeyError: If the dependency is not registered
        """
        if not cls._initialized:
            # Import here to avoid circular imports
            from core.metrics import Metrics
            from core.docstring_processor import DocstringProcessor

            # Register default dependencies
            if 'metrics_calculator' not in cls._dependencies:
                cls._dependencies['metrics_calculator'] = Metrics()
            if 'docstring_processor' not in cls._dependencies:
                cls._dependencies['docstring_processor'] = DocstringProcessor()
            cls._initialized = True

        if name not in cls._dependencies:
            raise KeyError(
                f"Dependency '{name}' not found. Available dependencies: {list(cls._dependencies.keys())}")
        return cls._dependencies[name]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered dependencies."""
        cls._dependencies.clear()
        cls._initialized = False


@dataclass
class MetricData:
    """Container for code metrics."""
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    maintainability_index: float = 0.0
    halstead_metrics: Dict[str, float] = field(default_factory=dict)
    lines_of_code: int = 0
    complexity_graph: Optional[str] = None
    total_functions: int = 0
    scanned_functions: int = 0
    total_classes: int = 0
    scanned_classes: int = 0

    @property
    def function_scan_ratio(self) -> float:
        """Calculate the ratio of successfully scanned functions."""
        return self.scanned_functions / self.total_functions if self.total_functions > 0 else 0.0

    @property
    def class_scan_ratio(self) -> float:
        """Calculate the ratio of successfully scanned classes."""
        return self.scanned_classes / self.total_classes if self.total_classes > 0 else 0.0


@dataclass
class BaseData:
    """Base class for data structures with common fields."""
    name: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedResponse:
    """Response from parsing operations."""
    content: Dict[str, Any]
    format_type: str
    parsing_time: float
    validation_success: bool
    errors: List[str]
    metadata: Dict[str, Any]


@dataclass
class DocstringData:
    """Google Style docstring representation."""
    summary: str  # First line brief description
    args: List[Dict[str, str]] = field(
        default_factory=list)  # param name, type, description
    returns: Dict[str, str] = field(default_factory=lambda: {
                                    "type": "None", "description": ""})  # type and description of return value
    raises: List[Dict[str, str]] = field(
        default_factory=list)  # exception type and description
    description: Optional[str] = None  # Detailed description
    metadata: Dict[str, Any] = field(default_factory=dict)
    complexity: Optional[int] = None
    validation_status: bool = False
    validation_errors: List[str] = field(default_factory=list)


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
    metrics: MetricData = field(default_factory=MetricData)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    decorators: List[str] = field(default_factory=list)
    complexity_warnings: List[str] = field(default_factory=list)
    ast_node: Optional[ast.AST] = None
    metric_calculator: Callable = None  # New field for dependency injection

    def __post_init__(self):
        """Initialize dependencies."""
        if self.metric_calculator is None:
            self.metric_calculator = Injector.get('metrics_calculator')
        if self.source:
            self.metrics = self.metric_calculator.calculate_metrics(
                self.source)


@dataclass
class ExtractedFunction(ExtractedElement):
    """Represents an extracted function with its metadata."""
    args: List[ExtractedArgument] = field(default_factory=list)
    returns: Dict[str, str] = field(default_factory=lambda: {
                                    "type": "Any", "description": ""})
    raises: List[Dict[str, str]] = field(default_factory=list)
    body_summary: Optional[str] = None
    docstring_info: Optional[DocstringData] = None
    is_async: bool = False
    is_method: bool = False
    parent_class: Optional[str] = None
    docstring_parser: Callable = None  # New field for dependency injection

    def __post_init__(self):
        """Initialize dependencies."""
        super().__post_init__()
        if self.docstring_parser is None:
            self.docstring_parser = Injector.get('docstring_parser')
        self.docstring_info = self.docstring_parser(self.docstring)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ExtractedFunction to a dictionary.

        Returns:
            Dictionary representation of the function
        """
        return {
            'name': self.name,
            'lineno': self.lineno,
            'source': self.source,
            'docstring': self.docstring,
            'metrics': self.metrics.__dict__,
            'dependencies': {k: list(v) for k, v in self.dependencies.items()},
            'decorators': self.decorators,
            'complexity_warnings': self.complexity_warnings,
            'args': [{'name': a.name, 'type': a.type, 'default_value': a.default_value, 'is_required': a.is_required}
                     for a in self.args],
            'returns': self.returns,
            'raises': self.raises,
            'body_summary': self.body_summary,
            'docstring_info': self.docstring_info.__dict__ if self.docstring_info else None,
            'is_async': self.is_async,
            'is_method': self.is_method,
            'parent_class': self.parent_class
        }


@dataclass
class ExtractedClass(ExtractedElement):
    """Represents a class extracted from code."""
    methods: List[ExtractedFunction] = field(default_factory=list)
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    instance_attributes: List[Dict[str, Any]] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)
    metaclass: Optional[str] = None
    is_exception: bool = False
    docstring_parser: Callable = None  # New field for dependency injection
    docstring_info: Optional[DocstringData] = None  # Add missing field

    def __post_init__(self):
        """Initialize dependencies."""
        super().__post_init__()
        if self.docstring_parser is None:
            self.docstring_parser = Injector.get('docstring_parser')
        if self.docstring_info is None:  # Only parse if not provided
            self.docstring_info = self.docstring_parser(self.docstring)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ExtractedClass to a dictionary.

        Returns:
            Dictionary representation of the class
        """
        # Handle dependencies conversion safely
        deps = {}
        for k, v in self.dependencies.items():
            if isinstance(v, (list, set)):
                deps[k] = list(v)
            elif isinstance(v, (int, float)):
                deps[k] = [v]
            else:
                deps[k] = []

        return {
            'name': self.name,
            'lineno': self.lineno,
            'source': self.source,
            'docstring': self.docstring,
            'metrics': self.metrics.__dict__,
            'dependencies': deps,
            'decorators': self.decorators,
            'complexity_warnings': self.complexity_warnings,
            'methods': [m.to_dict() for m in self.methods],
            'attributes': self.attributes,
            'instance_attributes': self.instance_attributes,
            'bases': self.bases,
            'metaclass': self.metaclass,
            'is_exception': self.is_exception,
            'docstring_info': self.docstring_info.__dict__ if self.docstring_info else None
        }


@dataclass
class ExtractionResult:
    """Result of code extraction process."""
    module_docstring: Dict[str, Any] = field(default_factory=dict)
    module_name: str = ""
    file_path: str = ""
    classes: List[ExtractedClass] = field(default_factory=list)
    functions: List[ExtractedFunction] = field(default_factory=list)
    variables: List[Dict[str, Any]] = field(default_factory=list)
    constants: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    maintainability_index: Optional[float] = None
    source_code: str = ""
    imports: List[Any] = field(default_factory=list)
    metrics: MetricData = field(default_factory=MetricData)
    metric_calculator: Callable = None  # New field for dependency injection

    def __post_init__(self):
        """Initialize dependencies."""
        if self.metric_calculator is None:
            self.metric_calculator = Injector.get('metrics_calculator')
        if hasattr(self.metric_calculator, 'calculate_metrics'):
            self.metrics = self.metric_calculator.calculate_metrics(
                self.source_code)


@dataclass
class ProcessingResult:
    """Result of AI processing operation."""
    content: Dict[str, Any]
    usage: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    is_cached: bool = False
    processing_time: float = 0.0
    validation_status: bool = False
    validation_errors: List[str] = field(default_factory=list)
    schema_errors: List[str] = field(default_factory=list)


@dataclass
class DocumentationContext:
    """Context for documentation generation."""
    source_code: str
    module_path: Path
    include_source: bool = True
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    ai_generated: Optional[Dict[str, Any]] = field(default_factory=dict)
    classes: Optional[List[ExtractedClass]] = field(default_factory=list)
    functions: Optional[List[ExtractedFunction]] = field(default_factory=list)
    constants: Optional[List[Any]] = field(default_factory=list)
    changes: Optional[List[Any]] = field(default_factory=list)

    def get_cache_key(self) -> str:
        """Generate cache key."""
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
    tree: Optional[ast.AST] = None
    function_extractor: Any = None  # Type will be set at runtime
    class_extractor: Any = None  # Type will be set at runtime
    dependency_analyzer: Any = None  # Type will be set at runtime

    def __post_init__(self) -> None:
        """Initialize AST if needed."""
        if self.tree is None and self.source_code:
            try:
                self.source_code = self._fix_indentation(self.source_code)
                self.tree = ast.parse(self.source_code)
            except SyntaxError as e:
                raise ValueError(f"Failed to parse source code: {e}")

        if self.source_code is None and self.tree is not None:
            try:
                if hasattr(ast, "unparse"):
                    self.source_code = ast.unparse(self.tree)
            except Exception as e:
                raise ValueError(f"Failed to unparse AST: {e}")

    def _fix_indentation(self, code: str) -> str:
        """Fix inconsistent indentation in the source code."""
        lines = code.splitlines()
        fixed_lines = []
        for line in lines:
            fixed_lines.append(line.replace('\t', '    '))
        return '\n'.join(fixed_lines)


@dataclass
class DocumentationData:
    """Documentation data structure."""
    module_name: str
    module_path: Path
    module_summary: str
    source_code: str
    docstring_data: DocstringData
    ai_content: Dict[str, Any]
    code_metadata: Dict[str, Any]
    glossary: Dict[str, Dict[str, str]] = field(default_factory=dict)
    changes: List[Dict[str, Any]] = field(default_factory=list)
    complexity_scores: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    validation_status: bool = False
    validation_errors: List[str] = field(default_factory=list)
    docstring_parser: Callable = None  # New field for dependency injection

    def __post_init__(self):
        """Initialize dependencies."""
        if self.docstring_parser is None:
            self.docstring_parser = Injector.get('docstring_parser')
        self.docstring_data = self.docstring_parser(self.source_code)

        # Ensure module summary is never None
        if not self.module_summary:
            self.module_summary = (
                self.ai_content.get('summary') or
                self.docstring_data.summary or
                "No module summary available."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert DocumentationData to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the documentation data
        """
        return {
            'module_name': self.module_name,
            'module_path': str(self.module_path),
            'module_summary': self.module_summary,
            'source_code': self.source_code,
            'docstring_data': {
                'summary': self.docstring_data.summary,
                'description': self.docstring_data.description,
                'args': self.docstring_data.args,
                'returns': self.docstring_data.returns,
                'raises': self.docstring_data.raises,
                'complexity': self.docstring_data.complexity,
                'validation_status': self.docstring_data.validation_status,
                'validation_errors': self.docstring_data.validation_errors
            },
            'ai_content': self.ai_content,
            'code_metadata': self.code_metadata,
            'glossary': self.glossary,
            'changes': self.changes,
            'complexity_scores': self.complexity_scores,
            'metrics': self.metrics,
            'validation_status': self.validation_status,
            'validation_errors': self.validation_errors
        }
