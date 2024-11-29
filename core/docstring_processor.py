"""
Core docstring processing module with integrated metrics and extraction capabilities.
"""

import ast
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from jsonschema import validate, ValidationError
from core.logger import LoggerSetup
from core.metrics import Metrics
from core.code_extraction import CodeExtractor, ExtractedFunction, ExtractedClass

logger = LoggerSetup.get_logger(__name__)

# Schemas
GOOGLE_STYLE_DOCSTRING_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "params": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["name", "type", "description"]
            }
        },
        "returns": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["type", "description"]
        },
        "raises": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "exception": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["exception", "description"]
            }
        }
    },
    "required": ["description", "params", "returns", "raises"]
}

EXTRACT_INFORMATION_TOOL = {
    "name": "extract_information",
    "description": "Extracts information from functions, classes, methods, and docstrings.",
    "strict": True,
    "parameters": {
        "type": "object",
        "required": [
            "source_code",
            "include_docstrings",
            "include_methods",
            "include_classes"
        ],
        "properties": {
            "source_code": {
                "type": "string",
                "description": "The source code from which to extract information."
            },
            "include_docstrings": {
                "type": "boolean",
                "description": "Flag to include/exclude docstrings in the extraction process."
            },
            "include_methods": {
                "type": "boolean",
                "description": "Flag to include/exclude methods in the extraction process."
            },
            "include_classes": {
                "type": "boolean",
                "description": "Flag to include classes in the extraction process."
            },
            "documentation_style": {
                "type": "string",
                "description": "The style of the documentation to extract.",
                "enum": ["google", "numpy", "rst"],
                "default": "google"
            },
            "output_format": {
                "type": "string",
                "description": "The desired format for the extracted information.",
                "enum": ["plain", "json", "markdown"],
                "default": "json"
            }
        },
        "additionalProperties": False
    }
}

# Docstring and Metrics Data Classes
@dataclass
class DocstringMetrics:
    """Metrics for evaluating the quality and complexity of docstrings."""
    length: int
    sections_count: int
    args_count: int
    cognitive_complexity: float
    completeness_score: float


@dataclass
class DocstringData:
    """Represents parsed docstring data with associated metrics."""
    summary: str
    description: str
    args: List[Dict[str, Optional[str]]]
    returns: Dict[str, Optional[str]]
    raises: Optional[List[Dict[str, Optional[str]]]] = None
    metrics: Optional[DocstringMetrics] = None
    extraction_context: Optional[Dict[str, Any]] = None


@dataclass
class DocumentationSection:
    """Represents a section of documentation."""
    title: str
    content: str
    subsections: Optional[List['DocumentationSection']] = None


class DocstringProcessor:
    """Processor for handling docstrings with integrated metrics and extraction capabilities."""

    def __init__(self) -> None:
        """Initialize the DocstringProcessor with a metrics calculator and code extractor."""
        self.metrics_calculator = Metrics()
        self.code_extractor = CodeExtractor()
        self.min_length: Dict[str, int] = {
            'summary': 10,
            'description': 10
        }

    def process_node(self, node: ast.AST, source_code: str) -> DocstringData:
        """
        Process an AST node to extract and analyze docstring information.

        Args:
            node (ast.AST): The AST node representing a function or class.
            source_code (str): The source code containing the node.

        Returns:
            DocstringData: The processed docstring data with metrics and context.
        """
        try:
            extraction_result = self.code_extractor.extract_code(source_code)
            
            extracted_info = None
            if isinstance(node, ast.ClassDef):
                extracted_info = next((c for c in extraction_result.classes if c.name == node.name), None)
            elif isinstance(node, ast.FunctionDef):
                extracted_info = next((f for f in extraction_result.functions if f.name == node.name), None)

            docstring_data = self.parse(ast.get_docstring(node) or '')
            docstring_data.extraction_context = self._convert_extracted_info(extracted_info)
            docstring_data.metrics = self._convert_to_docstring_metrics(extracted_info.metrics if extracted_info else {})
            
            return docstring_data
        except Exception as e:
            logger.error(f"Error processing node: {e}")
            return DocstringData("", "", [], {}, [])

    def _convert_extracted_info(self, extracted_info: Union[ExtractedFunction, ExtractedClass, None]) -> Dict[str, Any]:
        """
        Convert extracted information into a context dictionary.

        Args:
            extracted_info (Union[ExtractedFunction, ExtractedClass, None]): The extracted function or class information.

        Returns:
            Dict[str, Any]: A dictionary containing the context of the extracted information.
        """
        if not extracted_info:
            return {}
        
        if isinstance(extracted_info, ExtractedFunction):
            return {
                'type': 'function',
                'name': extracted_info.name,
                'args': [vars(arg) for arg in extracted_info.args],
                'returns': extracted_info.return_type,
                'is_method': extracted_info.is_method,
                'is_async': extracted_info.is_async,
                'metrics': extracted_info.metrics
            }
        elif isinstance(extracted_info, ExtractedClass):
            return {
                'type': 'class',
                'name': extracted_info.name,
                'bases': extracted_info.bases,
                'methods': [vars(method) for method in extracted_info.methods],
                'attributes': extracted_info.attributes,
                'metrics': extracted_info.metrics
            }
        return {}

    def _convert_to_docstring_metrics(self, metrics: Dict[str, Any]) -> DocstringMetrics:
        """
        Convert metrics from CodeExtractor to DocstringMetrics.

        Args:
            metrics (Dict[str, Any]): The metrics dictionary from CodeExtractor.

        Returns:
            DocstringMetrics: The converted docstring metrics.
        """
        return DocstringMetrics(
            length=metrics.get('total_lines', 0),
            sections_count=len(metrics.get('sections', [])),
            args_count=metrics.get('parameter_count', 0),
            cognitive_complexity=metrics.get('cognitive_complexity', 0.0),
            completeness_score=metrics.get('maintainability_index', 0.0) / 100
        )

    def parse(self, docstring: str) -> DocstringData:
        """
        Parse a raw docstring into a structured format.

        Args:
            docstring (str): The raw docstring text.

        Returns:
            DocstringData: The parsed docstring data.
        """
        try:
            if not docstring.strip():
                return DocstringData("", "", [], {}, [])

            # Example heuristic parsing (extendable with AI tools or regex)
            summary = docstring.split("\n\n")[0].strip()  # First paragraph is the summary
            description = "\n\n".join(docstring.split("\n\n")[1:]).strip()  # Rest is the description

            # Parse structured sections
            args, returns, raises = [], {}, []
            if "Args:" in docstring:
                args = self._parse_args_section(docstring)

            if "Returns:" in docstring:
                returns = self._parse_returns_section(docstring)

            if "Raises:" in docstring:
                raises = self._parse_raises_section(docstring)

            return DocstringData(summary, description, args, returns, raises)
        except Exception as e:
            logger.error(f"Error parsing docstring: {e}")
            return DocstringData("", "", [], {}, [])

    def format(self, docstring_data: DocstringData) -> str:
        """
        Format structured docstring data into a docstring string.

        Args:
            docstring_data (DocstringData): The structured docstring data.

        Returns:
            str: The formatted docstring.
        """
        docstring_lines = [docstring_data.summary, "", docstring_data.description]

        if docstring_data.args:
            docstring_lines.append("Args:")
            for arg in docstring_data.args:
                docstring_lines.append(
                    f"    {arg['name']} ({arg.get('type', 'Any')}): {arg.get('description', '')}"
                )

        if docstring_data.returns:
            docstring_lines.append("")
            docstring_lines.append("Returns:")
            docstring_lines.append(
                f"    {docstring_data.returns['type']}: {docstring_data.returns['description']}"
            )

        if docstring_data.raises:
            docstring_lines.append("")
            docstring_lines.append("Raises:")
            for exc in docstring_data.raises:
                docstring_lines.append(
                    f"    {exc['exception']}: {exc['description']}"
                )

        return "\n".join(docstring_lines)

    def insert(self, node: ast.AST, docstring: str) -> bool:
        """
        Insert a docstring into an AST node.

        Args:
            node (ast.AST): The AST node to insert the docstring into.
            docstring (str): The docstring to insert.

        Returns:
            bool: True if insertion was successful, False otherwise.
        """
        try:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                node.body.insert(0, docstring_node)  # Add docstring as the first body element
                return True
            else:
                logger.warning(f"Cannot insert docstring into node type: {type(node).__name__}")
                return False
        except Exception as e:
            logger.error(f"Error inserting docstring: {e}")
            return False

    def validate_docstring(self, docstring: str) -> bool:
        """
        Validate a docstring against the Google style schema.

        Args:
            docstring (str): The docstring to validate.

        Returns:
            bool: True if the docstring is valid, False otherwise.
        """
        try:
            docstring_data = self.parse(docstring)
            validate(instance=docstring_data, schema=GOOGLE_STYLE_DOCSTRING_SCHEMA)
            return True
        except ValidationError as e:
            logger.error(f"Docstring validation error: {e}")
            return False

    def generate_markdown(self, source_code: str) -> str:
        """
        Generate markdown documentation from source code.

        Args:
            source_code (str): The source code to document.

        Returns:
            str: The generated markdown documentation.
        """
        try:
            extraction_result = self.code_extractor.extract_code(source_code)
            sections = []

            if extraction_result.module_docstring:
                sections.append(DocumentationSection(
                    title="Module Docstring",
                    content=extraction_result.module_docstring
                ))

            for cls in extraction_result.classes:
                sections.append(DocumentationSection(
                    title=f"Class: {cls.name}",
                    content=self.format(self.parse(cls.docstring or ''))
                ))

            for func in extraction_result.functions:
                sections.append(DocumentationSection(
                    title=f"Function: {func.name}",
                    content=self.format(self.parse(func.docstring or ''))
                ))

            markdown_content = self._generate_markdown_content(sections)
            return markdown_content
        except Exception as e:
            logger.error(f"Error generating markdown: {e}")
            return ""

    def _generate_markdown_content(self, sections: List[DocumentationSection]) -> str:
        """
        Generate markdown content from documentation sections.

        Args:
            sections (List[DocumentationSection]): The documentation sections.

        Returns:
            str: The generated markdown content.
        """
        markdown_lines = []
        for section in sections:
            markdown_lines.append(f"# {section.title}\n")
            markdown_lines.append(f"{section.content}\n")
            if section.subsections:
                for subsection in section.subsections:
                    markdown_lines.append(f"## {subsection.title}\n")
                    markdown_lines.append(f"{subsection.content}\n")
        return "\n".join(markdown_lines)
