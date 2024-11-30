"""
Docstring Processing Module

This module handles the parsing, validation, and formatting of docstrings.
It integrates metrics calculation and code extraction to provide a comprehensive
analysis of docstrings within Python code.
"""

import ast
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from core.logger import LoggerSetup
from core.metrics import Metrics
from core.code_extraction import CodeExtractor, ExtractedFunction, ExtractedClass

logger = LoggerSetup.get_logger(__name__)

@dataclass
class DocstringData:
    """Structured representation of a docstring."""
    summary: str
    description: str
    args: List[Dict[str, Any]]
    returns: Dict[str, Any]
    raises: List[Dict[str, Any]]
    metrics: Optional['DocstringMetrics'] = None
    extraction_context: Optional[Dict[str, Any]] = None
    complexity: Optional[int] = None

    def set_complexity(self, score: int) -> None:
        """Set the complexity score for the docstring."""
        self.complexity = score

@dataclass
class DocstringMetrics:
    """Metrics related to a docstring."""
    length: int
    sections_count: int
    args_count: int
    cognitive_complexity: float
    completeness_score: float

@dataclass
class DocumentationSection:
    """Represents a section of documentation."""
    title: str
    content: str
    subsections: Optional[List['DocumentationSection']] = field(default_factory=list)
    source_code: Optional[str] = None
    tables: Optional[List[str]] = field(default_factory=list)

class DocstringProcessor:
    """
    Processes docstrings by parsing, validating, and formatting them.
    Integrates metrics calculation and code extraction.
    """

    def __init__(self, min_length: Optional[Dict[str, int]] = None) -> None:
        """
        Initialize DocstringProcessor with optional minimum length requirements.

        Args:
            min_length (Optional[Dict[str, int]]): Minimum length requirements for docstring sections.
        """
        self.min_length = min_length or {
            'summary': 10,
            'description': 20
        }
        self.metrics_calculator = Metrics()
        self.code_extractor = CodeExtractor()

    def extract(self, node: ast.AST, source_code: str) -> DocstringData:
        """
        Extract and process the docstring from an AST node, including metrics and extraction context.

        Args:
            node (ast.AST): The AST node to extract the docstring from.
            source_code (str): The source code containing the node.

        Returns:
            DocstringData: The extracted and processed docstring data.
        """
        try:
            raw_docstring = ast.get_docstring(node) or ""
            docstring_data = self.parse(raw_docstring)
            extraction_result = self.code_extractor.extract_code(source_code)
            extracted_info = self._get_extracted_info(node, extraction_result)
            docstring_data.extraction_context = self._convert_extracted_info(extracted_info)
            docstring_data.metrics = self._convert_to_docstring_metrics(extracted_info.metrics if extracted_info else {})
            return docstring_data
        except Exception as e:
            logger.error(f"Error processing node: {e}")
            return DocstringData("", "", [], {}, [])

    def _get_extracted_info(self, node: ast.AST, extraction_result: Any) -> Union[ExtractedFunction, ExtractedClass, None]:
        """
        Retrieve extracted information for a given node.

        Args:
            node (ast.AST): The AST node for which to retrieve information.
            extraction_result (Any): The result of code extraction.

        Returns:
            Union[ExtractedFunction, ExtractedClass, None]: The extracted information or None if not found.
        """
        if isinstance(node, ast.ClassDef):
            return next((c for c in extraction_result.classes if c.name == node.name), None)
        elif isinstance(node, ast.FunctionDef):
            return next((f for f in extraction_result.functions if f.name == node.name), None)
        return None

    def _convert_extracted_info(self, extracted_info: Union[ExtractedFunction, ExtractedClass, None]) -> Dict[str, Any]:
        """
        Convert extracted information into a context dictionary.

        Args:
            extracted_info (Union[ExtractedFunction, ExtractedClass, None]): The extracted information.

        Returns:
            Dict[str, Any]: A dictionary representing the extracted context.
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
            metrics (Dict[str, Any]): The metrics to convert.

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

    def parse(self, docstring: str, style: str = 'google') -> DocstringData:
        """
        Parse a raw docstring into a structured format.

        Args:
            docstring (str): The raw docstring text.
            style (str): The docstring style to parse ('google', 'numpy', 'sphinx').

        Returns:
            DocstringData: The parsed docstring data.
        """
        try:
            if not docstring.strip():
                return DocstringData("", "", [], {}, [])

            # Use a third-party library like docstring_parser
            from docstring_parser import parse

            parsed = parse(docstring, style=style)

            args = [
                {
                    'name': param.arg_name,
                    'type': param.type_name or 'Any',
                    'description': param.description or ''
                }
                for param in parsed.params
            ]

            returns = {
                'type': parsed.returns.type_name if parsed.returns else 'Any',
                'description': parsed.returns.description if parsed.returns else ''
            }

            raises = [
                {
                    'exception': e.type_name or 'Exception',
                    'description': e.description or ''
                }
                for e in parsed.raises
            ] if parsed.raises else []

            return DocstringData(
                summary=parsed.short_description or '',
                description=parsed.long_description or '',
                args=args,
                returns=returns,
                raises=raises
            )
        except Exception as e:
            logger.error(f"Error parsing docstring: {e}")
            return DocstringData("", "", [], {}, [])

    def validate(self, docstring_data: DocstringData) -> Tuple[bool, List[str]]:
        """
        Validate the structured docstring data.

        Args:
            docstring_data (DocstringData): The docstring data to validate.

        Returns:
            Tuple[bool, List[str]]: A tuple containing a boolean indicating validation success,
            and a list of error messages if validation fails.
        """
        errors = []
        if len(docstring_data.summary) < self.min_length['summary']:
            errors.append("Summary is too short.")
        if len(docstring_data.description) < self.min_length['description']:
            errors.append("Description is too short.")
        if not docstring_data.args:
            errors.append("Arguments section is missing.")
        if not docstring_data.returns:
            errors.append("Returns section is missing.")
        is_valid = not errors
        return is_valid, errors

    def format(self, docstring_data: DocstringData, complexity_score: Optional[int] = None) -> str:
        """
        Format structured docstring data into a formatted docstring string.

        Args:
            docstring_data (DocstringData): The structured docstring data to format.
            complexity_score (Optional[int]): An optional complexity score to include in the docstring.

        Returns:
            str: The formatted docstring.
        """
        docstring_lines = []

        # Add summary if present
        if docstring_data.summary:
            docstring_lines.append(docstring_data.summary)
            docstring_lines.append("")

        # Add description
        if docstring_data.description:
            docstring_lines.append(docstring_data.description)
            docstring_lines.append("")

        # Add arguments section
        if docstring_data.args:
            docstring_lines.append("Args:")
            for arg in docstring_data.args:
                docstring_lines.append(
                    f"    {arg['name']} ({arg.get('type', 'Any')}): {arg.get('description', '')}"
                )
            docstring_lines.append("")

        # Add returns section
        if docstring_data.returns:
            docstring_lines.append("Returns:")
            docstring_lines.append(
                f"    {docstring_data.returns.get('type', 'Any')}: "
                f"{docstring_data.returns.get('description', '')}"
            )
            docstring_lines.append("")

        # Add raises section
        if docstring_data.raises:
            docstring_lines.append("Raises:")
            for exc in docstring_data.raises:
                docstring_lines.append(
                    f"    {exc.get('exception', 'Exception')}: {exc.get('description', '')}"
                )
            docstring_lines.append("")

        # Add complexity score if provided
        if complexity_score is not None:
            warning = " ⚠️" if complexity_score > 10 else ""
            docstring_lines.append(f"Complexity Score: {complexity_score}{warning}")

        return "\n".join(docstring_lines).strip()

    def _format_module_docstring(self, docstring_data: DocstringData, complexity_metrics: Dict[str, Any]) -> str:
        """
        Format module-level docstring with complexity scores.

        Args:
            docstring_data (DocstringData): The structured docstring data for the module.
            complexity_metrics (Dict[str, Any]): Complexity metrics to include in the docstring.

        Returns:
            str: The formatted module-level docstring.
        """
        docstring_lines = []

        # Add summary and description
        if docstring_data.summary:
            docstring_lines.extend([docstring_data.summary, ""])
        if docstring_data.description:
            docstring_lines.extend([docstring_data.description, ""])

        # Add complexity scores section if metrics are provided
        if complexity_metrics:
            docstring_lines.append("Complexity Scores:")
            for name, score in complexity_metrics.items():
                warning = " ⚠️" if score > 10 else ""
                docstring_lines.append(f"    {name}: {score}{warning}")

        return "\n".join(docstring_lines).strip()

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