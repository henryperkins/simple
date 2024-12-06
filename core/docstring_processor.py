"""
Module for processing and validating docstrings.

This module provides utilities for parsing, validating, and formatting docstrings in Python code.
It supports parsing docstrings from various formats, validating them against a schema, integrating
them into source code, and generating consolidated documentation.

Functionality includes:
- Parsing raw docstrings or docstring dictionaries into structured data.
- Validating docstrings against a predefined schema.
- Inserting or replacing docstrings in abstract syntax trees (ASTs) of source code.
- Generating updated source code from modified ASTs.
- Formatting docstrings in a consistent manner.
- Generating consolidated documentation from docstring entries.

"""

import ast
import json
import sys
from typing import Optional, Dict, Any, List, Union, Tuple

from docstring_parser import parse as parse_docstring

from core.logger import LoggerSetup
from core.metrics import Metrics
from core.types import DocstringData
from exceptions import DocumentationError
from core.utils import FormattingUtils, ValidationUtils
from core.schema_loader import load_schema

try:
    import astor
except ImportError as e:
    raise ImportError(
        "The 'astor' library is required for Python versions < 3.9 to generate code from AST. "
        "Please install it using 'pip install astor'."
    ) from e


class DocstringProcessor:
    """
    Processes docstrings by parsing and validating them.

    This class provides methods to parse raw docstrings, validate them,
    integrate them into source code, and generate documentation.

    Attributes:
        logger (Logger): Logger instance for logging.
        metrics (Metrics): Metrics instance for performance tracking.
        docstring_schema (Dict[str, Any]): Schema for validating docstrings.

    """

    def __init__(self, metrics: Optional[Metrics] = None) -> None:
        """
        Initialize the DocstringProcessor.

        Args:
            metrics (Optional[Metrics]): An optional Metrics instance for tracking performance.

        """
        self.logger = LoggerSetup.get_logger(__name__)
        self.metrics = metrics or Metrics()
        self.docstring_schema = load_schema("docstring_schema")

    def parse(self, docstring: Union[Dict[str, Any], str]) -> DocstringData:
        """
        Parse a raw docstring into structured format.

        Args:
            docstring (Union[Dict[str, Any], str]): The raw docstring to parse.

        Returns:
            DocstringData: The parsed docstring encapsulated in a DocstringData object.

        Raises:
            DocumentationError: If the docstring is invalid or cannot be parsed.

        """
        try:
            if isinstance(docstring, dict):
                # Validate dictionary structure
                self._validate_docstring_dict(docstring)
                # Create DocstringData from dictionary
                return self._create_docstring_data_from_dict(docstring)
            elif isinstance(docstring, str):
                # Try parsing as JSON first
                docstring_str = docstring.strip()
                if docstring_str.startswith('{') and docstring_str.endswith('}'):
                    try:
                        doc_dict = json.loads(docstring_str)
                        return self.parse(doc_dict)
                    except json.JSONDecodeError as e:
                        self.logger.error("JSON parsing error: %s", e)
                        raise DocumentationError(
                            f"Invalid JSON format in docstring: {e}"
                        ) from e
                # Fall back to standard docstring parsing
                try:
                    parsed = parse_docstring(docstring_str)
                    return self._create_docstring_data_from_parsed(parsed)
                except Exception as e:
                    self.logger.error("Docstring parsing error: %s", e)
                    raise DocumentationError(f"Failed to parse docstring: {e}") from e
            raise DocumentationError("Docstring must be either a dictionary or a string.")
        except DocumentationError:
            # Re-raise DocumentationError without modification
            raise
        except Exception as e:
            self.logger.error("Unexpected error in parse method: %s", e)
            raise DocumentationError(f"Unexpected error during parsing: {e}") from e

    def _validate_docstring_dict(self, docstring_dict: Dict[str, Any]) -> None:
        """
        Validate the structure of the docstring dictionary.

        Args:
            docstring_dict (Dict[str, Any]): The docstring dictionary to validate.

        Raises:
            DocumentationError: If required keys are missing in the docstring.

        """
        required_keys = {'summary', 'description', 'args', 'returns', 'raises', 'complexity'}
        missing_keys = required_keys - docstring_dict.keys()
        if missing_keys:
            raise DocumentationError(f"Docstring dictionary missing keys: {missing_keys}")

    def _create_docstring_data_from_dict(self, docstring_dict: Dict[str, Any]) -> DocstringData:
        """Create DocstringData from a validated dictionary."""
        try:
            # Ensure empty lists for args and raises if not present
            docstring_dict['args'] = docstring_dict.get('args', [])
            docstring_dict['raises'] = docstring_dict.get('raises', [])

            # Ensure 'returns' is a dictionary with required fields
            returns = docstring_dict.get('returns', {})
            if not isinstance(returns, dict):
                returns = {'type': 'Any', 'description': ''}
            if not returns.get('type'):
                returns['type'] = 'Any'
            if not returns.get('description'):
                returns['description'] = ''

            # Validate with more lenient requirements
            is_valid, errors = ValidationUtils.validate_docstring(docstring_dict, self.docstring_schema)
            if not is_valid:
                # Log errors but don't fail if only missing optional fields
                self.logger.warning(f"Docstring validation warnings: {errors}")

            return DocstringData(
                summary=docstring_dict.get('summary', ''),
                description=docstring_dict.get('description', ''),
                args=docstring_dict['args'],
                returns=returns,
                raises=docstring_dict['raises'],
                complexity=docstring_dict.get('complexity', 1)
            )
        except Exception as e:
            self.logger.error("Error creating DocstringData from dict: %s", e)
            raise DocumentationError(f"Failed to create DocstringData from dict: {e}") from e

    def _create_docstring_data_from_parsed(self, parsed_docstring) -> DocstringData:
        """
        Create DocstringData from parsed docstring object.

        Args:
            parsed_docstring: The parsed docstring object from docstring_parser.

        Returns:
            DocstringData: The docstring data encapsulated in a DocstringData object.

        Raises:
            DocumentationError: If data cannot be created from the parsed docstring.

        """
        try:
            return DocstringData(
                summary=parsed_docstring.short_description or '',
                description=parsed_docstring.long_description or '',
                args=[{
                    'name': param.arg_name or 'unknown',
                    'type': param.type_name or 'Any',
                    'description': param.description or ''
                } for param in parsed_docstring.params],
                returns={
                    'type': parsed_docstring.returns.type_name if parsed_docstring.returns else 'Any',
                    'description': parsed_docstring.returns.description if parsed_docstring.returns else ''
                },
                raises=[{
                    'exception': exc.type_name or 'Exception',
                    'description': exc.description or ''
                } for exc in (parsed_docstring.raises or [])],
                complexity=1  # Default complexity as it may not be available
            )
        except Exception as e:
            self.logger.error("Error creating DocstringData: %s", e)
            raise DocumentationError(f"Failed to create DocstringData: {e}") from e

    def process_batch(self, doc_entries: List[Dict[str, Any]], source_code: str) -> Dict[str, Any]:
        """Process a batch of documentation entries and integrate them into the source code."""
        try:
            # Parse the source code into an AST
            tree = ast.parse(source_code)

            # Print doc_entries for debugging
            self.logger.debug(f"Doc entries received: {doc_entries}")

            # Ensure each entry has required fields
            processed_entries = []
            for entry in doc_entries:
                # If entry is a single docstring dict, wrap it with required fields
                if 'summary' in entry and 'name' not in entry:
                    # Try to extract name from AST
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                            processed_entries.append({
                                'name': node.name,
                                'docstring': self.format(DocstringData(**entry)),
                                'type': type(node).__name__
                            })
                            break
                else:
                    processed_entries.append(entry)

            # Insert or replace docstrings
            tree = self._insert_docstrings(tree, processed_entries)

            # Generate the updated source code from the AST
            updated_code = self._generate_code_from_ast(tree)

            # Generate the consolidated documentation
            documentation = self._generate_documentation(processed_entries)

            return {'code': updated_code, 'documentation': documentation}
        except Exception as e:
            self.logger.error("Unexpected error processing batch: %s", e, exc_info=True)
            raise DocumentationError(f"Failed to process batch: {e}") from e

    def _insert_docstrings(self, tree: ast.AST, doc_entries: List[Dict[str, Any]]) -> ast.AST:
        """Insert docstrings into the AST based on doc_entries."""
        # Create a mapping from name to docstring with better error handling
        docstring_map = {}
        for entry in doc_entries:
            if isinstance(entry, dict):
                name = entry.get('name')
                if name:
                    docstring_map[name] = entry.get('docstring', 'No docstring provided')
                else:
                    self.logger.warning(f"Entry missing name field: {entry}")

        self.logger.debug(f"Created docstring map: {docstring_map}")

        class DocstringInserter(ast.NodeTransformer):
            """An AST node transformer to insert docstrings into function, class, and module nodes."""

            def visit_FunctionDef(self, node):
                """Visit function definitions to insert docstrings."""
                self.generic_visit(node)
                self._insert_docstring(node)
                return node

            def visit_AsyncFunctionDef(self, node):
                """Visit async function definitions to insert docstrings."""
                self.generic_visit(node)
                self._insert_docstring(node)
                return node

            def visit_ClassDef(self, node):
                """Visit class definitions to insert docstrings."""
                self.generic_visit(node)
                self._insert_docstring(node)
                return node

            def visit_Module(self, node):
                """Visit module nodes to insert docstrings."""
                self.generic_visit(node)
                if not ast.get_docstring(node, clean=False):
                    docstring = docstring_map.get('__module__', 'No module docstring provided')
                    docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                    node.body.insert(0, docstring_node)
                return node

            def _insert_docstring(self, node):
                """Helper method to insert or replace docstrings."""
                docstring = docstring_map.get(node.name, 'No docstring provided')
                if sys.version_info >= (3, 8):
                    docstring_value = ast.Constant(value=docstring)
                else:
                    docstring_value = ast.Str(s=docstring)

                docstring_node = ast.Expr(value=docstring_value)

                if ast.get_docstring(node, clean=False) is not None:
                    # Replace existing docstring
                    node.body[0] = docstring_node
                else:
                    # Insert docstring at the beginning
                    node.body.insert(0, docstring_node)

        return DocstringInserter().visit(tree)

    def _generate_code_from_ast(self, tree: ast.AST) -> str:
        """
        Generate code from an AST.

        Args:
            tree (ast.AST): The abstract syntax tree from which to generate code.

        Returns:
            str: The generated source code as a string.

        Raises:
            DocumentationError: If code generation fails.

        """
        try:
            if hasattr(ast, "unparse"):
                return ast.unparse(tree)
            return astor.to_source(tree)
        except Exception as e:
            self.logger.error("Error generating code from AST: %s", e)
            raise DocumentationError(f"Failed to generate code from AST: {e}") from e

    def _generate_documentation(self, doc_entries: List[Dict[str, Any]]) -> str:
        """
        Generate consolidated documentation from doc_entries.

        Args:
            doc_entries (List[Dict[str, Any]]): A list of documentation entries.

        Returns:
            str: The consolidated documentation as a string.

        """
        # Placeholder for markdown generation logic
        documentation = ""
        for entry in doc_entries:
            documentation += f"### {entry['name']}\n\n{entry['docstring']}\n\n"
        return documentation

    def format(self, data: DocstringData) -> str:
        """
        Format structured docstring data into a string.

        Args:
            data (DocstringData): The docstring data to format.

        Returns:
            str: The formatted docstring as a string.

        """
        return FormattingUtils.format_docstring(data.__dict__)

    def validate(self, data: DocstringData) -> Tuple[bool, List[str]]:
        """
        Validate docstring data against requirements.

        Args:
            data (DocstringData): The docstring data to validate.

        Returns:
            Tuple[bool, List[str]]: A tuple containing a boolean indicating validation success,
            and a list of error messages if any.

        """
        return ValidationUtils.validate_docstring(data.__dict__, self.docstring_schema)