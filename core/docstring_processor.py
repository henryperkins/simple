"""
Module for processing and validating docstrings.

This module provides utilities for parsing, validating, and formatting docstrings in Python code.
It supports parsing docstrings from various formats, validating them against a schema, integrating
them into source code, and generating updated source code from modified ASTs.

Classes:
    DocstringProcessor: Class for processing and validating docstrings.

"""

import ast
import json
import sys
from typing import Optional, Dict, Any, List, Union, Tuple, Type

from docstring_parser import parse as parse_docstring, Docstring

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
        self.docstring_schema: Dict[str, Any] = load_schema("docstring_schema")

    def _validate_docstring_dict(self, docstring_dict: Dict[str, Any]) -> None:
        """
        Validate the structure of the docstring dictionary.

        Args:
            docstring_dict (Dict[str, Any]): The docstring dictionary to validate.

        Raises:
            DocumentationError: If required keys are missing in the docstring.
        """
        required_keys = {'summary', 'description', 'args', 'returns', 'raises'}  # Remove complexity as required
        missing_keys = required_keys - docstring_dict.keys()
        if missing_keys:
            raise DocumentationError(f"Docstring dictionary missing keys: {missing_keys}")

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
                # Add complexity if missing
                if 'complexity' not in docstring:
                    docstring['complexity'] = 1  # Default complexity
                # Create DocstringData from dictionary
                return self._create_docstring_data_from_dict(docstring)
            elif isinstance(docstring, str):
                # Try parsing as JSON first
                docstring_str = docstring.strip()
                if docstring_str.startswith('{') and docstring_str.endswith('}'):
                    try:
                        doc_dict = json.loads(docstring_str)
                        if 'complexity' not in doc_dict:
                            doc_dict['complexity'] = 1  # Default complexity
                        return self.parse(doc_dict)
                    except json.JSONDecodeError as e:
                        self.logger.error("JSON parsing error: %s", e)
                        raise DocumentationError(
                            f"Invalid JSON format in docstring: {e}"
                        ) from e
                # Fall back to standard docstring parsing
                try:
                    parsed: Docstring = parse_docstring(docstring_str)
                    docstring_dict = {
                        'summary': parsed.short_description or '',
                        'description': parsed.long_description or '',
                        'args': [{'name': p.arg_name, 'type': p.type_name or 'Any', 'description': p.description or ''}
                                 for p in parsed.params],
                        'returns': {
                            'type': parsed.returns.type_name if parsed.returns else 'Any',
                            'description': parsed.returns.description if parsed.returns else ''
                        },
                        'raises': [{'exception': e.type_name, 'description': e.description}
                                   for e in (parsed.raises or [])],
                        'complexity': 1  # Default complexity
                    }
                    return self._create_docstring_data_from_dict(docstring_dict)
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

            # Ensure complexity has a default value
            complexity = docstring_dict.get('complexity', 1)

            return DocstringData(
                summary=docstring_dict.get('summary', ''),
                description=docstring_dict.get('description', ''),
                args=docstring_dict['args'],
                returns=returns,
                raises=docstring_dict['raises'],
                complexity=complexity
            )
        except Exception as e:
            self.logger.error("Error creating DocstringData from dict: %s", e)
            raise DocumentationError(f"Failed to create DocstringData from dict: {e}") from e

    def _create_docstring_data_from_parsed(self, parsed_docstring: Docstring) -> DocstringData:
        """
        Create DocstringData from parsed docstring object.

        Args:
            parsed_docstring (Docstring): The parsed docstring object from docstring_parser.

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

    def process_batch(self, doc_entries: List[Dict[str, Any]], source_code: str) -> Dict[str, str]:
        """Process a batch of documentation entries and integrate them into the source code."""
        try:
            # Parse the source code into an AST
            tree = ast.parse(source_code)

            self.logger.debug(f"Processing {len(doc_entries)} documentation entries")
            self.logger.debug(f"Doc entries received: {json.dumps(doc_entries, indent=2)}")

            # Ensure each entry has required fields
            processed_entries: List[Dict[str, Any]] = []
            for entry in doc_entries:
                try:
                    self.logger.debug(f"Processing entry: {json.dumps(entry, indent=2)}")

                    # Handle both direct docstring dicts and wrapped entries
                    if not isinstance(entry, dict):
                        self.logger.error(f"Entry is not a dictionary: {type(entry)}")
                        continue

                    # If entry is a docstring dict (has summary but no name)
                    if 'summary' in entry and 'name' not in entry:
                        # Look for a matching node in the AST
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                                docstring = self.format(DocstringData(**entry))
                                processed_entries.append({
                                    'name': node.name,
                                    'docstring': docstring,
                                    'type': type(node).__name__
                                })
                                self.logger.debug(f"Created processed entry for {node.name}")
                                break
                    else:
                        # Entry already has a name
                        if 'docstring' not in entry and 'summary' in entry:
                            # Format the entry as a docstring
                            entry['docstring'] = self.format(DocstringData(**entry))
                        processed_entries.append(entry)
                        self.logger.debug(f"Added entry with name: {entry.get('name')}")

                except Exception as e:
                    self.logger.error(f"Error processing entry: {e}", exc_info=True)
                    continue

            if not processed_entries:
                self.logger.error("No valid entries were processed")
                return {'code': source_code, 'documentation': ""}

            self.logger.debug(f"Processed entries: {json.dumps(processed_entries, indent=2)}")

            # Insert docstrings into the AST
            updated_tree: Optional[ast.AST] = self._insert_docstrings(tree, processed_entries)
            if not updated_tree:
                self.logger.error("Failed to update AST with docstrings")
                return {'code': source_code, 'documentation': ""}

            # Generate the updated source code
            try:
                updated_code: Optional[str] = self._generate_code_from_ast(updated_tree)
                if not updated_code:
                    self.logger.error("Failed to generate code from AST")
                    return {'code': source_code, 'documentation': ""}
            except Exception as e:
                self.logger.error(f"Error generating code: {e}", exc_info=True)
                return {'code': source_code, 'documentation': ""}

            # Generate the documentation
            try:
                documentation: str = self._generate_documentation(processed_entries)
                if not documentation:
                    self.logger.error("Failed to generate documentation")
                    return {'code': updated_code, 'documentation': ""}
            except Exception as e:
                self.logger.error(f"Error generating documentation: {e}", exc_info=True)
                return {'code': updated_code, 'documentation': ""}

            self.logger.info("Successfully processed batch")
            return {'code': updated_code, 'documentation': documentation}

        except Exception as e:
            self.logger.error(f"Unexpected error processing batch: {e}", exc_info=True)
            return {'code': source_code, 'documentation': ""}

    def _generate_documentation(self, doc_entries: List[Dict[str, Any]]) -> str:
        """Generate consolidated documentation from doc_entries."""
        try:
            self.logger.debug(f"Generating documentation for {len(doc_entries)} entries")

            # Start with a header
            doc_parts: List[str] = ["# API Documentation\n\n"]

            # Add module-level documentation if present
            module_entry = next((entry for entry in doc_entries if entry.get('name') == '__module__'), None)
            if module_entry:
                doc_parts.extend([
                    "## Module Overview\n\n",
                    f"{module_entry.get('docstring', '')}\n\n"
                ])

            # Process classes
            class_entries = [entry for entry in doc_entries
                             if entry.get('type') == 'ClassDef']
            if class_entries:
                doc_parts.append("## Classes\n\n")
                for entry in class_entries:
                    doc_parts.extend([
                        f"### {entry['name']}\n\n",
                        f"{entry.get('docstring', '')}\n\n"
                    ])

            # Process functions
            func_entries = [entry for entry in doc_entries
                            if entry.get('type') in ('FunctionDef', 'AsyncFunctionDef')]
            if func_entries:
                doc_parts.append("## Functions\n\n")
                for entry in func_entries:
                    doc_parts.extend([
                        f"### {entry['name']}\n\n",
                        f"{entry.get('docstring', '')}\n\n"
                    ])

            documentation: str = ''.join(doc_parts)
            self.logger.debug(f"Generated documentation length: {len(documentation)}")
            return documentation

        except Exception as e:
            self.logger.error(f"Error generating documentation: {e}", exc_info=True)
            return ""

    def _insert_docstrings(self, tree: ast.AST, doc_entries: List[Dict[str, Any]]) -> Optional[ast.AST]:
        """Insert docstrings into the AST."""
        try:
            # Create a mapping from name to docstring
            docstring_map: Dict[str, str] = {
                entry['name']: entry['docstring']
                for entry in doc_entries
                if 'name' in entry and 'docstring' in entry
            }

            self.logger.debug(f"Created docstring map with {len(docstring_map)} entries")

            class DocstringInserter(ast.NodeTransformer):
                def visit_Module(self, node: ast.Module) -> ast.AST:
                    self.generic_visit(node)
                    if '__module__' in docstring_map and not ast.get_docstring(node):
                        docstring_node = ast.Expr(value=ast.Constant(value=docstring_map['__module__']))
                        node.body.insert(0, docstring_node)
                    return node

                def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                    self.generic_visit(node)
                    if node.name in docstring_map:
                        docstring = docstring_map[node.name]
                        docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                        if len(node.body) > 0 and isinstance(node.body[0], ast.Expr):
                            node.body[0] = docstring_node
                        else:
                            node.body.insert(0, docstring_node)
                    return node

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
                    return self.visit_FunctionDef(node)  # Reuse FunctionDef logic

                def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
                    self.generic_visit(node)
                    if node.name in docstring_map:
                        docstring = docstring_map[node.name]
                        docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                        if len(node.body) > 0 and isinstance(node.body[0], ast.Expr):
                            node.body[0] = docstring_node
                        else:
                            node.body.insert(0, docstring_node)
                    return node

            transformer = DocstringInserter()
            new_tree = transformer.visit(tree)
            return new_tree

        except Exception as e:
            self.logger.error(f"Error inserting docstrings: {e}", exc_info=True)
            return None

    def _generate_code_from_ast(self, tree: ast.AST) -> Optional[str]:
        """Generate code from an AST."""
        try:
            if hasattr(ast, 'unparse'):  # Python 3.9+
                return ast.unparse(tree)
            else:
                import astor
                return astor.to_source(tree)
        except Exception as e:
            self.logger.error(f"Error generating code from AST: {e}", exc_info=True)
            return None

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
            data: The docstring data to validate.

        Returns:
            Tuple[bool, List[str]]: A tuple containing a boolean indicating validation success,
            and a list of error messages if any.

        """
        return ValidationUtils.validate_docstring(data.__dict__, self.docstring_schema)
