"""Docstring processing module."""

import ast
import json
import sys
from typing import Optional, Dict, Any, List, Union, Tuple
from docstring_parser import parse as parse_docstring
from core.logger import LoggerSetup
from core.metrics import Metrics
from core.types import DocstringData
from exceptions import DocumentationError
from core.utils import FormattingUtils
from core.schema_loader import load_schema

try:
    import astor
except ImportError as e:
    raise ImportError(
        "The 'astor' library is required for Python versions < 3.9 to generate code from AST. "
        "Please install it using 'pip install astor'."
    ) from e

class ValidationUtils:
    """Utility methods for validation."""

    @staticmethod
    def validate_docstring(docstring: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate the generated docstring JSON against the schema."""
        errors = []
        required_fields = {"summary", "description", "args", "returns", "raises", "complexity"}

        for field in required_fields:
            if field not in docstring:
                errors.append(f"Missing required field: {field}")
            elif not docstring[field]:
                errors.append(f"Empty value for required field: {field}")

        # Validate args structure
        for arg in docstring.get("args", []):
            if not all(key in arg for key in ["name", "type", "description"]):
                missing_keys = [key for key in ["name", "type", "description"] if key not in arg]
                errors.append(f"Incomplete argument specification: Missing {', '.join(missing_keys)} for {arg}")

        # Validate raises section
        for exc in docstring.get("raises", []):
            if "exception" not in exc or "description" not in exc:
                errors.append(f"Incomplete raises specification: {exc}")

        return len(errors) == 0, errors

class DocstringProcessor:
    """Processes docstrings by parsing and validating them."""

    def __init__(self, metrics: Optional[Metrics] = None) -> None:
        """Initialize docstring processor."""
        self.logger = LoggerSetup.get_logger(__name__)
        self.metrics = metrics or Metrics()
        self.docstring_schema = load_schema("docstring_schema")

    def parse(self, docstring: Union[Dict[str, Any], str]) -> DocstringData:
        """Parse a raw docstring into structured format."""
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
        """Validate the structure of the docstring dictionary."""
        required_keys = {'summary', 'description', 'args', 'returns', 'raises', 'complexity'}
        missing_keys = required_keys - docstring_dict.keys()
        if missing_keys:
            raise DocumentationError(f"Docstring dictionary missing keys: {missing_keys}")
    
    def _create_docstring_data_from_dict(self, docstring_dict: Dict[str, Any]) -> DocstringData:
        """Create DocstringData from a validated dictionary."""
        try:
            # Ensure 'returns' is a dictionary with required fields
            returns = docstring_dict.get('returns', {})
            if not isinstance(returns, dict):
                returns = {'type': 'Any', 'description': ''}

            # Validate the docstring against the schema
            is_valid, errors = ValidationUtils.validate_docstring(docstring_dict, self.docstring_schema)
            if not is_valid:
                raise DocumentationError(f"Validation errors found: {errors}")

            return DocstringData(
                summary=docstring_dict.get('summary', ''),
                description=docstring_dict.get('description', ''),
                args=docstring_dict.get('args', []),
                returns=returns,
                raises=docstring_dict.get('raises', []),
                complexity=docstring_dict.get('complexity', 1)
            )
        except Exception as e:
            self.logger.error("Error creating DocstringData from dict: %s", e)
            raise DocumentationError(f"Failed to create DocstringData from dict: {e}") from e

    def _create_docstring_data_from_parsed(self, parsed_docstring) -> DocstringData:
        """Create DocstringData from parsed docstring object."""
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
        """
        Process a batch of documentation entries and integrate them into the source code.
        
        Returns:
            Dict[str, Any]: A dictionary containing the updated code and documentation.
        """
        try:
            # Parse the source code into an AST
            tree = ast.parse(source_code)

            # Insert or replace docstrings
            tree = self._insert_docstrings(tree, doc_entries)

            # Generate the updated source code from the AST
            updated_code = self._generate_code_from_ast(tree)

            # Validate the resulting code
            is_valid, errors = ValidationUtils.validate_docstring(doc_entries[0], self.docstring_schema)
            if not is_valid:
                raise DocumentationError(f"Validation errors found: {errors}")

            # Generate the consolidated documentation
            documentation = self._generate_documentation(doc_entries)

            return {'code': updated_code, 'documentation': documentation}
        except Exception as e:
            logger.error("Error processing batch: %s", e)
            raise
    
    def _insert_docstrings(self, tree: ast.AST, doc_entries: List[Dict[str, Any]]) -> ast.AST:
        """Insert docstrings into the AST based on doc_entries."""
        # Create a mapping from name to docstring
        docstring_map = {entry['name']: entry.get('docstring', 'No docstring provided') for entry in doc_entries}

        class DocstringInserter(ast.NodeTransformer):
            """Inserts docstrings into AST nodes."""

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
        """Generate code from an AST."""
        try:
            if hasattr(ast, "unparse"):
                return ast.unparse(tree)
            return astor.to_source(tree)
        except Exception as e:
            self.logger.error("Error generating code from AST: %s", e)
            raise DocumentationError(f"Failed to generate code from AST: {e}") from e

    def _generate_documentation(self, doc_entries: List[Dict[str, Any]]) -> str:
        """Generate consolidated documentation from doc_entries."""
        # Placeholder for markdown generation logic
        documentation = ""
        for entry in doc_entries:
            documentation += f"### {entry['name']}\n\n{entry['docstring']}\n\n"
        return documentation

    def format(self, data: DocstringData) -> str:
        """Format structured docstring data into a string."""
        return FormattingUtils.format_docstring(data.__dict__)

    def validate(self, data: DocstringData) -> Tuple[bool, List[str]]:
        """Validate docstring data against requirements."""
        return ValidationUtils.validate_docstring(data.__dict__, self.docstring_schema)
