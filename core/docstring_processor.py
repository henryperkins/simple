import ast
import json
from typing import Optional, Dict, Any, List, Union, Tuple
from docstring_parser import parse as parse_docstring, Docstring
from core.logger import LoggerSetup, log_error, log_debug, log_info
from core.types.base import Injector
from core.metrics import Metrics
from core.types import DocstringData


class DocumentationError(Exception):
    """Exception raised for errors in the documentation."""
    pass


try:
    import astor
except ImportError as e:
    raise ImportError(
        "The 'astor' library is required for Python versions < 3.9 to generate code from AST. "
        "Please install it using 'pip install astor'."
    ) from e

# Set up the logger
logger = LoggerSetup.get_logger(__name__)


class DocstringProcessor:
    """
    Processes docstrings by parsing and validating them.

    This class provides methods to parse raw docstrings, validate them,
    integrate them into source code, and generate documentation.
    """

    def __init__(self, metrics: Metrics | None = None) -> None:
        """Initialize the DocstringProcessor.

        Args:
            metrics (Optional[Metrics]): The metrics instance for handling code metrics.
        """
        self.logger = logger
        self.metrics = metrics or Injector.get('metrics_calculator')
        self.docstring_schema: Dict[str, Any] = {}  # Placeholder for schema

    def _validate_docstring_dict(self, docstring_dict: Dict[str, Any]) -> None:
        """Validate that required keys exist in the docstring dictionary.

        Args:
            docstring_dict (Dict[str, Any]): The docstring dictionary to validate.

        Raises:
            DocumentationError: If required keys are missing from the docstring dictionary.
        """
        required_keys = {'summary', 'description', 'args', 'returns', 'raises'}
        missing_keys = required_keys - docstring_dict.keys()
        if missing_keys:
            self.logger.warning(
                f"Docstring dictionary missing keys: {missing_keys}")
            raise DocumentationError(
                f"Docstring dictionary missing keys: {missing_keys}")

    def parse(self, docstring: Union[Dict[str, Any], str]) -> DocstringData:
        """Parse a docstring from a string or dictionary.

        Args:
            docstring (Union[Dict[str, Any], str]): The docstring to parse.

        Returns:
            DocstringData: A structured representation of the parsed docstring.
        """
        try:
            self.logger.debug(
                f"Parsing docstring of type: {type(docstring).__name__}")
            if isinstance(docstring, dict):
                self._validate_docstring_dict(docstring)
                if 'complexity' not in docstring:
                    docstring['complexity'] = 1
                docstring_data = self._create_docstring_data_from_dict(
                    docstring)
            elif isinstance(docstring, str):
                docstring_str = docstring.strip()
                self.logger.debug(
                    f"Docstring length: {len(docstring_str)} characters")
                if docstring_str.startswith('{') and docstring_str.endswith('}'):
                    doc_dict = json.loads(docstring_str)
                    if 'complexity' not in doc_dict:
                        doc_dict['complexity'] = 1
                    docstring_data = self.parse(doc_dict)
                else:
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
                        'complexity': 1
                    }
                    docstring_data = self._create_docstring_data_from_dict(
                        docstring_dict)
            else:
                raise DocumentationError(
                    "Docstring must be either a dictionary or a string.")

            # Validate the docstring data
            is_valid, errors = self.validate(docstring_data)
            docstring_data.validation_status = is_valid
            docstring_data.validation_errors = errors

            if not is_valid:
                self.logger.warning(
                    f"Docstring validation failed with errors: {errors}")

            return docstring_data

        except DocumentationError:
            raise
        except Exception as e:
            log_error(
                f"Unexpected error in parse method: {e}",
                extra={"sanitized_info": {"error": str(e)}}
            )
            raise DocumentationError(
                f"Unexpected error during parsing: {e}") from e

    def _create_docstring_data_from_dict(self, docstring_dict: Dict[str, Any]) -> DocstringData:
        """Create DocstringData from a dictionary representation.

        Args:
            docstring_dict (Dict[str, Any]): The dictionary containing docstring information.

        Returns:
            DocstringData: The structured docstring data.
        """
        try:
            returns = docstring_dict.get('returns', {})
            if not isinstance(returns, dict):
                returns = {'type': 'Any', 'description': ''}
            if not returns.get('type'):
                returns['type'] = 'Any'
            if not returns.get('description'):
                returns['description'] = ''

            complexity = docstring_dict.get('complexity', 1)

            return DocstringData(
                summary=docstring_dict.get('summary', ''),
                description=docstring_dict.get('description', ''),
                args=docstring_dict.get('args', []),
                returns=returns,
                raises=docstring_dict.get('raises', []),
                complexity=complexity,
                validation_status=False,
                validation_errors=[]
            )
        except Exception as e:
            log_error(
                f"Error creating DocstringData from dict: {e}",
                extra={"sanitized_info": {"error": str(e)}}
            )
            raise DocumentationError(
                f"Failed to create DocstringData from dict: {e}") from e

    def process_batch(self, doc_entries: List[Dict[str, Any]], source_code: str) -> Dict[str, str]:
        """Process a batch of docstring entries and integrate them into the source code.

        Args:
            doc_entries (List[Dict[str, Any]]): The docstring entries to process.
            source_code (str): The source code to integrate the docstrings into.

        Returns:
            Dict[str, str]: A dictionary containing the updated code and documentation.
        """
        try:
            tree = ast.parse(source_code)
            log_debug(f"Processing {len(doc_entries)} documentation entries", extra={
                      "sanitized_info": {}})

            processed_entries: List[Dict[str, Any]] = []
            for entry in doc_entries:
                try:
                    log_debug(f"Processing entry: {entry}", extra={
                              "sanitized_info": {}})

                    if not isinstance(entry, dict):
                        log_error(
                            f"Entry is not a dictionary: {type(entry)}",
                            extra={"sanitized_info": {
                                "entry_type": type(entry).__name__}}
                        )
                        continue

                    if 'summary' in entry and 'name' not in entry:
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                                docstring = self.format(DocstringData(**entry))
                                processed_entries.append({
                                    'name': node.name,
                                    'docstring': docstring,
                                    'type': type(node).__name__
                                })
                                log_debug(f"Created processed entry for {node.name}", extra={
                                          "sanitized_info": {"node_name": node.name}})
                                break
                    else:
                        if 'docstring' not in entry and 'summary' in entry:
                            entry['docstring'] = self.format(
                                DocstringData(**entry))
                        processed_entries.append(entry)
                        log_debug(f"Added entry with name: {entry.get('name')}", extra={
                                  "sanitized_info": {"entry_name": entry.get('name')}})

                except Exception as e:
                    log_error(
                        f"Error processing entry: {e}",
                        extra={"sanitized_info": {"entry_error": str(e)}}
                    )
                    continue

            if not processed_entries:
                log_error(
                    "No valid entries were processed",
                    extra={"sanitized_info": {}}
                )
                return {'code': source_code, 'documentation': ""}

            updated_tree: Optional[ast.AST] = self._insert_docstrings(
                tree, processed_entries)
            if not updated_tree:
                log_error(
                    "Failed to update AST with docstrings",
                    extra={"sanitized_info": {}}
                )
                return {'code': source_code, 'documentation': ""}

            updated_code: Optional[str] = self._generate_code_from_ast(
                updated_tree)
            if not updated_code:
                log_error(
                    "Failed to generate code from AST",
                    extra={"sanitized_info": {}}
                )
                return {'code': source_code, 'documentation': ""}

            documentation: str = self._generate_documentation(
                processed_entries)
            if not documentation:
                log_error(
                    "Failed to generate documentation",
                    extra={"sanitized_info": {}}
                )
                return {'code': updated_code, 'documentation': ""}

            log_info(
                "Successfully processed batch",
                extra={"sanitized_info": {
                    "processed_entries_count": len(processed_entries)}}
            )
            return {'code': updated_code, 'documentation': documentation}

        except Exception as e:
            log_error(
                f"Unexpected error processing batch: {e}",
                extra={"sanitized_info": {"batch_error": str(e)}}
            )
            return {'code': source_code, 'documentation': ""}

    def _generate_documentation(self, doc_entries: List[Dict[str, Any]]) -> str:
        """Generate markdown documentation from processed docstring entries.

        Args:
            doc_entries (List[Dict[str, Any]]): The processed docstring entries.

        Returns:
            str: The generated markdown documentation.
        """
        try:
            log_debug(f"Generating documentation for {len(doc_entries)} entries", extra={
                      "sanitized_info": {}})
            doc_parts: List[str] = ["# API Documentation\n\n"]

            module_entry = next(
                (entry for entry in doc_entries if entry.get('name') == '__module__'), None)
            if module_entry:
                doc_parts.extend([
                    "## Module Overview\n\n",
                    f"{module_entry.get('docstring', '')}\n\n"
                ])

            class_entries = [
                entry for entry in doc_entries if entry.get('type') == 'ClassDef']
            if class_entries:
                doc_parts.append("## Classes\n\n")
                for entry in class_entries:
                    doc_parts.extend([
                        f"### {entry['name']}\n\n",
                        f"{entry.get('docstring', '')}\n\n"
                    ])

            func_entries = [entry for entry in doc_entries if entry.get(
                'type') in ('FunctionDef', 'AsyncFunctionDef')]
            if func_entries:
                doc_parts.append("## Functions\n\n")
                for entry in func_entries:
                    doc_parts.extend([
                        f"### {entry['name']}\n\n",
                        f"{entry.get('docstring', '')}\n\n"
                    ])

            documentation: str = ''.join(doc_parts)
            log_debug(f"Generated documentation length: {len(documentation)}", extra={
                      "sanitized_info": {}})
            return documentation

        except Exception as e:
            log_error(
                f"Error generating documentation: {e}",
                extra={"sanitized_info": {"documentation_error": str(e)}}
            )
            return ""

    def _insert_docstrings(self, tree: ast.AST, doc_entries: List[Dict[str, Any]]) -> Optional[ast.AST]:
        """Insert docstrings into the AST at relevant locations for each entry.

        Args:
            tree (ast.AST): The AST representation of the code.
            doc_entries (List[Dict[str, Any]]): The list of processed docstrings and their locations.

        Returns:
            Optional[ast.AST]: The updated AST with docstrings inserted, or None on failure.
        """
        try:
            docstring_map: Dict[str, str] = {
                entry['name']: entry['docstring']
                for entry in doc_entries
                if 'name' in entry and 'docstring' in entry
            }

            log_debug(f"Created docstring map with {len(docstring_map)} entries", extra={
                      "sanitized_info": {}})

            class DocstringInserter(ast.NodeTransformer):
                def visit_Module(self, node: ast.Module) -> ast.Module:
                    self.generic_visit(node)
                    if '__module__' in docstring_map and not ast.get_docstring(node):
                        docstring_node = ast.Expr(value=ast.Constant(
                            value=docstring_map['__module__']))
                        node.body.insert(0, docstring_node)
                    return node

                def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                    self.generic_visit(node)
                    if node.name in docstring_map:
                        docstring = docstring_map[node.name]
                        docstring_node = ast.Expr(
                            value=ast.Constant(value=docstring))
                        if len(node.body) > 0 and isinstance(node.body[0], ast.Expr):
                            node.body[0] = docstring_node
                        else:
                            node.body.insert(0, docstring_node)
                    return node

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
                    self.generic_visit(node)
                    if node.name in docstring_map:
                        docstring = docstring_map[node.name]
                        docstring_node = ast.Expr(
                            value=ast.Constant(value=docstring))
                        if len(node.body) > 0 and isinstance(node.body[0], ast.Expr):
                            node.body[0] = docstring_node
                        else:
                            node.body.insert(0, docstring_node)
                    return node

                def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
                    self.generic_visit(node)
                    if node.name in docstring_map:
                        docstring = docstring_map[node.name]
                        docstring_node = ast.Expr(
                            value=ast.Constant(value=docstring))
                        if len(node.body) > 0 and isinstance(node.body[0], ast.Expr):
                            node.body[0] = docstring_node
                        else:
                            node.body.insert(0, docstring_node)
                    return node

            transformer = DocstringInserter()
            new_tree = transformer.visit(tree)
            return new_tree

        except Exception as e:
            log_error(
                f"Error inserting docstrings: {e}",
                extra={"sanitized_info": {"insert_docstrings_error": str(e)}}
            )
            return None

    def _generate_code_from_ast(self, tree: ast.AST) -> Optional[str]:
        """Generate source code from an AST.

        Args:
            tree (ast.AST): The AST representation of the code.

        Returns:
            Optional[str]: The generated source code, or None on failure.
        """
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(tree)
            else:
                import astor
                return astor.to_source(tree)
        except Exception as e:
            log_error(
                f"Error generating code from AST: {e}",
                extra={"sanitized_info": {"generate_code_error": str(e)}}
            )
            return None

    def format(self, data: DocstringData) -> str:
        """Format the docstring data into a human-readable string.

        Args:
            data (DocstringData): The data object containing docstring information.

        Returns:
            str: The formatted docstring.
        """
        return f"{data.summary}\n\n{data.description}"

    def validate(self, data: DocstringData) -> Tuple[bool, List[str]]:
        """Validate the docstring data against the schema.

        Args:
            data (DocstringData): The data to validate.

        Returns:
            Tuple[bool, List[str]]: A tuple containing success flag and a list of validation errors.
        """
        errors = []
        required_fields = ['summary', 'description', 'args', 'returns']

        if not data.summary:
            errors.append("Summary is missing.")
        if not data.description:
            errors.append("Description is missing.")
        if not isinstance(data.args, list):
            errors.append("Args should be a list.")
        if not isinstance(data.returns, dict):
            errors.append("Returns should be a dictionary.")

        is_valid = len(errors) == 0
        return is_valid, errors
