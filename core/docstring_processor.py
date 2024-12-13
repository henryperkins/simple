import ast
import json
import os
from typing import Any, Dict, List, Union, Optional, Tuple, TYPE_CHECKING
from docstring_parser import parse as parse_docstring

from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.dependency_injection import Injector
from core.metrics import Metrics
from core.exceptions import DocumentationError
from utils import handle_error
from core.types.base import DocstringData


class DocstringProcessor:
    def __init__(self, metrics: Optional[Metrics] = None) -> None:
        self.logger = CorrelationLoggerAdapter(LoggerSetup.get_logger(__name__))
        self.metrics = metrics or Injector.get("metrics_calculator")
        self.docstring_schema: Dict[str, Any] = self._load_schema("docstring_schema.json")

    def _load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load a JSON schema for validation."""
        try:
            schema_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "schemas", schema_name
            )
            with open(schema_path, "r") as f:
                schema = json.load(f)
                self.logger.info(f"Schema '{schema_name}' loaded successfully.")
                return schema
        except FileNotFoundError:
            self.logger.error(f"Schema file '{schema_name}' not found.")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON in '{schema_name}': {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading schema '{schema_name}': {e}")
            raise

    @handle_error
    def __call__(self, docstring: Union[Dict[str, Any], str]) -> "DocstringData":
        try:
            if isinstance(docstring, dict):
                return self._create_docstring_data_from_dict(docstring)
            elif isinstance(docstring, str):
                return self.parse(docstring)
            else:
                raise DocumentationError(
                    "Docstring must be either a dictionary or a string."
                )
        except Exception as e:
            self.logger.error(f"Unexpected error during parsing: {e}")
            raise

    def parse(self, docstring: Union[Dict[str, Any], str]) -> "DocstringData":
        """Parse docstring into structured data.
        
        Args:
            docstring: Raw docstring as dict or string.

        Returns:
            DocstringData: Parsed docstring data.

        Raises:
            DocumentationError: If parsing fails.
        """
        try:
            if isinstance(docstring, dict):
                self.logger.debug("Received dictionary docstring, processing directly")
                return self._create_docstring_data_from_dict(docstring)
            
            docstring_str = str(docstring).strip()
            
            # Parse with docstring_parser
            parsed = parse_docstring(docstring_str)
            if parsed is None:
                self.logger.error("Failed to parse docstring")
                raise DocumentationError("Failed to parse the provided docstring")
                
            return self._create_docstring_data_from_parsed(parsed)
            
        except Exception as e:
            self.logger.error(f"Error parsing docstring: {e}")
            raise DocumentationError(f"Failed to parse docstring: {e}") from e

    def _create_docstring_data_from_parsed(self, parsed_docstring) -> "DocstringData":
        """Create DocstringData from a parsed docstring object."""
        return DocstringData(
            summary=parsed_docstring.short_description or "",
            description=parsed_docstring.long_description or "",
            args=[
                {
                    "name": param.arg_name,
                    "type": param.type_name or "Any",
                    "description": param.description or "",
                }
                for param in parsed_docstring.params
            ],
            returns={
                "type": parsed_docstring.returns.type_name if parsed_docstring.returns else "Any",
                "description": parsed_docstring.returns.description if parsed_docstring.returns else "",
            },
            raises=[
                {
                    "exception": exc.type_name or "Exception",
                    "description": exc.description or "",
                }
                for exc in parsed_docstring.raises
            ],
            complexity=1,
        )

    def _create_docstring_data_from_dict(
        self, docstring_dict: Dict[str, Any]
    ) -> "DocstringData":
        try:
            returns = docstring_dict.get("returns", {})
            if not isinstance(returns, dict):
                returns = {"type": "Any", "description": ""}
            if not returns.get("type"):
                returns["type"] = "Any"
            if not returns.get("description"):
                returns["description"] = ""

            complexity = docstring_dict.get("complexity", 1)

            return DocstringData(
                summary=docstring_dict.get("summary", "No summary provided."),
                description=docstring_dict.get("description", "No description provided."),
                args=docstring_dict.get("args", []),
                returns=returns,
                raises=docstring_dict.get("raises", []),
                complexity=complexity,
                validation_status=False,
                validation_errors=[],
            )
        except KeyError as e:
            self.logger.warning(f"Missing required key in docstring dict: {e}")
            raise DocumentationError(f"Docstring dictionary missing keys: {e}")

    @handle_error
    async def process_batch(
        self, doc_entries: List[Dict[str, Any]], source_code: str
    ) -> Dict[str, str]:
        try:
            tree = ast.parse(source_code)
            self.logger.debug(f"Processing {len(doc_entries)} documentation entries")

            processed_entries: List [Dict[str, Any]] = []
            for entry in doc_entries:
                try:
                   if "summary" in entry and "name" not in entry:
                        for node in ast.walk(tree):
                            if isinstance(
                                node,
                                (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef),
                            ):
                                docstring =  DocstringData(**entry).__dict__
                                processed_entries.append(
                                    {
                                        "name": node.name,
                                        "docstring": docstring,
                                        "type": type(node).__name__,
                                    }
                                )
                                self.logger.debug(
                                    f"Created processed entry for {node.name}"
                                )
                                break
                   else:
                        if "docstring" not in entry and "summary" in entry:
                            entry["docstring"] = DocstringData(**entry).__dict__
                        processed_entries.append(entry)
                        self.logger.debug(f"Added entry with name: {entry.get('name')}")

                except Exception as e:
                    self.logger.error(f"Error processing entry: {e}")
                    continue

            if not processed_entries:
                self.logger.error("No valid entries were processed")
                return {"code": source_code, "documentation": ""}

            updated_tree: Optional[ast.AST] = self._insert_docstrings(
                tree, processed_entries
            )
            if not updated_tree:
                self.logger.error("Failed to update AST with docstrings")
                return {"code": source_code, "documentation": ""}

            updated_code: Optional[str] = self._generate_code_from_ast(updated_tree)
            if not updated_code:
                self.logger.error("Failed to generate code from AST")
                return {"code": source_code, "documentation": ""}

            documentation = self._generate_documentation(processed_entries)
            self.logger.info("Successfully processed batch")
            return {"code": updated_code, "documentation": documentation}

        except Exception as e:
            self.logger.error(f"Unexpected error processing batch: {e}")
            raise DocumentationError(f"Failed to process batch: {e}")

    def _insert_docstrings(
        self, tree: ast.AST, doc_entries: List[Dict[str, Any]]
    ) -> Optional[ast.AST]:
        docstring_map = {
            entry["name"]: entry["docstring"]
            for entry in doc_entries
            if "name" in entry and "docstring" in entry
        }

        class DocstringInserter(ast.NodeTransformer):
            def visit_Module(self, node: ast.Module) -> ast.Module:
                self.generic_visit(node)
                if "__module__" in docstring_map and not ast.get_docstring(node):
                    docstring_node = ast.Expr(
                        value=ast.Constant(value=docstring_map["__module__"])
                    )
                    node.body.insert(0, docstring_node)
                return node

            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                self.generic_visit(node)
                if node.name in docstring_map:
                    docstring = docstring_map[node.name]
                    docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                    if len(node.body) > 0 and isinstance(node.body[0], ast.Expr):
                        node.body[0] = docstring_node
                    else:
                        node.body.insert(0, docstring_node)
                return node

            def visit_AsyncFunctionDef(
                self, node: ast.AsyncFunctionDef
            ) -> ast.AsyncFunctionDef:
                self.generic_visit(node)
                if node.name in docstring_map:
                    docstring = docstring_map[node.name]
                    docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                    if len(node.body) > 0 and isinstance(node.body[0], ast.Expr):
                        node.body[0] = docstring_node
                    else:
                        node.body.insert(0, docstring_node)
                return node

            def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
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

    def _generate_code_from_ast(self, tree: ast.AST) -> Optional[str]:
        try:
            if hasattr(ast, "unparse"):
                return ast.unparse(tree)
            else:
                try:
                    import astor
                    return astor.to_source(tree)
                except ImportError:
                    self.logger.warning("astor library not found, using ast.dump instead.")
                    return ast.dump(tree, annotate_fields=True)
        except Exception as e:
            self.logger.error(f"Error generating code from AST: {e}")
            raise DocumentationError(f"Failed to generate code from AST: {e}")

    def _generate_documentation(self, doc_entries: List[Dict[str, Any]]) -> str:
        doc_parts = ["# API Documentation\n\n"]

        module_entry = next(
            (entry for entry in doc_entries if entry.get("name") == "__module__"), None
        )
        if module_entry:
            doc_parts.extend(
                ["## Module Overview\n\n", f"{module_entry.get('docstring', '')}\n\n"]
            )

        class_entries = [
            entry for entry in doc_entries if entry.get("type") == "ClassDef"
        ]
        if class_entries:
            doc_parts.append("## Classes\n\n")
            for entry in class_entries:
                doc_parts.extend(
                    [f"### {entry['name']}\n\n", f"{entry.get('docstring', '')}\n\n"]
                )

        func_entries = [
            entry
            for entry in doc_entries
            if entry.get("type") in ("FunctionDef", "AsyncFunctionDef")
        ]
        if func_entries:
            doc_parts.append("## Functions\n\n")
            for entry in func_entries:
                doc_parts.extend(
                    [f"### {entry['name']}\n\n", f"{entry.get('docstring', '')}\n\n"]
                )

        return "".join(doc_parts)


@handle_error
def handle_extraction_error(
    e: Exception, errors: List[str], context: str, correlation_id: str, **kwargs: Any
) -> None:
    error_message = f"{context}: {str(e)}"
    errors.append(error_message)

    logger = CorrelationLoggerAdapter(LoggerSetup.get_logger(__name__))
    logger.error(
        f"Error in {context}: {e}", exc_info=True, extra={"sanitized_info": kwargs}
    )
