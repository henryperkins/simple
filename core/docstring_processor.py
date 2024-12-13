import ast
import json
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
        self.docstring_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "description": {"type": "string"},
                "args": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
                "returns": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "description": {"type": "string"},
                    },
                },
                "raises": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "exception": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
                "complexity": {"type": "integer"},
            },
            "required": ["summary", "description", "args", "returns"],
        }

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
        if isinstance(docstring, dict):
            return self._create_docstring_data_from_dict(docstring)
        else:
            docstring_str = docstring.strip()
            if docstring_str.startswith("{") and docstring_str.endswith("}"):
                doc_dict = json.loads(docstring_str)
                return self._create_docstring_data_from_dict(doc_dict)
            else:
                parsed = parse_docstring(docstring)
                if parsed is None:
                    raise DocumentationError("Failed to parse the provided docstring.")
                return self._create_docstring_data_from_dict(
                    {
                        "summary": parsed.short_description or "",
                        "description": parsed.long_description or "",
                        "args": [
                            {
                                "name": p.arg_name,
                                "type": p.type_name or "Any",
                                "description": p.description or "",
                            }
                            for p in parsed.params
                        ],
                        "returns": {
                            "type": parsed.returns.type_name if parsed.returns else "Any",
                            "description": (
                                parsed.returns.description if parsed.returns else ""
                            ),
                        },
                        "raises": [
                            {"exception": e.type_name, "description": e.description}
                            for e in (parsed.raises or [])
                        ],
                        "complexity": 1,
                    }
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
                summary=docstring_dict.get("summary", ""),
                description=docstring_dict.get("description", ""),
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
    def validate(self, data: "DocstringData") -> Tuple[bool, List[str]]:
        errors = []
        required_fields = ["summary", "description", "args", "returns"]

        if not data.summary:
            errors.append("Summary is missing.")
        if not data.description:
            errors.append("Description is missing.")
        if not isinstance(data.args, list):
            errors.append("Args should be a list.")
        if not isinstance(data.returns, dict):
            errors.append("Returns should be a dictionary.")
        if not isinstance(data.raises, list):
            errors.append("Raises should be a list.")
        if not isinstance(data.complexity, int):
            errors.append("Complexity should be an integer.")

        is_valid = len(errors) == 0
        return is_valid, errors

    @handle_error
    def format(self, data: "DocstringData") -> str:
        if not data.summary or not data.description:
            raise DocumentationError(
                "Summary or description is missing for formatting."
            )
        return f"{data.summary}\n\n{data.description}"

    @handle_error
    async def process_batch(
        self, doc_entries: List[Dict[str, Any]], source_code: str
    ) -> Dict[str, str]:
        try:
            tree = ast.parse(source_code)
            self.logger.debug(f"Processing {len(doc_entries)} documentation entries")

            processed_entries: List[Dict[str, Any]] = []
            for entry in doc_entries:
                try:
                    if "summary" in entry and "name" not in entry:
                        for node in ast.walk(tree):
                            if isinstance(
                                node,
                                (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef),
                            ):
                                docstring = self.format(DocstringData(**entry))
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
                            entry["docstring"] = self.format(DocstringData(**entry))
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
                import astor

                return astor.to_source(tree)
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
