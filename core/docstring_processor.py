"""Processes and validates docstrings."""

import ast
import json
import os
from collections.abc import Sequence
from typing import Any, cast, TypedDict, Union

try:
    from docstring_parser import parse, DocstringStyle, Docstring
except ImportError:
    print("Warning: docstring_parser not found. Install with: pip install docstring-parser")
    raise


from core.console import (
    print_info,
    print_phase_header,
    display_processing_phase,
    display_metrics,
    display_validation_results,
    create_status_table,
)
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.exceptions import DocumentationError
from core.types.docstring import DocstringData
from core.metrics_collector import MetricsCollector
from jsonschema import validate, ValidationError


class ReturnsDict(TypedDict):
    type: str
    description: str


class DocstringProcessor:
    """Processes and validates docstrings."""

    def __init__(self) -> None:
        """Initializes the DocstringProcessor."""
        self.logger = CorrelationLoggerAdapter(LoggerSetup.get_logger(__name__))
        self.docstring_schema: dict[str, Any] = self._load_schema("docstring_schema.json")
        self.metrics_collector = MetricsCollector()
        self.docstring_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_lines": 0,
            "avg_length": 0,
        }

    def _load_schema(self, schema_name: str) -> dict[str, Any]:
        """Load a JSON schema for validation."""
        try:
            schema_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "schemas", schema_name
            )
            with open(schema_path, "r") as f:
                schema = json.load(f)
                self.logger.info(
                    "Schema loaded successfully",
                    extra={
                        "schema_name": schema_name,
                        "sanitized_info": {"status": "success"}
                    }
                )
                return schema
        except FileNotFoundError:
            self.logger.error(
                "Schema file not found",
                extra={
                    "schema_name": schema_name,
                    "sanitized_info": {"status": "error", "type": "file_not_found"}
                }
            )
            raise
        except json.JSONDecodeError:
            self.logger.error(
                "Failed to parse JSON schema",
                extra={
                    "schema_name": schema_name,
                    "sanitized_info": {"status": "error", "type": "json_decode"}
                }
            )
            raise
        except Exception:
            self.logger.error(
                "Error loading schema",
                extra={
                    "schema_name": schema_name,
                    "sanitized_info": {"status": "error", "type": "unknown"}
                }
            )
            raise

    def __call__(self, docstring: Union[dict[str, Any], str]) -> DocstringData:
        """Processes a docstring, either as a dictionary or a string."""
        try:
            if isinstance(docstring, dict):
                return self._process_dict_docstring(docstring)
            return self.parse(docstring)
        except Exception as e:
            self.logger.error(
                "Unexpected error during parsing",
                extra={"sanitized_info": {"error": str(e)}}
            )
            raise

    def parse(self, docstring: Union[str, dict[str, Any]]) -> DocstringData:
        """Parse a docstring into structured data with validation."""
        try:
            if isinstance(docstring, dict):
                docstring_str = (
                    docstring.get("summary", "") or 
                    docstring.get("description", "") or 
                    str(docstring)
                )
            else:
                docstring_str = docstring

            result = self._parse_docstring_content(docstring_str)
            return DocstringData(**result)

        except Exception as e:
            self.docstring_stats["failed"] += 1
            self.logger.error(
                "Error parsing docstring",
                extra={"sanitized_info": {"error": str(e)}}
            )
            raise ValidationError(f"Failed to parse docstring: {e}")

    def _parse_docstring_content(self, docstring: str) -> dict[str, Any]:
        """Parse docstring content into structured format."""
        docstring_str = docstring.strip()
        lines = len(docstring_str.splitlines())
        length = len(docstring_str)

        # Update statistics
        self.docstring_stats["total_processed"] += 1
        self.docstring_stats["total_lines"] += lines
        self.docstring_stats["avg_length"] = (
            (self.docstring_stats["avg_length"] * (self.docstring_stats["total_processed"] - 1) + length) 
            // self.docstring_stats["total_processed"]
        )

        # Try parsing with AUTO style first, then fall back to specific styles
        parsed_docstring: Docstring
        try:
            parsed_docstring = parse(docstring_str, style=DocstringStyle.AUTO)
            self.docstring_stats["successful"] += 1
        except Exception:
            try:
                parsed_docstring = parse(docstring_str, style=DocstringStyle.GOOGLE)
                self.docstring_stats["successful"] += 1
            except Exception:
                try:
                    parsed_docstring = parse(docstring_str, style=DocstringStyle.REST)
                    self.docstring_stats["successful"] += 1
                except Exception as e:
                    self.docstring_stats["failed"] += 1
                    self.logger.error(
                        "Failed to parse docstring",
                        extra={"sanitized_info": {"error": str(e)}}
                    )
                    # Return minimal DocstringData
                    return {
                        "summary": docstring_str,
                        "description": "",
                        "args": [],
                        "returns": {"type": "Any", "description": ""},
                        "raises": [],
                        "complexity": 1
                    }

        # Display periodic statistics
        if self.docstring_stats["total_processed"] % 10 == 0:
            self._display_docstring_stats()
        
        # Extract data from parsed docstring
        args: list[dict[str, str | list[dict[str, str]]]] = []
        for param in parsed_docstring.params:
            arg: dict[str, str | list[dict[str, str]]] = {
                "name": param.arg_name or "",
                "type": param.type_name or "Any",
                "description": param.description or "",
                "nested": []
            }
            args.append(arg)

        returns_dict: ReturnsDict = {
            "type": "Any",
            "description": "",
        }
        if parsed_docstring.returns:
            returns_dict["type"] = parsed_docstring.returns.type_name or "Any"
            returns_dict["description"] = parsed_docstring.returns.description or ""

        raises: list[dict[str, str]] = []
        for exc in parsed_docstring.raises:
            exc_dict = {
                "exception": exc.type_name or "Exception",
                "description": exc.description or "",
            }
            raises.append(exc_dict)

        result = {
            "summary": parsed_docstring.short_description or "No summary available.",
            "description": parsed_docstring.long_description or "No description provided.",
            "args": args,
            "returns": returns_dict,
            "raises": raises,
            "complexity": 1,
        }

        return result

    def _display_docstring_stats(self) -> None:
        """Display current docstring processing statistics."""
        display_metrics({
            "Total Processed": self.docstring_stats["total_processed"],
            "Successfully Parsed": self.docstring_stats["successful"],
            "Failed to Parse": self.docstring_stats["failed"],
            "Average Length": f"{self.docstring_stats['avg_length']}",
            "Total Lines": self.docstring_stats["total_lines"],
            "Success Rate": f"{(self.docstring_stats['successful'] / self.docstring_stats['total_processed'] * 100):.1f}%"
        }, title="Docstring Processing Statistics")

    def _process_dict_docstring(self, docstring_dict: dict[str, Any]) -> DocstringData:
        """Process a docstring provided as a dictionary."""
        try:
            print_phase_header("Validating Docstring Dictionary")
            self._validate_dict_docstring(docstring_dict)
            return self._create_docstring_data_from_dict(docstring_dict)
        except Exception as e:
            self.logger.error(
                "Error processing dictionary docstring",
                extra={"sanitized_info": {"error": str(e)}}
            )
            raise

    def _validate_dict_docstring(self, docstring_dict: dict[str, Any]) -> bool:
        """Validate a docstring dictionary against the schema."""
        try:
            validate(instance=docstring_dict, schema=self.docstring_schema)
            display_validation_results({"schema_validation": True})
            return True
        except ValidationError as e:
            display_validation_results(
                {"schema_validation": False},
                {"schema_validation": str(e)}
            )
            raise DocumentationError(f"Docstring dictionary validation failed: {e}")
        except Exception as e:
            display_validation_results(
                {"schema_validation": False},
                {"schema_validation": str(e)}
            )
            raise DocumentationError(f"Unexpected error during validation: {e}")

    def _create_docstring_data_from_dict(
        self, docstring_dict: dict[str, Any]
    ) -> DocstringData:
        """Create DocstringData from a validated dictionary."""
        try:
            print_phase_header("Creating DocstringData")
            cleaned_dict = {
                "summary": docstring_dict.get("summary", "No summary provided."),
                "description": docstring_dict.get("description", "No description provided."),
                "args": docstring_dict.get("args", []),
                "returns": docstring_dict.get("returns", {"type": "Any", "description": ""}),
                "raises": docstring_dict.get("raises", []),
                "complexity": docstring_dict.get("complexity", 1)
            }

            # Ensure returns has the correct structure
            if not isinstance(cleaned_dict["returns"], dict):
                cleaned_dict["returns"] = {"type": "Any", "description": ""}
            if "type" not in cleaned_dict["returns"]:
                cleaned_dict["returns"]["type"] = "Any"
            if "description" not in cleaned_dict["returns"]:
                cleaned_dict["returns"]["description"] = ""

            # Create DocstringData with explicit arguments
            result = DocstringData(
                summary=cleaned_dict["summary"],
                description=cleaned_dict["description"],
                args=cleaned_dict["args"],
                returns=cast(dict[str, str], cleaned_dict["returns"]),
                raises=cleaned_dict["raises"],
                complexity=cleaned_dict["complexity"]
            )

            create_status_table("DocstringData Created", {
                "Has Summary": bool(result.summary),
                "Has Description": bool(result.description),
                "Number of Args": len(result.args),
                "Has Returns": bool(result.returns),
                "Number of Raises": len(result.raises),
                "Complexity": result.complexity
            })

            return result

        except KeyError as e:
            self.logger.error(
                "Missing required key in docstring dict",
                extra={"sanitized_info": {"error": str(e)}}
            )
            raise DocumentationError(f"Docstring dictionary missing keys: {e}")

    def process_batch(
        self, doc_entries: Sequence[dict[str, Any]], source_code: str
    ) -> dict[str, str]:
        """Process a batch of documentation entries."""
        try:
            print_phase_header("Processing Documentation Batch")
            tree = ast.parse(source_code)
            print_info(f"Processing {len(doc_entries)} documentation entries")

            processed_entries: list[dict[str, Any]] = []
            for entry in doc_entries:
                try:
                    if "summary" in entry and "name" not in entry:
                        for node in ast.walk(tree):
                            if isinstance(
                                node,
                                (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef),
                            ):
                                docstring = self(entry)
                                processed_entries.append(
                                    {
                                        "name": node.name,
                                        "docstring": docstring.to_dict(),
                                        "type": type(node).__name__,
                                    }
                                )
                                print_info(f"Processed entry for {node.name}")
                                break
                    else:
                        if "docstring" not in entry and "summary" in entry:
                            docstring = self(entry)
                            entry["docstring"] = docstring.to_dict()
                        processed_entries.append(entry)
                        print_info(f"Added entry: {entry.get('name')}")

                except Exception as e:
                    self.logger.error(
                        "Error processing entry",
                        extra={"sanitized_info": {"error": str(e)}}
                    )
                    continue

            if not processed_entries:
                self.logger.error(
                    "No valid entries were processed",
                    extra={"sanitized_info": {"status": "error", "type": "no_entries"}}
                )
                return {"code": source_code, "documentation": ""}

            updated_tree = self._insert_docstrings(tree, processed_entries)
            if not updated_tree:
                self.logger.error(
                    "Failed to update AST with docstrings",
                    extra={"sanitized_info": {"status": "error", "type": "ast_update"}}
                )
                return {"code": source_code, "documentation": ""}

            updated_code = self._generate_code_from_ast(updated_tree)
            if not updated_code:
                self.logger.error(
                    "Failed to generate code from AST",
                    extra={"sanitized_info": {"status": "error", "type": "code_gen"}}
                )
                return {"code": source_code, "documentation": ""}

            documentation = self._generate_documentation(processed_entries)
            
            display_processing_phase("Batch Processing Complete", {
                "Entries Processed": len(processed_entries),
                "Code Updated": bool(updated_code),
                "Documentation Generated": bool(documentation)
            })

            return {"code": updated_code, "documentation": documentation}

        except Exception as e:
            self.logger.error(
                "Unexpected error processing batch",
                extra={"sanitized_info": {"error": str(e)}}
            )
            raise DocumentationError(f"Failed to process batch: {e}")

    def _insert_docstrings(
        self, tree: ast.AST, doc_entries: Sequence[dict[str, Any]]
    ) -> ast.AST | None:
        """Insert docstrings into AST nodes."""
        docstring_map = {
            entry["name"]: entry["docstring"]
            for entry in doc_entries
            if "name" in entry and "docstring" in entry
        }

        class DocstringInserter(ast.NodeTransformer):
            """Inserts docstrings into AST nodes."""
            def visit_Module(self, node: ast.Module) -> ast.Module:
                """Visit module nodes."""
                self.generic_visit(node)
                if "__module__" in docstring_map and not ast.get_docstring(node):
                    docstring_node = ast.Expr(
                        value=ast.Constant(value=docstring_map["__module__"])
                    )
                    node.body.insert(0, docstring_node)
                return node

            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                """Visit function definition nodes."""
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
                """Visit async function definition nodes."""
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
                """Visit class definition nodes."""
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

    def _generate_code_from_ast(self, tree: ast.AST) -> str | None:
        """Generate source code from AST."""
        try:
            if hasattr(ast, "unparse"):  # Python 3.9+
                return ast.unparse(tree)
            else:
                try:
                    import astor
                    return astor.to_source(tree)
                except ImportError:
                    self.logger.error(
                        "astor package not found and Python < 3.9",
                        extra={"sanitized_info": {"status": "error", "type": "import"}}
                    )
                    raise DocumentationError("Cannot generate code: astor not found and Python < 3.9")
        except Exception as e:
            self.logger.error(
                "Error generating code from AST",
                extra={"sanitized_info": {"error": str(e)}}
            )
            raise DocumentationError(f"Failed to generate code from AST: {e}")

    def _generate_documentation(self, doc_entries: Sequence[dict[str, Any]]) -> str:
        """Generate documentation from processed entries."""
        print_phase_header("Generating Documentation")
        doc_parts = ["# API Documentation\n\n"]

        module_entry = next(
            (entry for entry in doc_entries if entry.get("name") == "__module__"), None
        )
        if module_entry and module_entry.get("docstring"):
            docstring = module_entry["docstring"]
            doc_parts.extend([
                "## Module Overview\n\n",
                f"{docstring.get('summary', '')}\n\n",
                f"{docstring.get('description', '')}\n\n"
            ])

        class_entries = [
            entry for entry in doc_entries if entry.get("type") == "ClassDef"
        ]
        if class_entries:
            print_info(f"Generating documentation for {len(class_entries)} classes")
            doc_parts.append("## Classes\n\n")
            for entry in class_entries:
                self._generate_doc_section(doc_parts, entry)

        func_entries = [
            entry
            for entry in doc_entries
            if entry.get("type") in ("FunctionDef", "AsyncFunctionDef")
        ]
        if func_entries:
            print_info(f"Generating documentation for {len(func_entries)} functions")
            doc_parts.append("## Functions\n\n")
            for entry in func_entries:
                self._generate_doc_section(doc_parts, entry)

        return "".join(doc_parts)

    def _generate_doc_section(self, doc_parts: list[str], entry: dict[str, Any]) -> None:
        """Helper function to generate documentation for a single entry."""
        if entry.get("docstring"):
            docstring = entry["docstring"]
            doc_parts.extend([
                f"### {entry['name']}\n\n",
                f"{docstring.get('summary', '')}\n\n",
                f"{docstring.get('description', '')}\n\n"
            ])
            
            if docstring.get("args"):
                doc_parts.append("**Arguments:**\n\n")
                for arg in docstring["args"]:
                    doc_parts.append(f"- `{arg['name']}` (`{arg['type']}`): {arg['description']}\n")
                doc_parts.append("\n")
            
            if docstring.get("returns"):
                returns = docstring["returns"]
                doc_parts.append(f"**Returns:** `{returns['type']}` - {returns['description']}\n\n")
            
            if docstring.get("raises"):
                doc_parts.append("**Raises:**\n\n")
                for exc in docstring["raises"]:
                    doc_parts.append(f"- `{exc['exception']}`: {exc['description']}\n")
                doc_parts.append("\n")
