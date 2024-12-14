"""Processes and validates docstrings."""

import ast
import json
import os
from collections.abc import Sequence
from typing import Any, cast, TypedDict

try:
    from docstring_parser import parse, DocstringStyle, Docstring
except ImportError:
    print("Warning: docstring_parser not found. Install with: pip install docstring-parser")
    raise

from core.console import (
    print_info,
    print_error,
    display_validation_results,
    display_processing_phase
)
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.exceptions import DocumentationError, ValidationError
from core.types.base import DocstringData
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

    def _load_schema(self, schema_name: str) -> dict[str, Any]:
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

    def __call__(self, docstring: dict[str, Any] | str) -> DocstringData:
        """Processes a docstring, either as a dictionary or a string."""
        try:
            if isinstance(docstring, dict):
                return self._process_dict_docstring(docstring)
            return self.parse(docstring)
        except Exception as e:
            self.logger.error(f"Unexpected error during parsing: {e}")
            raise

    def parse(self, docstring: str) -> DocstringData:
        """Parse a docstring into structured data with validation."""
        try:
            parsed_data = self._parse_docstring_content(docstring)

            # Ensure parsed_data is a dictionary before unpacking
            if isinstance(parsed_data, DocstringData):
                parsed_data = parsed_data.to_dict()

            # Add default values for required fields with non-empty description
            parsed_data.setdefault("description", "No description provided.")
            if not parsed_data["description"] or not parsed_data["description"].strip():
                parsed_data["description"] = "No description provided."

            parsed_data.setdefault("summary", "No summary available.")
            if not parsed_data["summary"] or not parsed_data["summary"].strip():
                parsed_data["summary"] = "No summary available."

            parsed_data.setdefault("args", [])
            parsed_data.setdefault("returns", {"type": "Any", "description": "No return value documented."})
            parsed_data.setdefault("raises", [])

            # Create and validate DocstringData instance
            docstring_data = DocstringData(**parsed_data)
            is_valid, validation_errors = docstring_data.validate()

            if not is_valid:
                self.logger.error(f"Docstring validation failed: {validation_errors}")
                raise ValidationError(f"Invalid docstring format: {validation_errors}")

            return docstring_data

        except Exception as e:
            self.logger.error(f"Error parsing docstring: {e}")
            raise ValidationError(f"Failed to parse docstring: {e}")

    def _parse_docstring_content(self, docstring: str) -> dict[str, Any]:
        """Parse docstring content into structured format."""
        docstring_str = docstring.strip()
        display_processing_phase("Docstring Analysis", {
            "Format": "String",
            "Length": len(docstring_str),
            "Lines": len(docstring_str.splitlines())
        })

        # Try parsing with AUTO style first, then fall back to specific styles if needed
        parsed_docstring: Docstring
        try:
            parsed_docstring = parse(docstring_str, style=DocstringStyle.AUTO)
        except Exception:
            try:
                parsed_docstring = parse(docstring_str, style=DocstringStyle.GOOGLE)
            except Exception:
                try:
                    parsed_docstring = parse(docstring_str, style=DocstringStyle.REST)
                except Exception as e:
                    self.logger.error(f"Failed to parse docstring: {e}")
                    # Return minimal DocstringData with just the raw docstring as summary
                    return {
                        "summary": docstring_str,
                        "description": "",
                        "args": [],
                        "returns": {"type": "Any", "description": ""},
                        "raises": [],
                        "complexity": 1
                    }
        
        # Debug logging
        self.logger.debug(f"Raw docstring: {docstring_str}")
        self.logger.debug(f"Parsed docstring: {parsed_docstring}")
        if parsed_docstring.params:
            self.logger.debug(f"Found {len(parsed_docstring.params)} parameters")
            for param in parsed_docstring.params:
                self.logger.debug(f"Parameter: {param.arg_name} ({param.type_name}): {param.description}")

        # Extract data from parsed docstring
        args: list[dict[str, str | list[dict[str, str]]]] = []
        for param in parsed_docstring.params:
            arg: dict[str, str | list[dict[str, str]]] = {
                "name": param.arg_name or "",
                "type": param.type_name or "Any",
                "description": param.description or "",
                "nested": []  # Add empty list for nested parameters
            }
            self.logger.debug(f"Adding argument: {arg}")
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
            self.logger.debug(f"Adding exception: {exc_dict}")
            raises.append(exc_dict)

        result = {
            "summary": parsed_docstring.short_description or "No summary available.",
            "description": parsed_docstring.long_description or "No description provided.",
            "args": args,
            "returns": returns_dict,
            "raises": raises,
            "complexity": 1,
        }

        # Display validation results
        display_validation_results({
            "Summary": bool(result["summary"]),
            "Description": bool(result["description"]),
            "Arguments": bool(result["args"]),
            "Returns": bool(result["returns"]),
            "Exceptions": bool(result["raises"])
        }, {
            "Arguments": f"Count: {len(args)}",
            "Exceptions": f"Count: {len(raises)}"
        })

        return result

    def _process_dict_docstring(self, docstring_dict: dict[str, Any]) -> DocstringData:
        """Process a docstring provided as a dictionary."""
        try:
            print_info("Validating docstring dictionary structure...")
            self._validate_dict_docstring(docstring_dict)
            return self._create_docstring_data_from_dict(docstring_dict)
        except Exception as e:
            self.logger.error(f"Error processing dictionary docstring: {e}")
            raise

    def _validate_dict_docstring(self, docstring_dict: dict[str, Any]) -> bool:
        """Validate a docstring dictionary against the schema."""
        try:
            validate(instance=docstring_dict, schema=self.docstring_schema)
            print_info("Docstring dictionary validated successfully.")
            return True
        except ValidationError as e:
            print_error(f"Docstring dictionary validation failed: {e}")
            raise DocumentationError(f"Docstring dictionary validation failed: {e}")
        except Exception as e:
            print_error(f"Unexpected error during validation: {e}")
            raise DocumentationError(f"Unexpected error during validation: {e}")

    def _create_docstring_data_from_dict(
        self, docstring_dict: dict[str, Any]
    ) -> DocstringData:
        """Create DocstringData from a validated dictionary."""
        try:
            # Extract only the fields we want from the input dictionary
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
            return DocstringData(
                summary=cleaned_dict["summary"],
                description=cleaned_dict["description"],
                args=cleaned_dict["args"],
                returns=cast(ReturnsDict, cleaned_dict["returns"]),
                raises=cleaned_dict["raises"],
                complexity=cleaned_dict["complexity"]
            )
        except KeyError as e:
            self.logger.warning(f"Missing required key in docstring dict: {e}")
            raise DocumentationError(f"Docstring dictionary missing keys: {e}")

    def process_batch(
        self, doc_entries: Sequence[dict[str, Any]], source_code: str
    ) -> dict[str, str]:
        """Process a batch of documentation entries."""
        try:
            tree = ast.parse(source_code)
            self.logger.debug(f"Processing {len(doc_entries)} documentation entries")

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
                                self.logger.debug(
                                    f"Created processed entry for {node.name}"
                                )
                                break
                    else:
                        if "docstring" not in entry and "summary" in entry:
                            docstring = self(entry)
                            entry["docstring"] = docstring.to_dict()
                        processed_entries.append(entry)
                        self.logger.debug(f"Added entry with name: {entry.get('name')}")

                except Exception as e:
                    self.logger.error(f"Error processing entry: {e}")
                    continue

            if not processed_entries:
                self.logger.error("No valid entries were processed")
                return {"code": source_code, "documentation": ""}

            updated_tree = self._insert_docstrings(tree, processed_entries)
            if not updated_tree:
                self.logger.error("Failed to update AST with docstrings")
                return {"code": source_code, "documentation": ""}

            updated_code = self._generate_code_from_ast(updated_tree)
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
                    docstring_node = ast.Expr(value=ast.Str(s=docstring))
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
                    docstring_node = ast.Expr(value=ast.Str(s=docstring))
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
                    docstring_node = ast.Expr(value=ast.Str(s=docstring))
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
                import astor
                return astor.to_source(tree)
        except Exception as e:
            self.logger.error(f"Error generating code from AST: {e}")
            raise DocumentationError(f"Failed to generate code from AST: {e}")

    def _generate_documentation(self, doc_entries: Sequence[dict[str, Any]]) -> str:
        """Generate documentation from processed entries."""
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
            doc_parts.append("## Classes\n\n")
            for entry in class_entries:
                self._generate_doc_section(doc_parts, entry)

        func_entries = [
            entry
            for entry in doc_entries
            if entry.get("type") in ("FunctionDef", "AsyncFunctionDef")
        ]
        if func_entries:
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
