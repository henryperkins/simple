"""Processes and validates docstrings."""

import json
import os
from typing import Any, Union, cast, TypedDict

try:
    from docstring_parser import parse, DocstringStyle, Docstring
except ImportError:
    print(
        "Warning: docstring_parser not found. Install with: pip install docstring-parser"
    )
    raise

from core.console import display_metrics
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.types.docstring import DocstringData
from core.metrics_collector import MetricsCollector
from jsonschema import validate, ValidationError


class ReturnsDict(TypedDict):
    """Return type dict."""

    type: str
    description: str


class DocstringProcessor:
    """Processes and validates docstrings."""

    def __init__(self, correlation_id: str | None = None) -> None:
        """Initializes the DocstringProcessor."""
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__), extra={"correlation_id": correlation_id}
        )
        self.docstring_schema: dict[str, Any] = self._load_schema(
            "docstring_schema.json"
        )
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)
        self.docstring_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_lines": 0,
            "avg_length": 0,
        }

    def _load_schema(self, schema_name: str) -> dict[str, Any]:
        """Loads a JSON schema for validation."""

        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "schemas", schema_name
        )
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            self.logger.info(
                "Schema loaded successfully",
                extra={"schema_name": schema_name, "status": "success"},
            )
            return schema
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(
                f"Error loading schema: {e}",
                extra={"schema_name": schema_name, "status": "error"},
            )
            raise

    def parse(self, docstring: str) -> DocstringData:
        """Parses a docstring string into structured data."""

        result = self._parse_docstring_content(docstring)
        return DocstringData(**result)

    def _parse_docstring_content(self, docstring: str) -> dict[str, Any]:
        """Parses docstring content into a structured dictionary."""

        docstring_str = docstring.strip()
        lines = len(docstring_str.splitlines())
        length = len(docstring_str)

        self.docstring_stats["total_processed"] += 1
        self.docstring_stats["total_lines"] += lines
        self.docstring_stats["avg_length"] = (
            self.docstring_stats["avg_length"]
            * (self.docstring_stats["total_processed"] - 1)
            + length
        ) // self.docstring_stats["total_processed"]

        try:
            parsed_docstring = parse(docstring_str, style=DocstringStyle.AUTO)
            self.docstring_stats["successful"] += 1
        except Exception:
            for style in [DocstringStyle.GOOGLE, DocstringStyle.REST]:
                try:
                    parsed_docstring = parse(docstring_str, style=style)
                    self.docstring_stats["successful"] += 1
                    break  # Exit loop if parsing successful
                except Exception as e:
                    self.logger.debug(
                        f"Failed to parse with style {style}: {e}",
                        extra={"style": style},
                    )
            else:  # No break, all styles failed
                self.docstring_stats["failed"] += 1
                self.logger.warning(
                    "Failed to parse docstring with any style",
                    extra={"docstring": docstring_str[:50]},
                )
                return {  # Return a default DocstringData dictionary
                    "summary": docstring_str,
                    "description": "",
                    "args": [],
                    "returns": {"type": "Any", "description": ""},
                    "raises": [],
                    "complexity": 1,
                }

        if self.docstring_stats["total_processed"] % 10 == 0:
            self._display_docstring_stats()

        args = [
            {
                "name": param.arg_name or "",
                "type": param.type_name or "Any",
                "description": param.description or "",
                "nested": [],  # Placeholder for nested arguments
            }
            for param in parsed_docstring.params
        ]

        returns_dict: ReturnsDict = {"type": "Any", "description": ""}
        if parsed_docstring.returns:
            returns_dict["type"] = parsed_docstring.returns.type_name or "Any"
            returns_dict["description"] = parsed_docstring.returns.description or ""

        raises = [
            {
                "exception": exc.type_name or "Exception",
                "description": exc.description or "",
            }
            for exc in parsed_docstring.raises
        ]

        return {
            "summary": parsed_docstring.short_description or "No summary available.",
            "description": parsed_docstring.long_description
            or "No description provided.",
            "args": args,
            "returns": returns_dict,
            "raises": raises,
            "complexity": 1,  # Placeholder for complexity calculation
        }

    def _display_docstring_stats(self) -> None:
        """Displays current docstring processing statistics."""

        display_metrics(
            {
                "Total Processed": self.docstring_stats["total_processed"],
                "Successfully Parsed": self.docstring_stats["successful"],
                "Failed to Parse": self.docstring_stats["failed"],
                "Average Length": f"{self.docstring_stats['avg_length']}",
                "Total Lines": self.docstring_stats["total_lines"],
                "Success Rate": f"{(self.docstring_stats['successful'] / self.docstring_stats['total_processed'] * 100):.1f}%",
            },
            title="Docstring Processing Statistics",
        )

    def validate(self, docstring_data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validates a docstring dictionary against the schema."""

        try:
            validate(instance=docstring_data, schema=self.docstring_schema)
            return True, []
        except ValidationError as e:
            return False, [str(e)]
