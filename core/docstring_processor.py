"""Processes and validates docstrings."""

import json
import time
from typing import Any, Tuple, List, Dict, Optional

from docstring_parser import DocstringStyle, parse
from jsonschema import validate, ValidationError

from core.console import display_metrics, print_status, print_error, print_success
from core.exceptions import DataValidationError
from core.logger import CorrelationLoggerAdapter, LoggerSetup
from core.metrics_collector import MetricsCollector
from core.types.base import ProcessingResult
from core.types.docstring import DocstringData
from utils import log_and_raise_error


class DocstringProcessor:
    """Processes and validates docstrings."""

    def __init__(
        self, correlation_id: Optional[str] = None, schema_path: Optional[str] = None
    ) -> None:
        """Initializes the DocstringProcessor."""
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__), extra={"correlation_id": correlation_id}
        )
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)
        self.docstring_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_lines": 0,
            "avg_length": 0,
        }
        self.docstring_stats["common_issues"] = {
            "Missing Args": 0,
            "Missing Returns": 0,
            "Missing Summary": 0,
        }
        self.correlation_id = correlation_id
        self.schema_path = schema_path or "core/schemas/docstring_schema.json"
        self._load_schema()

    def _load_schema(self) -> None:
        """Loads the docstring schema from a file."""
        try:
            with open(self.schema_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
        except FileNotFoundError as e:
            log_and_raise_error(
                self.logger,
                e,
                DataValidationError,
                f"Schema file not found: {self.schema_path}",
                self.correlation_id,
            )
        except json.JSONDecodeError as e:
            log_and_raise_error(
                self.logger,
                e,
                DataValidationError,
                f"Error decoding JSON schema: {self.schema_path}",
                self.correlation_id,
            )

    def parse(self, docstring: str, context: Optional[str] = None) -> DocstringData:
        """Parses a docstring string into structured data."""
        try:
            result = self._parse_docstring_content(docstring, context)
            return DocstringData(
                summary=result["summary"],
                description=result["description"],
                args=result["args"],
                returns=result["returns"],
                raises=result["raises"],
                complexity=result["complexity"],
            )
        except Exception as e:
            log_and_raise_error(
                self.logger,
                e,
                DataValidationError,
                f"Error parsing docstring{f' for {context}' if context else ''}",
                self.correlation_id,
            )
            return DocstringData(
                summary="Failed to parse docstring",
                description=str(e),
                args=[],
                returns={"type": "Any", "description": ""},
                raises=[],
                complexity=1,
            )

    def _parse_docstring_content(
        self, docstring: str, context: Optional[str] = None
    ) -> dict[str, Any]:
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

        parsed_docstring = None
        parsed_style = None
        try:
            parsed_docstring = parse(docstring_str, style=DocstringStyle.AUTO)
            parsed_style = "AUTO"
        except Exception:
            for style in [DocstringStyle.GOOGLE, DocstringStyle.REST]:
                try:
                    parsed_docstring = parse(docstring_str, style=style)
                    parsed_style = style.name
                    break
                except Exception as e:
                    self.logger.debug(
                        f"Failed to parse with style {style}: {e}",
                        extra={"style": style},
                    )
            if parsed_docstring is None:
                self.docstring_stats["failed"] += 1
                self.logger.warning(
                    "Failed to parse docstring with any style",
                    extra={"docstring": docstring_str[:50]},
                )
                print_error(
                    f"Failed to parse docstring{f' for {context}' if context else ''}."
                )
                return {
                    "summary": docstring_str,
                    "description": "",
                    "args": [],
                    "returns": {"type": "Any", "description": ""},
                    "raises": [],
                    "complexity": 1,
                }
            else:
                self.docstring_stats["successful"] += 1
                if context:
                    print_success(
                        f"Successfully parsed docstring for {context} (Style: {parsed_style})"
                    )
                else:
                    print_success(
                        f"Successfully parsed docstring (Style: {parsed_style})"
                    )

        if self.docstring_stats["total_processed"] % 10 == 0:
            self._display_common_issues()  # Corrected line

        docstring_data = {
            "summary": parsed_docstring.short_description or "No summary available.",
            "description": parsed_docstring.long_description
            or "No description provided.",
            "args": [
                {
                    "name": param.arg_name or "",
                    "type": param.type_name or "Any",
                    "description": param.description or "",
                    "nested": [],
                }
                for param in parsed_docstring.params
            ],
            "returns": {
                "type": (
                    parsed_docstring.returns.type_name
                    if parsed_docstring.returns
                    else "Any"
                ),
                "description": (
                    parsed_docstring.returns.description
                    if parsed_docstring.returns
                    else ""
                ),
            },
            "raises": [
                {
                    "exception": exc.type_name or "Exception",
                    "description": exc.description or "",
                }
                for exc in parsed_docstring.raises
            ],
            "complexity": 1,
        }

        # Track common issues
        self._track_common_issues(docstring_data)
        return docstring_data

    def validate(
        self, docstring_data: Dict[str, Any], context: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """Validates a docstring dictionary against the schema."""
        try:
            is_valid, errors = self._validate_docstring_data(docstring_data)
            if is_valid:
                self.metrics_collector.collect_validation_metrics(success=True)
                if context:
                    print_success(f"Docstring validation passed for {context}")
                else:
                    print_success("Docstring validation passed")
                return True, []
            else:
                self.metrics_collector.collect_validation_metrics(success=False)
                if context:
                    print_error(f"Docstring validation failed for {context}: {errors}")
                else:
                    print_error(f"Docstring validation failed: {errors}")
                return False, errors
        except Exception as e:
            self.metrics_collector.collect_validation_metrics(success=False)
            log_and_raise_error(
                self.logger,
                e,
                DataValidationError,
                "Error during docstring validation",
                self.correlation_id,
                context=context,
            )
            return False, [str(e)]

    def _validate_docstring_data(
        self, docstring_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Internal method to validate docstring data against the schema."""
        try:
            validate(instance=docstring_data, schema=self.schema)
            return True, []
        except ValidationError as e:
            return False, [str(e)]

    def _track_common_issues(self, docstring_data: Dict[str, Any]) -> None:
        """Track common issues in docstrings."""
        if not docstring_data.get("args"):
            self.docstring_stats["common_issues"]["Missing Args"] += 1
        if not docstring_data.get("returns"):
            self.docstring_stats["common_issues"]["Missing Returns"] += 1
        if not docstring_data.get("summary"):
            self.docstring_stats["common_issues"]["Missing Summary"] += 1

    def _display_common_issues(self) -> None:
        """Display common docstring issues."""
        display_metrics(
            self.docstring_stats["common_issues"],
            title="Common Docstring Issues",
        )

    async def process_docstring(
        self, docstring: str, context: Optional[str] = None
    ) -> ProcessingResult:
        """Process a docstring and return structured results."""
        start_time = time.time()
        try:
            print_status(
                f"Processing docstring{f' for {context}' if context else ''}..."
            )
            parsed_data = self.parse(docstring, context)
            is_valid, errors = self.validate(parsed_data.to_dict(), context)

            if not is_valid:
                log_and_raise_error(
                    self.logger,
                    DataValidationError(f"Docstring validation failed: {errors}"),
                    DataValidationError,
                    "Docstring validation failed",
                    self.correlation_id,
                    errors=errors,
                )

            processing_time = time.time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="docstring_processing",
                success=True,
                duration=processing_time,
                metadata={
                    "lines": len(docstring.splitlines()),
                    "length": len(docstring),
                    "has_args": bool(parsed_data.args),
                    "has_returns": bool(parsed_data.returns.get("description")),
                    "has_raises": bool(parsed_data.raises),
                },
            )

            return ProcessingResult(
                content=parsed_data.to_dict(),
                usage={},  # No token usage for docstring processing
                metrics={
                    "processing_time": processing_time,
                    "validation_success": True,
                },
                validation_status=True,
                validation_errors=[],
                schema_errors=[],
            )
        except Exception as e:
            processing_time = time.time() - start_time
            log_and_raise_error(
                self.logger,
                e,
                DataValidationError,
                "Error processing docstring",
                self.correlation_id,
            )
            return ProcessingResult(
                content={},
                usage={},
                metrics={
                    "processing_time": processing_time,
                    "validation_success": False,
                },
                validation_status=False,
                validation_errors=[str(e)],
                schema_errors=[],
            )
