"""Processes and validates docstrings."""

import json
import time
from typing import Any, Tuple, List, Dict, Optional

from docstring_parser import DocstringStyle, parse
from jsonschema import validate, ValidationError

from core.console import display_metrics
from core.exceptions import DataValidationError
from core.logger import CorrelationLoggerAdapter, LoggerSetup
from core.metrics_collector import MetricsCollector
from core.types.base import ProcessingResult
from core.types.docstring import DocstringData


class DocstringProcessor:
    """Processes and validates docstrings."""

    def __init__(self, correlation_id: Optional[str] = None, schema_path: Optional[str] = None) -> None:
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
        self.correlation_id = correlation_id
        self.schema_path = schema_path or "core/schemas/docstring_schema.json"
        self._load_schema()

    def _load_schema(self) -> None:
        """Loads the docstring schema from a file."""
        try:
            with open(self.schema_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
        except FileNotFoundError as e:
            self.logger.error(f"Schema file not found: {self.schema_path} - {e}", exc_info=True)
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON schema: {self.schema_path} - {e}", exc_info=True)
            raise

    def parse(self, docstring: str) -> DocstringData:
        """Parses a docstring string into structured data."""
        try:
            result = self._parse_docstring_content(docstring)
            return DocstringData(
                summary=result["summary"],
                description=result["description"],
                args=result["args"],
                returns=result["returns"],
                raises=result["raises"],
                complexity=result["complexity"]
            )
        except Exception as e:
            self.logger.error(f"Error parsing docstring: {e}", exc_info=True)
            return DocstringData(
                summary="Failed to parse docstring",
                description=str(e),
                args=[],
                returns={"type": "Any", "description": ""},
                raises=[],
                complexity=1
            )

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
                    break
                except Exception as e:
                    self.logger.debug(
                        f"Failed to parse with style {style}: {e}",
                        extra={"style": style},
                    )
            else:
                self.docstring_stats["failed"] += 1
                self.logger.warning(
                    "Failed to parse docstring with any style",
                    extra={"docstring": docstring_str[:50]},
                )
                return {
                    "summary": docstring_str,
                    "description": "",
                    "args": [],
                    "returns": {"type": "Any", "description": ""},
                    "raises": [],
                    "complexity": 1,
                }

        if self.docstring_stats["total_processed"] % 10 == 0:
            self._display_docstring_stats()

        return {
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
                "type": parsed_docstring.returns.type_name if parsed_docstring.returns else "Any",
                "description": parsed_docstring.returns.description if parsed_docstring.returns else ""
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

    def validate(self, docstring_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates a docstring dictionary against the schema."""
        try:
            validate(instance=docstring_data, schema=self.schema)
            self.metrics_collector.collect_validation_metrics(success=True)
            return True, []
        except ValidationError as e:
            self.metrics_collector.collect_validation_metrics(success=False)
            return False, [str(e)]

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

    async def process_docstring(
        self, 
        docstring: str
    ) -> ProcessingResult:
        """Process a docstring and return structured results."""
        start_time = time.time()
        try:
            parsed_data = self.parse(docstring)
            is_valid, errors = self.validate(parsed_data.to_dict())
            
            if not is_valid:
                raise DataValidationError(f"Docstring validation failed: {errors}")

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
                    "has_raises": bool(parsed_data.raises)
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
                schema_errors=[]
            )
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing docstring: {e}", exc_info=True)
            await self.metrics_collector.track_operation(
                operation_type="docstring_processing",
                success=False,
                duration=processing_time,
                metadata={"error": str(e)},
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
                schema_errors=[]
            )