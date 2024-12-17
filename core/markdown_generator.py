"""
Markdown documentation generator module.
"""

from collections.abc import Sequence
from typing import Any, TypedDict
from datetime import datetime

from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.types import DocumentationData, ExtractedClass, MetricData
from core.exceptions import DocumentationError


class FunctionDict(TypedDict, total=False):
    name: str
    metrics: MetricData
    args: list[dict[str, Any]]
    returns: dict[str, str]


class ConstantDict(TypedDict, total=False):
    name: str
    type: str
    value: str


class MarkdownGenerator:
    """Generates formatted markdown documentation."""

    def __init__(self, correlation_id: str | None = None) -> None:
        """
        Initialize the markdown generator.

        Args:
            correlation_id: Optional correlation ID for tracking related operations.
        """
        self.correlation_id = correlation_id
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__),
            extra={"correlation_id": self.correlation_id},
        )

    def _escape_markdown(self, text: str) -> str:
        """Escape special markdown characters while preserving intended formatting."""
        # Only escape specific characters that would affect table formatting
        table_special_chars = ["|", "\\"]
        for char in table_special_chars:
            text = text.replace(char, "\\" + char)
        return text

    def _format_code_block(self, code: str, language: str = "") -> str:
        """Format code block with proper markdown syntax."""
        return f"```{language}\n{code}\n```"

    def _format_parameter_list(self, params: list[str], max_length: int = 80) -> str:
        """Format parameter list with proper line breaks."""
        if not params:
            return "()"

        # Always return single line for table cells
        return f"({', '.join(params)})"

    def _format_table_header(self, headers: list[str]) -> list[str]:
        """Format a markdown table header with proper alignment."""
        header_row = f"| {' | '.join(headers)} |"
        separator = f"|{'|'.join(['---' for _ in headers])}|"
        return [header_row, separator]

    def _format_table_row(self, columns: list[str], wrap_code: bool = True) -> str:
        """Format a table row with proper escaping and code wrapping."""
        # Clean and format each column
        formatted_cols = []
        for col in columns:
            # Clean whitespace and newlines
            cleaned = str(col).replace("\n", " ").strip()
            escaped = self._escape_markdown(cleaned)
            if wrap_code and escaped != "N/A":
                escaped = f"`{escaped}`"
            formatted_cols.append(escaped)

        return f"| {' | '.join(formatted_cols)} |"

    def _generate_metadata_section(self, file_path: str, module_name: str) -> str:
        """Generate metadata section with file and module info."""
        return f"""---
Module: {module_name}
File: {file_path}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
---

"""

    def _generate_class_tables(self, classes: Sequence[dict[str, Any]]) -> str:
        """Generate markdown tables for classes with improved formatting."""
        if not classes:
            return ""

        tables = ["## Classes\n"]

        # Add class hierarchy information
        for cls_dict in classes:
            cls = ExtractedClass(**cls_dict)
            if cls.inheritance_chain:
                tables.append(f"\n### Class Hierarchy for {cls.name}\n")
                tables.append("```\n" + " -> ".join(cls.inheritance_chain) + "\n```\n")

        # Add interface information
        if any(cls.get("interfaces") for cls in classes):
            tables.append("\n### Implemented Interfaces\n")
            for cls in classes:
                if cls.interfaces:
                    tables.append(f"- `{cls.name}`: {', '.join(cls.interfaces)}\n")

        # Add properties table
        if any(cls.get("property_methods") for cls in classes):
            tables.extend(
                [
                    "\n### Properties\n",
                    "| Class | Property | Type | Access |",
                    "|-------|----------|------|--------|",
                ]
            )
            for cls in classes:
                for cls_dict in classes:
                    for prop in cls_dict.get("property_methods", []):
                        access = "Read/Write" if prop.get("has_setter") else "Read-only"
                        tables.append(
                            f"| `{cls_dict.get('name', 'Unknown')}` | `{prop.get('name', 'Unknown')}` | `{prop.get('type', 'Unknown')}` | {access} |"
                        )
        # Overview table
        tables.extend(
            [
                "## Classes\n",
                "| Class | Description | Complexity |",
                "|-------|-------------|------------|",
            ]
        )

        for cls_dict in classes:
            cls_dict.pop("_logger", None)
            cls = ExtractedClass(**cls_dict)
            description = (
                cls.docstring_info.summary if cls.docstring_info else "No description"
            )
            complexity = self._get_complexity(cls.metrics)
            tables.append(f"| `{cls.name}` | {description} | {complexity} |")

        # Methods table with improved formatting
        tables.extend(
            [
                "\n### Methods\n",
                "| Class | Method | Parameters | Returns | Description |",
                "|-------|--------|------------|---------|-------------|",
            ]
        )

        for cls_dict in classes:
            cls = ExtractedClass(**cls_dict)
            for method in cls.methods:
                docstring_info = method.get_docstring_info()
                desc = docstring_info.summary if docstring_info else "No description"
                desc = desc.replace("\n", " ").strip()

                params_str = self._format_parameter_list(
                    [
                        f"{arg.name}: {arg.type}" if arg.type else arg.name
                        for arg in method.args
                    ]
                )
                returns_str = (
                    method.returns.get("type", "Any") if method.returns else "Any"
                )

                tables.append(
                    f"| `{cls.name}` | `{method.name}` | `{params_str}` | `{returns_str}` | {desc} |"
                )

        return "\n".join(tables)

    def _generate_function_tables(self, functions: Sequence[FunctionDict]) -> str:
        """Generate the functions section."""
        if not functions:
            return ""

        # Improve table formatting
        table_lines = [
            "## Functions\n",
            "| Function | Parameters | Returns | Complexity Score* |",
            "|----------|------------|---------|------------------|",
        ]

        for func in functions:
            params: list[str] = []
            for arg in func.get("args", []):
                param = f"{arg.get('name', '')}: {arg.get('type', 'Any')}"
                if arg.get("default_value"):
                    param += f" = {arg.get('default_value')}"
                params.append(param)

            params_str = self._format_parameter_list(params)
            returns = func.get("returns", {})
            returns_str = returns.get("type", "None") if returns else "None"

            metrics = func.get("metrics", MetricData())
            complexity = self._get_complexity(metrics)
            warning = " (warning)" if complexity > 10 else ""

            # Escape special characters and wrap in code blocks
            func_name = self._escape_markdown(func.get("name", "Unknown"))
            params_str = self._escape_markdown(params_str)
            returns_str = self._escape_markdown(returns_str)

            table_lines.append(
                f"| `{func_name}` | `{params_str}` | `{returns_str}` | {complexity}{warning} |"
            )

        return "\n".join(table_lines)

    def _generate_header(self, module_name: str) -> str:
        """Generate the module header."""
        self.logger.debug(f"Generating header for module_name: {module_name}.")
        return f"# {self._escape_markdown(module_name)}"

    def _generate_toc(self) -> str:
        """Generate enhanced table of contents."""
        return """## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Classes](#classes)
  - [Methods](#methods)
- [Functions](#functions)
- [Constants & Variables](#constants--variables)
- [Dependencies](#dependencies)
- [Recent Changes](#recent-changes)
- [Source Code](#source-code)"""

    def _generate_overview(self, file_path: str, description: str) -> str:
        """Generate the overview section with enhanced module summary."""
        self.logger.debug(f"Generating overview for file_path: {file_path}")

        if not description or description.isspace():
            description = "No description available."
            self.logger.warning(f"No description provided for {file_path}")

        # Enhance the module summary with additional context if available
        module_type = self._infer_module_type(file_path)
        summary_prefix = f"This {module_type} module" if module_type else "This module"

        # Clean up and format description
        description = description.strip()
        if not description.endswith("."):
            description += "."

        # Format the description to start with the summary prefix if it doesn't already
        if not description.lower().startswith(summary_prefix.lower()):
            description = f"{summary_prefix} {description[0].lower()}{description[1:]}"
        # Remove duplicate "module" words
        description = description.replace("module module", "module")

        return f"""## Overview
**File:** `{self._escape_markdown(file_path)}`

**Description:**  
{self._escape_markdown(description)}"""

    def _infer_module_type(self, file_path: str) -> str:
        """Infer the type of module from its name and location."""
        path = file_path.lower()
        if "test" in path:
            return "testing"
        elif "api" in path:
            return "API"
        elif "utils" in path or "helpers" in path:
            return "utility"
        elif "models" in path:
            return "data model"
        elif "views" in path:  # pylint: disable=undefined-variable
            return "view"
        elif "controllers" in path:
            return "controller"
        elif "services" in path:
            return "service"
        return ""

    def _format_table_value(self, value: str) -> str:
        """Format a value for display in markdown tables."""
        if not value or value == "N/A":
            return "N/A"

        # Clean up value first
        value = value.strip()
        if value.startswith("constant"):
            value = value.replace("constant", "").strip()

        # Handle dictionary and list values
        if value.startswith("{") or value.startswith("["):
            # For long complex values, format as Python code block
            if len(value) > 60:
                return f"```python\n{value}\n```"
            return f"`{self._escape_markdown(value)}`"

        # Format other values
        return f"`{self._escape_markdown(value)}`"

    def _generate_constants_table(self, constants: Sequence[ConstantDict]) -> str:
        """Generate the constants section with proper formatting."""
        if not constants:
            return ""  # pylint: disable=undefined-variable

        sections = [
            "## Constants & Variables\n",
            "| Name | Type | Value |",
            "|------|------|--------|",
        ]  # pylint: disable=undefined-variable

        for const in constants:
            name = const.get("name", "Unknown")
            type_str = const.get("type", "Unknown")
            value = self._format_table_value(str(const.get("value", "N/A")))

            sections.append(f"| {name} | {type_str} | {value} |")

        return "\n".join(sections) + "\n"

    def _generate_recent_changes(self, changes: Sequence[dict[str, Any]]) -> str:
        """Generate the recent changes section."""
        if not changes:
            return ""

        change_lines: list[str] = ["## Recent Changes\n\n"]

        for change in changes:
            date = self._escape_markdown(change.get("date", "Unknown Date"))
            description = self._escape_markdown(
                change.get("description", "No description")
            )
            change_lines.append(f"- **{date}**: {description}\n")

        return "\n".join(change_lines)

    def _generate_source_code(self, source_code: str | None) -> str:
        """Generate the source code section."""
        if not source_code:
            self.logger.warning("Source code missing, skipping source code section")
            return ""

        return f"""## Source Code

{self._format_code_block(source_code, "python")}"""

    def _get_complexity(self, metrics: MetricData | dict[str, Any]) -> int:
        """Get cyclomatic complexity from metrics object."""
        if isinstance(metrics, dict):
            return metrics.get("cyclomatic_complexity", 1)
        return getattr(metrics, "cyclomatic_complexity", 1)

    def generate(self, documentation_data: DocumentationData) -> str:
        """Generate markdown documentation."""
        try:
            if not documentation_data:
                raise DocumentationError("Documentation data is None")

            # Validate source code
            if (
                not documentation_data.source_code
                or not documentation_data.source_code.strip()
            ):
                self.logger.error("Source code is missing")
                raise DocumentationError("source_code is required")

            # Generate markdown sections
            markdown_sections = [
                self._generate_metadata_section(
                    str(documentation_data.module_path), documentation_data.module_name
                ),
                self._generate_header(documentation_data.module_name),
                self._generate_toc(),
                self._generate_overview(
                    str(documentation_data.module_path),
                    str(documentation_data.module_summary),
                ),
                self._generate_class_tables(
                    documentation_data.code_metadata.get("classes", [])
                ),
                self._generate_function_tables(
                    documentation_data.code_metadata.get("functions", [])
                ),
                self._generate_constants_table(
                    documentation_data.code_metadata.get("constants", [])
                ),
                self._generate_recent_changes(
                    documentation_data.code_metadata.get("recent_changes", [])
                ),
                self._generate_source_code(documentation_data.source_code),
            ]

            return "\n\n".join(section for section in markdown_sections if section)

        except Exception as e:
            self.logger.error(f"Error generating markdown: {e}")
            self.logger.error(f"Error generating markdown: {e}")
            raise DocumentationError(f"Failed to generate markdown: {e}")
