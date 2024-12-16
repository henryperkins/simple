"""
Markdown documentation generator module.
"""

from collections.abc import Sequence
from typing import Any, TypedDict, cast

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
        """Escape special markdown characters."""
        special_chars = ['|', '*', '_', '[', ']', '(', ')', '#', '`', '>', '+', '-', '.', '!', '{', '}']
        for char in special_chars:
            text = text.replace(char, '\\' + char)
        return text

    def _format_code_block(self, code: str, language: str = "") -> str:
        """Format code block with proper markdown syntax."""
        return f"```{language}\n{code}\n```"

    def _format_parameter_list(self, params: list[str], max_length: int = 80) -> str:
        """Format parameter list with proper line breaks."""
        if not params:
            return "()"
        
        # Single line if short enough
        single_line = f"({', '.join(params)})"
        if len(single_line) <= max_length:
            return single_line
        
        # Multi-line format
        indent = "    "
        param_lines = [params[0]]
        current_line = params[0]
        
        for param in params[1:]:
            if len(current_line + ", " + param) <= max_length:
                current_line += ", " + param
                param_lines[-1] = current_line
            else:
                current_line = param
                param_lines.append(current_line)
        
        return f"(\n{indent}" + f",\n{indent}".join(param_lines) + "\n)"

    def _generate_class_tables(self, classes: Sequence[dict[str, Any]]) -> str:
        """Generate markdown tables for classes."""
        if not classes:
            return ""

        # First table: Class overview
        tables: list[str] = ["## Classes\n\n"]
        tables.append("| Class | Inherits From | Complexity Score* |\n")
        tables.append("|-------|---------------|------------------|\n")

        class_methods: list[str] = ["### Class Methods\n\n"]
        class_methods.append("| Class | Method | Parameters | Returns | Complexity Score* |\n")
        class_methods.append("|-------|---------|------------|---------|------------------|\n")

        for cls_dict in classes:
            cls_dict.pop('_logger', None)
            cls = ExtractedClass(**cls_dict)
            class_name = self._escape_markdown(cls.name)
            
            # Add class to overview table
            total_complexity = sum(
                self._get_complexity(method.metrics)
                for method in cls.methods
            )
            warning = " (warning)" if total_complexity > 20 else ""
            base_class = self._escape_markdown(cls.bases[0] if cls.bases else 'None')
            tables.append(f"| `{class_name}` | `{base_class}` | {total_complexity}{warning} |\n")

            # Add methods to methods table
            for method in cls.methods:
                params: list[str] = []
                for arg in method.args:
                    param = f"{arg.name}: {arg.type}" if arg.type else arg.name
                    if arg.default_value:
                        param += f" = {arg.default_value}"
                    params.append(param)
                
                params_str = self._format_parameter_list(params)
                returns_str = method.returns.get('type', 'None') if method.returns else 'None'
                complexity = self._get_complexity(method.metrics)
                warning = " (warning)" if complexity > 10 else ""
                
                # Escape special characters and wrap in code blocks
                method_name = self._escape_markdown(method.name)
                params_str = self._escape_markdown(params_str)
                returns_str = self._escape_markdown(returns_str)
                
                class_methods.append(
                    f"| `{class_name}` | `{method_name}` | `{params_str}` | `{returns_str}` | {complexity}{warning} |\n"
                )

        return "\n".join(tables) + "\n\n" + "\n".join(class_methods)

    def _generate_function_tables(self, functions: Sequence[FunctionDict]) -> str:
        """Generate the functions section."""
        if not functions:
            return ""

        table_lines: list[str] = ["## Functions\n\n"]
        table_lines.append("| Function | Parameters | Returns | Complexity Score* |\n")
        table_lines.append("|----------|------------|---------|------------------|\n")

        for func in functions:
            params: list[str] = []
            for arg in func.get("args", []):
                param = f"{arg.get('name', '')}: {arg.get('type', 'Any')}"
                if arg.get("default_value"):
                    param += f" = {arg.get('default_value')}"
                params.append(param)
            
            params_str = self._format_parameter_list(params)
            returns = func.get("returns", {})
            returns_str = returns.get('type', 'None') if returns else 'None'

            metrics = func.get("metrics", MetricData())
            complexity = self._get_complexity(metrics)
            warning = " (warning)" if complexity > 10 else ""

            # Escape special characters and wrap in code blocks
            func_name = self._escape_markdown(func.get('name', 'Unknown'))
            params_str = self._escape_markdown(params_str)
            returns_str = self._escape_markdown(returns_str)

            table_lines.append(
                f"| `{func_name}` | `{params_str}` | `{returns_str}` | {complexity}{warning} |\n"
            )

        return "\n".join(table_lines)

    def _generate_header(self, module_name: str) -> str:
        """Generate the module header."""
        self.logger.debug(f"Generating header for module_name: {module_name}.")
        return f"# {self._escape_markdown(module_name)}"

    def _generate_toc(self) -> str:
        """Generate table of contents."""
        return """## Table of Contents
- [Overview](#overview)
- [Classes](#classes)
- [Functions](#functions)
- [Constants and Variables](#constants-and-variables)
- [Recent Changes](#recent-changes)
- [Source Code](#source-code)"""

    def _generate_overview(self, file_path: str, description: str) -> str:
        """Generate the overview section."""
        self.logger.debug(f"Generating overview for file_path: {file_path}")

        if not description or description.isspace():
            description = "No description available."
            self.logger.warning(f"No description provided for {file_path}")

        return f"""## Overview
**File:** `{self._escape_markdown(file_path)}`

**Description:**  
{self._escape_markdown(description)}"""

    def _get_complexity(self, metrics: MetricData | dict[str, Any]) -> int:
        """Get cyclomatic complexity from metrics object."""
        if isinstance(metrics, dict):
            return 1  # Default complexity for dict metrics
        return metrics.cyclomatic_complexity

    def _generate_constants_table(self, constants: Sequence[ConstantDict]) -> str:
        """Generate the constants section."""
        if not constants:
            return ""

        table_lines: list[str] = [
            "## Constants and Variables\n\n",
            "| Name | Type | Value |\n",
            "|------|------|-------|\n",
        ]

        for const in constants:
            name = self._escape_markdown(const.get('name', 'Unknown'))
            type_str = self._escape_markdown(const.get('type', 'Unknown'))
            value = self._escape_markdown(const.get('value', 'Unknown'))
            
            table_lines.append(
                f"| `{name}` | `{type_str}` | `{value}` |\n"
            )

        return "\n".join(table_lines)

    def _generate_recent_changes(self, changes: Sequence[dict[str, Any]]) -> str:
        """Generate the recent changes section."""
        if not changes:
            return ""

        change_lines: list[str] = ["## Recent Changes\n\n"]

        for change in changes:
            date = self._escape_markdown(change.get('date', 'Unknown Date'))
            description = self._escape_markdown(change.get('description', 'No description'))
            change_lines.append(
                f"- **{date}**: {description}\n"
            )

        return "\n".join(change_lines)

    def _generate_source_code(self, source_code: str | None) -> str:
        """Generate the source code section."""
        if not source_code:
            self.logger.warning("Source code missing, skipping source code section")
            return ""

        return f"""## Source Code

{self._format_code_block(source_code, "python")}"""

    def generate(self, documentation_data: DocumentationData) -> str:
        """Generate markdown documentation."""
        try:
            self.logger.debug(
                f"Generating markdown for module: {documentation_data.module_name}"
            )
            self.logger.debug(
                f"DocumentationData content: {documentation_data.to_dict()}"
            )

            if not documentation_data.source_code:
                self.logger.error(
                    "Source code is missing - cannot generate documentation"
                )
                return "# Error: Missing Source Code\n\nDocumentation cannot be generated without source code."

            module_info: dict[str, Any] = {
                "module_name": documentation_data.module_name,
                "file_path": str(documentation_data.module_path),
                "description": documentation_data.module_summary or "No description available.",
                "lines_of_code": len(documentation_data.source_code.splitlines()),
                "dependencies": getattr(documentation_data, 'dependencies', []) or [],
                "changes": getattr(documentation_data, 'changes', []) or []
            }

            markdown_sections: list[str] = [
                self._generate_header(str(module_info["module_name"])),
                self._generate_toc(),
                self._generate_overview(
                    str(module_info["file_path"]),
                    str(module_info["description"])
                ),
                self._generate_class_tables(
                    cast(list[dict[str, Any]], documentation_data.code_metadata.get("classes", []))
                ),
                self._generate_function_tables(
                    cast(list[FunctionDict], documentation_data.code_metadata.get("functions", []))
                ),
                self._generate_constants_table(
                    cast(list[ConstantDict], documentation_data.code_metadata.get("constants", []))
                ),
                self._generate_recent_changes(module_info["changes"]),
                self._generate_source_code(documentation_data.source_code),
                "*Please note: Complexity scores provide an overview of the logic complexity and performance characteristics. A higher score may indicate a need for optimization. Methods with scores above 10 are marked with (warning).*"
            ]

            markdown = "\n\n".join(section for section in markdown_sections if section)
            self.logger.debug("Generated documentation successfully")
            return markdown

        except DocumentationError as de:
            error_msg = f"DocumentationError: {de} in markdown generation with correlation ID: {self.correlation_id}"
            self.logger.error(error_msg, extra={"correlation_id": self.correlation_id})
            return f"# Error Generating Documentation\n\nDocumentationError: {de}"
        except Exception as e:
            error_msg = f"Unexpected error: {e} in markdown generation with correlation ID: {self.correlation_id}"
            self.logger.error(
                error_msg, exc_info=True, extra={"correlation_id": self.correlation_id}
            )
            return f"# Error Generating Documentation\n\nAn error occurred: {e}"
