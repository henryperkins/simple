"""Markdown documentation generator module."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MarkdownConfig:
    """Configuration for markdown generation."""
    include_toc: bool = True
    include_timestamp: bool = True
    code_language: str = "python"
    include_source: bool = True

class MarkdownGenerator:
    """Generates formatted markdown documentation."""

    def __init__(self, config: Optional[MarkdownConfig] = None):
        """Initialize the markdown generator."""
        self.config = config or MarkdownConfig()

    def generate(self, context: Dict[str, Any]) -> str:
        """Generate complete markdown documentation."""
        sections = [
            self._generate_header(context),
            self._generate_overview(context),
            self._generate_class_tables(context),
            self._generate_function_tables(context),
            self._generate_constants_table(context),
            self._generate_changes(context),
            self._generate_source_code(context)
        ]
        
        return "\n\n".join(filter(None, sections))

    def _generate_header(self, context: Dict[str, Any]) -> str:
        """Generate the module header."""
        return f"# Module: {context['module_name']}"

    def _generate_overview(self, context: Dict[str, Any]) -> str:
        """Generate the overview section."""
        return "\n".join([
            "## Overview",
            f"**File:** `{context['file_path']}`",
            f"**Description:** {context['description']}"
        ])

    def _generate_class_tables(self, context: Dict[str, Any]) -> str:
        """Generate the classes section with tables."""
        if not context.get('classes'):
            return ""

        # Main classes table
        classes_table = [
            "## Classes",
            "",
            "| Class | Inherits From | Complexity Score* |",
            "|-------|---------------|------------------|"
        ]

        # Methods table
        methods_table = [
            "### Class Methods",
            "",
            "| Class | Method | Parameters | Returns | Complexity Score* |",
            "|-------|--------|------------|---------|------------------|"
        ]

        for cls in context['classes']:
            # Access attributes directly
            complexity = cls.metrics.get('complexity', 0)
            warning = " ⚠️" if complexity > 10 else ""
            bases = ", ".join(cls.bases)
            classes_table.append(
                f"| `{cls.name}` | `{bases or 'None'}` | {complexity}{warning} |"
            )

            # Add methods to methods table
            for method in cls.methods:
                method_complexity = method.metrics.get('complexity', 0)
                method_warning = " ⚠️" if method_complexity > 10 else ""
                params = ", ".join(
                    f"{arg.name}: {arg.type}" for arg in method.args
                )
                methods_table.append(
                    f"| `{cls.name}` | `{method.name}` | "
                    f"`({params})` | `{method.return_type}` | "
                    f"{method_complexity}{method_warning} |"
                )

        return "\n".join(classes_table + [""] + methods_table)

    def _generate_function_tables(self, context: Dict[str, Any]) -> str:
        """Generate the functions section."""
        if not context.get('functions'):
            return ""

        lines = [
            "## Functions",
            "",
            "| Function | Parameters | Returns | Complexity Score* |",
            "|----------|------------|---------|------------------|"
        ]

        for func in context['functions']:
            complexity = func.metrics.get('complexity', 0)
            warning = " ⚠️" if complexity > 10 else ""
            params = ", ".join(
                f"{arg.name}: {arg.type}" + 
                (f" = {arg.default_value}" if arg.default_value else "")
                for arg in func.args
            )
            lines.append(
                f"| `{func.name}` | `({params})` | "
                f"`{func.return_type}` | {complexity}{warning} |"
            )

        return "\n".join(lines)

    def _generate_constants_table(self, context: Dict[str, Any]) -> str:
        """Generate the constants section."""
        if not context.get('constants'):
            return ""

        lines = [
            "## Constants and Variables",
            "",
            "| Name | Type | Value |",
            "|------|------|-------|"
        ]

        for const in context['constants']:
            lines.append(
                f"| `{const['name']}` | `{const['type']}` | `{const['value']}` |"
            )

        return "\n".join(lines)

    def _generate_changes(self, context: Dict[str, Any]) -> str:
        """Generate the recent changes section."""
        if not context.get('changes'):
            return ""

        lines = ["## Recent Changes"]
        
        for change in context.get('changes', []):
            date = change.get('date', datetime.now().strftime('%Y-%m-%d'))
            description = change.get('description', '')
            lines.append(f"- [{date}] {description}")

        return "\n".join(lines)

    def _generate_source_code(self, context: Dict[str, Any]) -> str:
        """Generate the source code section."""
        if not self.config.include_source or not context.get('source_code'):
            return ""

        complexity_scores = []
        
        # Collect complexity scores from functions and methods
        for func in context.get('functions', []):
            complexity = func.metrics.get('complexity', 0)
            warning = " ⚠️" if complexity > 10 else ""
            complexity_scores.append(f"    {func.name}: {complexity}{warning}")

        for cls in context.get('classes', []):
            for method in cls.methods:
                complexity = method.metrics.get('complexity', 0)
                warning = " ⚠️" if complexity > 10 else ""
                complexity_scores.append(
                    f"    {method.name}: {complexity}{warning}"
                )

        docstring = f'"""Module for handling {context.get("description", "[description]")}.\n\n'
        if complexity_scores:
            docstring += "Complexity Scores:\n" + "\n".join(complexity_scores) + '\n'
        docstring += '"""\n\n'

        return "\n".join([
            "## Source Code",
            f"```{self.config.code_language}",
            docstring + context['source_code'],
            "```"
        ])
