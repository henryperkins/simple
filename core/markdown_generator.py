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
        sections = [
            self._generate_header(context),
            self._generate_overview(context),
            self._generate_ai_doc_section(context),
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
                    f"{arg.name}: {arg.type or 'Any'}" + 
                    (f" = {arg.default_value}" if arg.default_value else "")
                    for arg in method.args
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
    
    def _generate_ai_doc_section(self, context: Dict[str, Any]) -> str:
        """
        Generates the AI documentation section of the markdown document.
        This method processes the AI-generated documentation from the context and formats it
        into a structured markdown section including summary, description, arguments,
        return values, and exceptions.
        Args:
            context (Dict[str, Any]): A dictionary containing the 'ai_documentation' key
                with nested documentation details including summary, description,
                arguments, returns, and raises information.
        Returns:
            str: A formatted markdown string containing the AI documentation section.
                Returns an empty string if no AI documentation is present in the context.
        Examples of context structure:
            {
                'ai_documentation': {
                    'summary': 'Function summary',
                    'description': 'Detailed description',
                    'args': [
                        {'name': 'arg1', 'type': 'str', 'description': 'arg1 description'}
                    ],
                    'returns': {'type': 'str', 'description': 'return description'},
                    'raises': [
                        {'exception': 'ValueError', 'description': 'error description'}
                }
            }
        """
        ai_docs = context.get('ai_documentation', {})
        if ai_docs:
            sections = [
                "## AI-Generated Documentation\n\n",
                "**Summary:** " + (ai_docs.get('summary', '') or "No summary provided") + "\n\n",
                "**Description:** " + (ai_docs.get('description', '') or "No description provided") + "\n\n"
            ]
            
            if ai_docs.get('args'):
                sections.append("**Arguments:**")
                for arg in ai_docs['args']:
                    sections.append(f"- **{arg['name']}** ({arg['type']}): {arg['description']}")
                sections.append("\n")

            if ai_docs.get('returns'):
                sections.append(f"**Returns:** {ai_docs['returns']['type']} - {ai_docs['returns']['description']}\n\n")

            if ai_docs.get('raises'):
                sections.append("**Raises:**")
                for raise_ in ai_docs['raises']:
                    sections.append(f"- **{raise_['exception']}**: {raise_['description']}")
                sections.append("\n")

            return "\n".join(sections)
        return ""

    def _format_ai_docs(self, ai_docs: Dict[str, Any]) -> str:
        """
        Formats AI-generated documentation into a markdown string.

        Args:
            ai_docs (Dict[str, Any]): A dictionary containing AI-generated documentation with the following possible keys:
                - summary: A brief summary of the code
                - description: A detailed description
                - args: List of argument dictionaries with name, type and description
                - returns: Dictionary with return type and description
                - raises: List of exception dictionaries with exception name and description

        Returns:
            str: Formatted markdown string with sections for summary, description, arguments, returns and raises where available

        Example structure of ai_docs:
            {
                'summary': 'Brief summary',
                'description': 'Detailed description',
                'args': [
                    {'name': 'arg1', 'type': 'str', 'description': 'arg1 description'}
                ],
                'returns': {'type': 'str', 'description': 'return description'},
                'raises': [
                    {'exception': 'ValueError', 'description': 'error description'}
                ]
            }
        """
        sections = []
        if ai_docs.get('summary'):
            sections.append(f"**Summary:** {ai_docs['summary']}")
        if ai_docs.get('description'):
            sections.append(f"**Description:** {ai_docs['description']}")
        if ai_docs.get('args'):
            sections.append("**Arguments:**")
            for arg in ai_docs['args']:
                sections.append(f"- **{arg['name']}** ({arg['type']}): {arg['description']}")
        if ai_docs.get('returns'):
            sections.append(f"**Returns:** {ai_docs['returns']['type']} - {ai_docs['returns']['description']}")
        if ai_docs.get('raises'):
            sections.append("**Raises:**")
            for raise_ in ai_docs['raises']:
                sections.append(f"- **{raise_['exception']}**: {raise_['description']}")
        return "\n\n".join(sections)