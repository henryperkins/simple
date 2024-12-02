"""
Markdown Documentation Generator Module.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from core.logger import LoggerSetup
from core.code_extraction import ExtractedClass, ExtractedFunction, ExtractedArgument

@dataclass
class MarkdownConfig:
    """Configuration for markdown generation."""
    title_prefix: str = "Module: "
    include_metrics: bool = True
    include_toc: bool = True
    include_timestamp: bool = True
    code_language: str = "python"
    heading_offset: int = 0
    max_heading_level: int = 6
    include_source: bool = True
    include_constants: bool = True
    table_alignment: str = "left"

class MarkdownGenerator:
    """Generates formatted markdown documentation."""

    def __init__(self, config: Optional[MarkdownConfig] = None):
        self.logger = LoggerSetup.get_logger(__name__)
        self.config = config or MarkdownConfig()

    def generate(self, context: Dict[str, Any]) -> str:
        """
        Generate complete markdown documentation.
        
        Args:
            context (Dict[str, Any]): The documentation context.
            
        Returns:
            str: The generated markdown documentation.
        """
        sections = []
        
        # Generate each section with error handling
        section_generators = [
            ('header', self._generate_header),
            ('overview', self._generate_overview),
            ('classes', self._generate_classes_section),
            ('functions', self._generate_functions_section),
            ('constants', self._generate_constants_table),
            ('source', self._generate_source_section)
        ]
        
        for section_name, generator in section_generators:
            try:
                section_content = generator(context)
                if section_content:
                    sections.append(section_content)
            except Exception as e:
                self.logger.warning(f"Error generating {section_name} section: {e}")
                continue
        
        # Combine all successful sections
        content = "\n\n".join(sections)
        
        # Add table of contents if enabled
        if self.config.include_toc:
            try:
                toc = self._generate_toc(content)
                if toc:
                    content = f"{toc}\n\n{content}"
            except Exception as e:
                self.logger.warning(f"Error generating table of contents: {e}")
        
        return content

    def _generate_classes_section(self, context: Dict[str, Any]) -> str:
        """Generate classes section with tables."""
        classes = context.get('classes', [])
        if not classes:
            return ""

        sections = [
            "## Classes",
            "",
            "| Class | Inherits From | Complexity Score* |",
            "|-------|---------------|------------------|",
            *[self._format_class_row(cls) for cls in classes],
            "",
            "### Class Methods",
            "",
            "| Class | Method | Parameters | Returns | Complexity Score* |",
            "|-------|--------|------------|---------|------------------|",
            *[self._format_method_row(cls, method) 
              for cls in classes 
              for method in cls.methods]
        ]

        return "\n".join(sections)

    def _format_class_row(self, cls: ExtractedClass) -> str:
        """Format a class table row."""
        complexity = cls.metrics.get('complexity', 0)
        warning = " ⚠️" if complexity > 10 else ""
        bases = ", ".join(f"`{b}`" for b in cls.bases) or "None"

        return f"| `{cls.name}` | {bases} | {complexity}{warning} |"

    def _format_method_row(self, cls: ExtractedClass, method: ExtractedFunction) -> str:
        """Format a method table row."""
        params = self._format_params(method.args)
        complexity = method.metrics.get('complexity', 0)
        warning = " ⚠️" if complexity > 10 else ""

        return (
            f"| `{cls.name}` | `{method.name}` | `{params}` | "
            f"`{method.return_type or 'None'}` | {complexity}{warning} |"
        )

    def _generate_functions_section(self, context: Dict[str, Any]) -> str:
        """Generate functions section."""
        functions = context.get('functions', [])
        if not functions:
            return ""

        sections = [
            "## Functions",
            "",
            "| Function | Parameters | Returns | Complexity Score* |",
            "|----------|------------|---------|------------------|",
            *[self._format_function_row(func) for func in functions]
        ]

        return "\n".join(sections)

    def _format_function_row(self, func: ExtractedFunction) -> str:
        """Format a function table row."""
        params = self._format_params(func.args)
        complexity = func.metrics.get('complexity', 0)
        warning = " ⚠️" if complexity > 10 else ""

        return (
            f"| `{func.name}` | `{params}` | "
            f"`{func.return_type or 'None'}` | {complexity}{warning} |"
        )

    def _generate_overview(self, context: Dict[str, Any]) -> str:
        """Generate the overview section."""
        return "\n".join([
            "## Overview",
            f"**File:** `{context['file_path']}`",
            f"**Description:** {context.get('description', 'No description available.')}",
            "",
            "### Module Statistics",
            f"- Classes: {len(context.get('classes', []))}",
            f"- Functions: {len(context.get('functions', []))}",
            f"- Constants: {len(context.get('constants', []))}"
        ])

    def _generate_classes_table(self, classes: List[ExtractedClass]) -> str:
        """Generate the classes summary table."""
        if not classes:
            return ""

        return "\n".join([
            "## Classes",
            "",
            "| Class | Inherits From | Complexity Score* |",
            "|-------|---------------|------------------|",
            *[f"| `{cls.name}` | {', '.join(f'`{b}`' for b in cls.bases) or 'None'} | "
              f"{cls.metrics.get('complexity', 0)}"
              f"{' ⚠️' if cls.metrics.get('complexity', 0) > 10 else ''} |"
              for cls in classes]
        ])

    def _generate_class_methods_table(self, cls: ExtractedClass) -> str:
        """Generate methods table for a class."""
        lines = [
            "### Class Methods",
            "",
            "| Method | Parameters | Returns | Complexity Score* |",
            "|--------|------------|---------|------------------|"
        ]

        for method in cls.methods:
            params = self._format_params(method.args)
            complexity = method.metrics.get('complexity', 0)
            warning = " ⚠️" if complexity > 10 else ""

            lines.append(
                f"| `{method.name}` | `{params}` | "
                f"`{method.return_type or 'None'}` | {complexity}{warning} |"
            )

        return "\n".join(lines)

    def _generate_functions_table(self, functions: List[ExtractedFunction]) -> str:
        """Generate the functions table."""
        if not functions:
            return ""

        return "\n".join([
            "## Functions",
            "",
            "| Function | Parameters | Returns | Complexity Score* |",
            "|----------|------------|---------|------------------|",
            *[f"| `{func.name}` | `({self._format_params(func.args)})` | "
              f"`{func.return_type or 'None'}` | "
              f"{func.metrics.get('complexity', 0)}"
              f"{' ⚠️' if func.metrics.get('complexity', 0) > 10 else ''} |"
              for func in functions]
        ])

    def _generate_constants_table(self, context: Dict[str, Any]) -> str:
        """Generate the constants table."""
        constants = context.get('constants', [])

        if not constants:
            return ""

        valid_constants = []
        for const in constants:
            if isinstance(const, dict) and all(key in const for key in ['name', 'type', 'value']):
                valid_constants.append(const)
            elif isinstance(const, str):
                valid_constants.append({
                    'name': const,
                    'type': 'str',
                    'value': const
                })

        if not valid_constants:
            return ""

        return "\n".join([
            "## Constants and Variables",
            "",
            "| Name | Type | Value |",
            "|------|------|-------|",
            *[f"| `{const['name']}` | `{const['type']}` | `{const['value']}` |"
            for const in valid_constants]
        ])

    def _format_params(self, args: List[ExtractedArgument]) -> str:
        """Format parameters with types."""
        return ", ".join(
            f"{arg.name}: {arg.type_hint}"
            f"{' = ' + arg.default_value if arg.default_value else ''}"
            for arg in args
        )

    def _generate_source_section(self, context: Dict[str, Any]) -> str:
        """Generate the source code section."""
        if not self.config.include_source or 'source_code' not in context:
            return ""

        return "\n".join([
            "## Source Code",
            "",
            f"```{self.config.code_language}",
            context['source_code'],
            "```"
        ])

    def _generate_header(self, context: Dict[str, Any]) -> str:
        """Generate the document header."""
        module_name = context.get('module_name', 'Unknown Module')
        file_path = context.get('file_path', '')

        return "\n".join([
            f"# {self.config.title_prefix}{module_name}",
            "",
            f"**File:** `{file_path}`"
        ])

    def _generate_toc(self, content: str) -> str:
        """Generate table of contents from content headers."""
        lines = []
        for line in content.split('\n'):
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                if level <= self.config.max_heading_level:
                    text = line.lstrip('#').strip()
                    link = text.lower().replace(' ', '-')
                    indent = '  ' * (level - 1)
                    lines.append(f"{indent}- [{text}](#{link})")

        if lines:
            return "## Table of Contents\n" + "\n".join(lines)
        return ""

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics section with warnings."""
        lines = []

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                warning = " ⚠️" if value > 10 else ""
                lines.append(f"- {name}: {value}{warning}")
            elif isinstance(value, dict):
                lines.append(f"- {name}: {value}")

        return "\n".join(lines)