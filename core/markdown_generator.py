"""
Markdown Documentation Generator Module.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from core.logger import LoggerSetup
from core.code_extraction import ExtractedClass, ExtractedFunction

@dataclass
class MarkdownConfig:
    """Configuration for markdown generation."""
    include_toc: bool = True
    include_timestamp: bool = True
    code_language: str = "python"
    heading_offset: int = 0
    max_heading_level: int = 6
    include_source: bool = True

class MarkdownGenerator:
    """Generates formatted markdown documentation."""

    def __init__(self, config: Optional[MarkdownConfig] = None):
        """Initialize the markdown generator."""
        self.logger = LoggerSetup.get_logger(__name__)
        self.config = config or MarkdownConfig()

    def generate(self, context: Dict[str, Any]) -> str:
        """Generate complete markdown documentation."""
        sections = [
            self._generate_header(context),
            self._generate_overview(context),
            self._generate_classes(context.get('classes', [])),
            self._generate_functions(context.get('functions', [])),
            self._generate_source_code(context)
        ]
        
        content = "\n\n".join(filter(None, sections))
        
        if self.config.include_toc:
            content = self._generate_toc(content) + "\n\n" + content
            
        return content

    def _generate_header(self, context: Dict[str, Any]) -> str:
        """Generate the document header."""
        header = [
            "# " + context['module_name'],
            "",
            f"**File Path:** `{context['file_path']}`"
        ]
        
        if self.config.include_timestamp:
            header.extend([
                "",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ])
            
        return "\n".join(header)

    def _generate_overview(self, context: Dict[str, Any]) -> str:
        """Generate the overview section."""
        description = context.get('description', 'No description available.')
        metrics = context.get('metrics', {})
        
        return "\n".join([
            "## Overview",
            "",
            description,
            "",
            "### Module Statistics",
            f"- Classes: {len(context.get('classes', []))}",
            f"- Functions: {len(context.get('functions', []))}",
            f"- Constants: {len(context.get('constants', []))}",
            "",
            "### Complexity Metrics",
            *[f"- {key}: {value}" + (" ⚠️" if key == 'complexity' and value > 10 else "") 
              for key, value in metrics.items()]
        ])

    def _generate_classes(self, classes: List[ExtractedClass]) -> str:
        """Generate the classes section."""
        if not classes:
            return ""

        lines = [
            "## Classes",
            "",
            "| Class | Inherits From | Complexity | Methods |",
            "|-------|---------------|------------|----------|"
        ]

        for cls in classes:
            complexity = cls.metrics.get('complexity', 0)
            warning = " ⚠️" if complexity > 10 else ""
            lines.append(
                f"| `{cls.name}` | {', '.join(cls.bases) or 'None'} | "
                f"{complexity}{warning} | {len(cls.methods)} |"
            )

            if cls.docstring:
                lines.extend(["", f"### {cls.name}", "", cls.docstring])

            if cls.methods:
                lines.extend(self._generate_methods(cls.name, cls.methods))

        return "\n".join(lines)

    def _generate_methods(self, class_name: str, methods: List[ExtractedFunction]) -> List[str]:
        """Generate the methods section."""
        lines = [
            "",
            f"### {class_name} Methods",
            "",
            "| Method | Parameters | Returns | Complexity |",
            "|--------|------------|---------|------------|"
        ]

        for method in methods:
            complexity = method.metrics.get('complexity', 0)
            warning = " ⚠️" if complexity > 10 else ""
            
            params = ", ".join(
                f"{arg.name}: {arg.type_hint}" for arg in method.args
            )
            
            lines.append(
                f"| `{method.name}` | `{params}` | "
                f"`{method.return_type or 'None'}` | {complexity}{warning} |"
            )

            if method.docstring:
                lines.extend(["", method.docstring, ""])

        return lines

    def _generate_functions(self, functions: List[ExtractedFunction]) -> str:
        """Generate the functions section."""
        if not functions:
            return ""

        lines = [
            "## Functions",
            "",
            "| Function | Parameters | Returns | Complexity |",
            "|----------|------------|---------|------------|"
        ]

        for func in functions:
            complexity = func.metrics.get('complexity', 0)
            warning = " ⚠️" if complexity > 10 else ""
            
            params = ", ".join(
                f"{arg.name}: {arg.type_hint}" for arg in func.args
            )
            
            lines.append(
                f"| `{func.name}` | `{params}` | "
                f"`{func.return_type or 'None'}` | {complexity}{warning} |"
            )

            if func.docstring:
                lines.extend(["", func.docstring])

        return "\n".join(lines)

    def _generate_source_code(self, context: Dict[str, Any]) -> str:
        """Generate the source code section."""
        if not (context.get('source_code') and self.config.include_source):
            return ""

        metrics = context.get('metrics', {})
        
        return "\n".join([
            "## Source Code",
            "",
            "### Code Metrics",
            *[f"- {key}: {value}" + (" ⚠️" if key in ['complexity', 'total_lines'] and value > 10 else "")
              for key, value in metrics.items()],
            "",
            "```" + self.config.code_language,
            context['source_code'],
            "```"
        ])

    def _generate_toc(self, content: str) -> str:
        """Generate table of contents."""
        lines = ["## Table of Contents"]
        current_level = 0

        for line in content.split('\n'):
            if line.startswith('#'):
                # Count heading level
                level = len(line.split()[0]) - 1
                if level > 1:  # Skip title
                    title = line.lstrip('#').strip()
                    # Create anchor link
                    anchor = title.lower().replace(' ', '-')
                    # Add appropriate indentation
                    indent = '  ' * (level - 2)
                    lines.append(f"{indent}- [{title}](#{anchor})")

        return "\n".join(lines)