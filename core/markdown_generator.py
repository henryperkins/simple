"""
Markdown Documentation Generator Module

Generates formatted markdown documentation from documentation sections.
"""

from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
import unicodedata
import re
from pathlib import Path

from core.logger import LoggerSetup
from core.docstring_processor import DocumentationSection

logger = LoggerSetup.get_logger(__name__)

@dataclass
class MarkdownConfig:
    """Configuration for markdown generation.

    Attributes:
        include_toc (bool): Whether to include a table of contents.
        include_timestamp (bool): Whether to include a timestamp in the documentation.
        code_language (str): The programming language for syntax highlighting in code blocks.
        heading_offset (int): Offset for heading levels in the documentation.
        max_heading_level (int): Maximum heading level allowed (default is 6).
        include_source (bool): Option to include source code snippets in the documentation.
    """
    include_toc: bool = True
    include_timestamp: bool = True
    code_language: str = "python"
    heading_offset: int = 0
    max_heading_level: int = 6
    include_source: bool = True

class MarkdownGenerator:
    """Generates markdown documentation with consistent formatting."""

    def __init__(self, config: Optional[MarkdownConfig] = None):
        """Initialize the markdown generator with optional configuration."""
        self.config = config or MarkdownConfig()

    def generate(self, sections: List[DocumentationSection], module_path: Optional[Path] = None) -> str:
        """Generate complete markdown documentation following the template structure."""
        module_name = module_path.stem if module_path else "Unknown Module"
        
        sections = [
            self._generate_header(module_name),
            self._generate_overview(module_path, sections),
            self._generate_classes_section(sections),
            self._generate_class_methods_section(sections),
            self._generate_functions_section(sections),
            self._generate_constants_section(sections),
            self._generate_changes_section(sections),
            self._generate_source_section(sections)
        ]
        
        return "\n".join(filter(None, sections))

    def _generate_header(self, module_name: str) -> str:
        """Generate the module header."""
        return f"# Module: {module_name}\n"

    def _generate_overview(self, module_path: Optional[Path], sections: List[DocumentationSection]) -> str:
        """Generate the overview section."""
        overview_section = next((s for s in sections if s.title == "Overview"), None)
        description = overview_section.content if overview_section else "No description available."
        
        return "\n".join([
            "## Overview",
            f"**File:** `{str(module_path) if module_path else 'Unknown'}`",
            f"**Description:** {description}",
            ""
        ])

    def _generate_classes_section(self, sections: List[DocumentationSection]) -> str:
        """Generate the classes section."""
        classes_section = next((s for s in sections if s.title == "Classes"), None)
        if not classes_section:
            return ""

        lines = [
            "## Classes",
            "| Class | Inherits From | Complexity Score* |",
            "|-------|---------------|-------------------|"
        ]

        if hasattr(classes_section, 'tables'):
            lines.extend(classes_section.tables)

        return "\n".join(lines) + "\n"

    def _generate_class_methods_section(self, sections: List[DocumentationSection]) -> str:
        """Generate the class methods section."""
        methods_section = next((s for s in sections if s.title == "Class Methods"), None)
        if not methods_section:
            return ""

        lines = [
            "### Class Methods",
            "| Class | Method | Parameters | Returns | Complexity Score* |",
            "|-------|--------|------------|---------|-------------------|"
        ]

        if hasattr(methods_section, 'tables'):
            lines.extend(methods_section.tables)

        return "\n".join(lines) + "\n"

    def _generate_functions_section(self, sections: List[DocumentationSection]) -> str:
        """Generate the functions section."""
        functions_section = next((s for s in sections if s.title == "Functions"), None)
        if not functions_section:
            return ""

        lines = [
            "## Functions",
            "| Function | Parameters | Returns | Complexity Score* |",
            "|----------|------------|---------|-------------------|"
        ]

        if hasattr(functions_section, 'tables'):
            lines.extend(functions_section.tables)

        return "\n".join(lines) + "\n"

    def _generate_constants_section(self, sections: List[DocumentationSection]) -> str:
        """Generate the constants and variables section."""
        constants_section = next((s for s in sections if s.title == "Constants and Variables"), None)
        if not constants_section:
            return ""

        lines = [
            "## Constants and Variables",
            "| Name | Type | Value |",
            "|------|------|--------|"
        ]

        if hasattr(constants_section, 'tables'):
            lines.extend(constants_section.tables)

        return "\n".join(lines) + "\n"

    def _generate_changes_section(self, sections: List[DocumentationSection]) -> str:
        """Generate the recent changes section."""
        changes_section = next((s for s in sections if s.title == "Recent Changes"), None)
        content = []
        
        if changes_section and changes_section.content:
            content = [
                "## Recent Changes",
                changes_section.content,
                ""
            ]
        else:
            content = [
                "## Recent Changes",
                "- No recent changes recorded.",
                ""
            ]

        return "\n".join(content)

    def _generate_source_section(self, sections: List[DocumentationSection]) -> str:
        """Generate the source code section."""
        source_section = next((s for s in sections if s.title == "Source Code"), None)
        if not source_section or not source_section.source_code:
            return ""

        return "\n".join([
            "## Source Code",
            "```python",
            source_section.source_code,
            "```"
        ])
