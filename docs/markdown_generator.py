# markdown_generator.py
"""
Markdown Documentation Generator Module

Generates formatted markdown documentation from documentation sections.
"""

from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass

from core.logger import LoggerSetup
from core.docstring_processor import DocumentationSection

logger = LoggerSetup.get_logger(__name__)

@dataclass
class MarkdownConfig:
    """Configuration for markdown generation."""
    include_toc: bool = True
    include_timestamp: bool = True
    code_language: str = "python"
    heading_offset: int = 0

class MarkdownGenerator:
    """Generates markdown documentation with consistent formatting."""

    def __init__(self, config: Optional[MarkdownConfig] = None):
        """
        Initialize markdown generator with optional configuration.

        Args:
            config: Optional markdown generation configuration
        """
        self.config = config or MarkdownConfig()

    def generate(
        self,
        sections: List[DocumentationSection],
        include_source: bool = True,
        source_code: Optional[str] = None,
        module_path: Optional[str] = None
    ) -> str:
        """
        Generate complete markdown documentation.

        Args:
            sections: List of documentation sections
            include_source: Whether to include source code
            source_code: Optional source code to include
            module_path: Optional module path to include

        Returns:
            str: Generated markdown documentation
        """
        md_lines = []
        
        # Add header
        if self.config.include_timestamp:
            md_lines.extend([
                "# Documentation",
                "",
                f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
                ""
            ])

        # Add module path if provided
        if module_path:
            md_lines.extend([
                f"**Module Path:** `{module_path}`",
                ""
            ])

        # Generate table of contents if enabled
        if self.config.include_toc:
            md_lines.extend(self._generate_toc(sections))

        # Generate section content
        for section in sections:
            md_lines.extend(self._generate_section(section))

        # Add source code if included
        if include_source and source_code:
            md_lines.extend([
                "## Source Code",
                "",
                f"```{self.config.code_language}",
                source_code,
                "```",
                ""
            ])

        return "\n".join(md_lines)

    def _generate_toc(
        self,
        sections: List[DocumentationSection],
        level: int = 0
    ) -> List[str]:
        """Generate table of contents."""
        toc_lines = []
        
        if level == 0:
            toc_lines.extend([
                "## Table of Contents",
                ""
            ])

        for section in sections:
            indent = "    " * level
            link = self._create_link(section.title)
            toc_lines.append(f"{indent}- [{section.title}](#{link})")
            
            if section.subsections:
                toc_lines.extend(self._generate_toc(section.subsections, level + 1))

        if level == 0:
            toc_lines.append("")

        return toc_lines

    def _generate_section(
        self,
        section: DocumentationSection,
        level: int = 2
    ) -> List[str]:
        """Generate markdown for a documentation section."""
        md_lines = []
        
        # Add section header
        header_level = min(level + self.config.heading_offset, 6)
        md_lines.extend([
            f"{'#' * header_level} {section.title}",
            ""
        ])

        # Add section content
        if section.content:
            md_lines.extend([
                section.content,
                ""
            ])

        # Add subsections
        if section.subsections:
            for subsection in section.subsections:
                if subsection:  # Skip None subsections
                    md_lines.extend(self._generate_section(subsection, level + 1))

        return md_lines

    @staticmethod
    def _create_link(title: str) -> str:
        """Create markdown link from title."""
        return title.lower().replace(' ', '-').replace(':', '').replace('_', '-')