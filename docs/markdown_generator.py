"""  
Markdown Documentation Generator Module  
  
Generates formatted markdown documentation from documentation sections.  
"""  
  
from datetime import datetime  
from typing import List, Optional  
from dataclasses import dataclass  
import unicodedata  
import re  
  
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
    max_heading_level: int = 6  # Prevent headings beyond h6  
  
  
class MarkdownGenerator:  
    """Generates markdown documentation with consistent formatting."""  
  
    def __init__(self, config: Optional[MarkdownConfig] = None) -> None:  
        """  
        Initialize markdown generator with optional configuration.  
  
        Args:  
            config (Optional[MarkdownConfig]): Optional markdown generation configuration.  
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
            sections (List[DocumentationSection]): List of documentation sections.  
            include_source (bool): Whether to include source code.  
            source_code (Optional[str]): Optional source code to include.  
            module_path (Optional[str]): Optional module path to include.  
  
        Returns:  
            str: Generated markdown documentation.  
        """  
        md_lines: List[str] = []  
  
        if self.config.include_timestamp or module_path:  
            md_lines.append("# Documentation\n")  
  
        # Add timestamp and module path if provided  
        if self.config.include_timestamp:  
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  
            md_lines.append(f"*Generated on: {timestamp}*")  
  
        if module_path:  
            md_lines.append(f"**Module Path:** `{module_path}`")  
  
        if (self.config.include_timestamp or module_path):  
            md_lines.append("")  # Add empty line after header info  
  
        # Generate table of contents if enabled  
        if self.config.include_toc:  
            md_lines.extend(self._generate_toc(sections))  
  
        # Generate section content  
        for section in sections:  
            md_lines.extend(self._generate_section(section))  
  
        # Add source code if included  
        if include_source and source_code:  
            md_lines.extend([  
                "## Source Code\n",  
                f"```{self.config.code_language}",  
                source_code,  
                "```",  
                ""  
            ])  
  
        return "\n".join(md_lines).strip()  
  
    def _generate_toc(  
        self,  
        sections: List[DocumentationSection],  
        level: int = 0  
    ) -> List[str]:  
        """  
        Generate table of contents.  
  
        Args:  
            sections (List[DocumentationSection]): List of documentation sections.  
            level (int): Current indentation level.  
  
        Returns:  
            List[str]: List of markdown lines for the TOC.  
        """  
        toc_lines: List[str] = []  
  
        if level == 0:  
            toc_lines.append("## Table of Contents\n")  
  
        for section in sections:  
            if not section.title:  
                continue  # Skip sections without a title  
  
            indent = "    " * level  
            link = self._create_link(section.title)  
            toc_lines.append(f"{indent}- [{section.title}](#{link})")  
  
            if section.subsections:  
                toc_lines.extend(self._generate_toc(section.subsections, level + 1))  
  
        if level == 0:  
            toc_lines.append("")  # Add empty line after TOC  
  
        return toc_lines  
  
    def _generate_section(  
        self,  
        section: DocumentationSection,  
        level: int = 2  
    ) -> List[str]:  
        """  
        Generate markdown for a documentation section.  
  
        Args:  
            section (DocumentationSection): The documentation section.  
            level (int): Current heading level.  
  
        Returns:  
            List[str]: List of markdown lines for the section.  
        """  
        md_lines: List[str] = []  
        if not section.title and not section.content:  
            return md_lines  # Return empty list if there's no content  
  
        # Calculate heading level without exceeding max_heading_level  
        header_level = min(level + self.config.heading_offset, self.config.max_heading_level)  
        header_prefix = '#' * header_level  
  
        if section.title:  
            md_lines.append(f"{header_prefix} {section.title}\n")  
  
        if section.content:  
            md_lines.append(f"{section.content}\n")  
  
        # Add subsections recursively  
        for subsection in section.subsections or []:  
            md_lines.extend(self._generate_section(subsection, level + 1))  
  
        return md_lines  
  
    @staticmethod  
    def _create_link(title: str) -> str:  
        """  
        Create markdown link from title compatible with GitHub Flavored Markdown.  
  
        Args:  
            title (str): Section title.  
  
        Returns:  
            str: Link anchor compatible with markdown.  
        """  
        # Normalize the title  
        normalized_title = unicodedata.normalize('NFKD', title)  
        # Convert to lowercase  
        normalized_title = normalized_title.lower()  
        # Remove invalid characters  
        normalized_title = re.sub(r'[^\w\- ]+', '', normalized_title)  
        # Replace spaces with dashes  
        normalized_title = normalized_title.replace(' ', '-')  
        # Remove multiple consecutive dashes  
        normalized_title = re.sub(r'-{2,}', '-', normalized_title)  
        # Strip leading and trailing dashes  
        normalized_title = normalized_title.strip('-')  
        return normalized_title  