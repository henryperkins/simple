"""  
Markdown Documentation Generator Module  
  
Generates formatted markdown documentation from documentation sections.  
"""  
  
from datetime import datetime  
from typing import List, Optional, Dict, Any  
from dataclasses import dataclass  
import unicodedata  
import re  
import ast  
  
from core.logger import LoggerSetup  
from core.docstring_processor import DocumentationSection, DocstringProcessor, DocstringData  
  
logger = LoggerSetup.get_logger(__name__)  
  
  
@dataclass  
class MarkdownConfig:  
    """Configuration for markdown generation."""  
    include_toc: bool = True  
    include_timestamp: bool = True  
    code_language: str = "python"  
    heading_offset: int = 0  
    max_heading_level: int = 6  # Prevent headings beyond h6  
    include_source: bool = False  # Add this line to include source code snippets
  
  
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
  
    def generate_markdown_from_docstrings(self, source_code: str, metadata: Dict[str, Any], node: Optional[ast.AST] = None) -> str:  
        """  
        Generate markdown documentation from docstrings and AST information.  
  
        Args:  
            source_code (str): The source code to document.  
            metadata (Dict[str, Any]): Additional metadata for documentation.  
            node (Optional[ast.AST]): Optional AST node for context.  
  
        Returns:  
            str: Generated markdown documentation.  
        """  
        processor = DocstringProcessor()  
        tree = ast.parse(source_code)  
        sections = []  
  
        for child in ast.iter_child_nodes(tree):  
            if isinstance(child, (ast.FunctionDef, ast.ClassDef)):  
                docstring_data = processor.process_node(child, source_code)  
                section = DocumentationSection(  
                    title=child.name,  
                    content=processor.format(docstring_data)  
                )  
                sections.append(section)  
  
        return self.generate(sections, include_source=self.config.include_source, source_code=source_code)  
  
    def generate_markdown_documentation(self, module_name: str, module_path: str, description: str, classes: List[Dict[str, Any]], functions: List[Dict[str, Any]], constants: List[Dict[str, Any]], recent_changes: List[str], source_code: str) -> str:
        """
        Generate markdown documentation using the specified template.

        Args:
            module_name (str): The name of the module.
            module_path (str): The path to the module file.
            description (str): Brief description of the module.
            classes (List[Dict[str, Any]]): List of class information.
            functions (List[Dict[str, Any]]): List of function information.
            constants (List[Dict[str, Any]]): List of constant and variable information.
            recent_changes (List[str]): List of recent changes.
            source_code (str): The source code of the module.

        Returns:
            str: Generated markdown documentation.
        """
        md_lines: List[str] = []

        # Module header
        md_lines.append(f"# Module: {module_name}\n")
        md_lines.append("## Overview")
        md_lines.append(f"**File:** `{module_path}`")
        md_lines.append(f"**Description:** {description}\n")

        # Classes section
        md_lines.append("## Classes\n")
        md_lines.append("| Class | Inherits From | Complexity Score* |")
        md_lines.append("|-------|---------------|------------------|")
        for cls in classes:
            md_lines.append(f"| `{cls['name']}` | `{cls['inherits_from']}` | {cls['complexity_score']} |")

        # Class methods section
        md_lines.append("\n### Class Methods\n")
        md_lines.append("| Class | Method | Parameters | Returns | Complexity Score* |")
        md_lines.append("|-------|--------|------------|---------|------------------|")
        for cls in classes:
            for method in cls['methods']:
                md_lines.append(f"| `{cls['name']}` | `{method['name']}` | `{method['parameters']}` | `{method['returns']}` | {method['complexity_score']} |")

        # Functions section
        md_lines.append("\n## Functions\n")
        md_lines.append("| Function | Parameters | Returns | Complexity Score* |")
        md_lines.append("|----------|------------|---------|------------------|")
        for func in functions:
            md_lines.append(f"| `{func['name']}` | `{func['parameters']}` | `{func['returns']}` | {func['complexity_score']} |")

        # Constants and Variables section
        md_lines.append("\n## Constants and Variables\n")
        md_lines.append("| Name | Type | Value |")
        md_lines.append("|------|------|-------|")
        for const in constants:
            md_lines.append(f"| `{const['name']}` | `{const['type']}` | `{const['value']}` |")

        # Recent Changes section
        md_lines.append("\n## Recent Changes")
        for change in recent_changes:
            md_lines.append(f"- {change}")

        # Source Code section
        md_lines.append("\n## Source Code")
        md_lines.append(f"```{self.config.code_language}")
        md_lines.append(source_code)
        md_lines.append("```")

        return "\n".join(md_lines).strip()
