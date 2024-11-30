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
    def __init__(  
        self,  
        include_toc: bool = True,  
        include_timestamp: bool = True,  
        code_language: str = "python",  
        heading_offset: int = 0,  
        max_heading_level: int = 6,  
        include_source: bool = True  
    ):  
        self.include_toc = include_toc  
        self.include_timestamp = include_timestamp  
        self.code_language = code_language  
        self.heading_offset = heading_offset  
        self.max_heading_level = max_heading_level  
        self.include_source = include_source  
  
class MarkdownGenerator:  
    """Generates markdown documentation with consistent formatting."""  
  
    def __init__(self, config: Optional[MarkdownConfig] = None):  
        """Initialize the markdown generator with optional configuration."""  
        self.logger = LoggerSetup.get_logger(__name__)  # Initialize logger  
        self.config = config or MarkdownConfig()  
  
    def generate(self, sections: List[DocumentationSection], module_path: Optional[Path] = None) -> str:  
        """Generate complete markdown documentation following the template structure."""  
        self.logger.debug("Starting markdown generation.")  
        module_name = module_path.stem if module_path else "Unknown Module"  
  
        content_sections = [  
            self._generate_header(module_name),  
            self._generate_overview(module_path, sections),  
            self._generate_classes_section(sections),  
            self._generate_class_methods_section(sections),  
            self._generate_functions_section(sections),  
            self._generate_constants_section(sections),  
            self._generate_changes_section(sections),  
            self._generate_source_section(sections)  
        ]  
  
        # Filter out empty sections  
        final_content = "\n".join(filter(None, content_sections))  
        self.logger.debug("Markdown generation completed.")  
        return final_content  
  
    def _generate_header(self, module_name: str) -> str:  
        """Generate the module header."""  
        self.logger.debug(f"Generating header for module: {module_name}")  
        header = [f"# Module: {module_name}\n"]  
        if self.config.include_timestamp:  
            header.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")  
        return "\n".join(header)  
  
    def _generate_overview(self, module_path: Optional[Path], sections: List[DocumentationSection]) -> str:  
        """Generate the overview section."""  
        self.logger.debug("Generating overview section.")  
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
        self.logger.debug("Generating classes section.")  
        classes_section = next((s for s in sections if s.title == "Classes"), None)  
        if not classes_section or not classes_section.content.strip():  
            return ""  
  
        return classes_section.content  
  
    def _generate_class_methods_section(self, sections: List[DocumentationSection]) -> str:  
        """Generate the class methods section."""  
        self.logger.debug("Generating class methods section.")  
        methods_section = next((s for s in sections if s.title == "Class Methods"), None)  
        if not methods_section or not methods_section.content.strip():  
            return ""  
  
        return methods_section.content  
  
    def _generate_functions_section(self, sections: List[DocumentationSection]) -> str:  
        """Generate the functions section."""  
        self.logger.debug("Generating functions section.")  
        functions_section = next((s for s in sections if s.title == "Functions"), None)  
        if not functions_section or not functions_section.content.strip():  
            return ""  
  
        return functions_section.content  
  
    def _generate_constants_section(self, sections: List[DocumentationSection]) -> str:  
        """Generate the constants and variables section."""  
        self.logger.debug("Generating constants section.")  
        constants_section = next((s for s in sections if s.title == "Constants and Variables"), None)  
        if not constants_section or not constants_section.content.strip():  
            return ""  
  
        return constants_section.content  
  
    def _generate_changes_section(self, sections: List[DocumentationSection]) -> str:  
        """Generate the recent changes section."""  
        self.logger.debug("Generating changes section.")  
        changes_section = next((s for s in sections if s.title == "Recent Changes"), None)  
        if changes_section and changes_section.content.strip():  
            return changes_section.content  
        else:  
            return "\n".join([  
                "## Recent Changes",  
                "- No recent changes recorded.",  
                ""  
            ])  
  
    def _generate_source_section(self, sections: List[DocumentationSection]) -> str:  
        """Generate the source code section."""  
        self.logger.debug("Generating source code section.")  
        if not self.config.include_source:  
            return ""  
  
        source_section = next((s for s in sections if s.title == "Source Code"), None)  
        if not source_section or not source_section.content.strip():  
            return ""  
  
        return source_section.content  