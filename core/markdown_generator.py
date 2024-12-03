"""
Markdown Documentation Generator Module.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from core.logger import LoggerSetup
from core.extraction.function_extractor import ExtractedFunction
from core.extraction.class_extractor import ExtractedClass
from core.docstring_processor import DocstringProcessor

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

    def __init__(self, config: Optional[MarkdownConfig] = None,
                 docstring_processor: Optional[DocstringProcessor] = None):
        """Initialize the markdown generator."""
        self.logger = LoggerSetup.get_logger(__name__)
        self.config = config or MarkdownConfig()
        self.docstring_processor = docstring_processor or DocstringProcessor()

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
            toc = self._generate_toc(content)
            content = f"{toc}\n\n{content}"
            
        return content

    def _generate_methods(self, class_name: str, methods: List[Any]) -> List[str]:
        """Generate the methods section."""
        if not methods:
            return []

        lines = [
            "",
            f"### {class_name} Methods",
            "",
            "| Method | Parameters | Returns | Complexity |",
            "|--------|------------|---------|------------|"
        ]

        for method in methods:
            if isinstance(method, dict):
                name = method.get('name', '')
                params = []
                for arg in method.get('args', []):
                    arg_name = arg.get('name', '')
                    arg_type = arg.get('type_hint', 'Any')
                    params.append(f"{arg_name}: {arg_type}")
                
                return_type = method.get('return_type', 'None')
                docstring = method.get('docstring', '')
                metrics = method.get('metrics', {})
            else:
                name = getattr(method, 'name', '')
                params = []
                for arg in getattr(method, 'args', []):
                    arg_name = getattr(arg, 'name', '')
                    arg_type = getattr(arg, 'type_hint', 'Any')
                    params.append(f"{arg_name}: {arg_type}")
                
                return_type = getattr(method, 'return_type', 'None')
                docstring = getattr(method, 'docstring', '')
                metrics = getattr(method, 'metrics', {})

            complexity = metrics.get('complexity', 0) if isinstance(metrics, dict) else 0
            warning = " ⚠️" if complexity > 10 else ""

            lines.append(
                f"| `{name}` | {', '.join(params) or 'None'} | "
                f"`{return_type or 'None'}` | {complexity}{warning} |"
            )

            if docstring:
                # Use docstring processor to parse and format
                parsed_docstring = self.docstring_processor.parse(docstring)
                formatted_docstring = self.docstring_processor.format(parsed_docstring)
                if formatted_docstring:
                    lines.extend(["", formatted_docstring, ""])

        return lines

    def _generate_classes(self, classes: List[Any]) -> str:
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
            if isinstance(cls, dict):
                name = cls.get('name', '')
                bases = cls.get('bases', [])
                methods = cls.get('methods', [])
                docstring = cls.get('docstring', '')
                metrics = cls.get('metrics', {})
            else:
                name = getattr(cls, 'name', '')
                bases = getattr(cls, 'bases', [])
                methods = getattr(cls, 'methods', [])
                docstring = getattr(cls, 'docstring', '')
                metrics = getattr(cls, 'metrics', {})

            complexity = metrics.get('complexity', 0) if isinstance(metrics, dict) else 0
            warning = " ⚠️" if complexity > 10 else ""
            
            lines.append(
                f"| `{name}` | {', '.join(bases) or 'None'} | "
                f"{complexity}{warning} | {len(methods)} |"
            )

            if docstring:
                # Use docstring processor to parse and format
                parsed_docstring = self.docstring_processor.parse(docstring)
                formatted_docstring = self.docstring_processor.format(parsed_docstring)
                if formatted_docstring:
                    lines.extend(["", f"### {name}", "", formatted_docstring])

            if methods:
                lines.extend(self._generate_methods(name, methods))

        return "\n".join(lines)

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
            # Handle both object and dict cases
            if isinstance(func, dict):
                metrics = func.get('metrics', {})
                name = func.get('name', '')
                args = func.get('args', [])
                return_type = func.get('return_type')
                docstring = func.get('docstring', '')
            else:
                metrics = getattr(func, 'metrics', {})
                name = getattr(func, 'name', '')
                args = getattr(func, 'args', [])
                return_type = getattr(func, 'return_type', None)
                docstring = getattr(func, 'docstring', '')

            complexity = metrics.get('complexity', 0) if isinstance(metrics, dict) else 0
            warning = " ⚠️" if complexity > 10 else ""

            # Handle args for both dict and object cases
            param_list = []
            for arg in args:
                if isinstance(arg, dict):
                    arg_name = arg.get('name', '')
                    arg_type = arg.get('type_hint', 'Any')
                else:
                    arg_name = getattr(arg, 'name', '')
                    arg_type = getattr(arg, 'type_hint', 'Any')
                param_list.append(f"{arg_name}: {arg_type}")
                
            params = ", ".join(param_list)
            
            lines.append(
                f"| `{name}` | `{params}` | "
                f"`{return_type or 'None'}` | {complexity}{warning} |"
            )

            if docstring:
                lines.extend(["", docstring])

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