"""
Documentation Management Module

Handles docstring operations and documentation generation with improved structure
and centralized processing.
"""

class DocumentationError(Exception):
    """Custom exception for documentation generation errors."""
    def __init__(self, message: str, details: dict):
        self.details = details
        super().__init__(message)

import ast
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field
from core.logger import LoggerSetup
from core.docstring_processor import DocstringProcessor, DocumentationSection
from .markdown_generator import MarkdownGenerator, MarkdownConfig
from core.code_extraction import CodeExtractor, ExtractionContext, ExtractedClass, ExtractedFunction, ExtractedArgument
from core.metrics import Metrics

logger = LoggerSetup.get_logger(__name__)

@dataclass
class DocumentationContext:
    """
    Holds context for documentation generation.

    Attributes:
        source_code (str): The source code to be documented.
        module_path (Optional[Path]): The path to the module file.
        include_source (bool): Flag to include source code in the documentation.
        metadata (Dict[str, Any]): Additional metadata for documentation.
    """
    source_code: str
    module_path: Optional[Path] = None
    include_source: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocStringManager:
    """
    Manages docstring operations and documentation generation.

    Attributes:
        context (DocumentationContext): The context for documentation generation.
        cache (Optional[Any]): Optional cache for storing intermediate results.
    """

    def __init__(self, context: DocumentationContext, cache: Optional[Any] = None) -> None:
        """
        Initialize DocStringManager with context and optional cache.

        Args:
            context (DocumentationContext): The context for documentation generation.
            cache (Optional[Any]): Optional cache for storing intermediate results.
        """
        self.context = context
        self.tree: ast.Module = ast.parse(context.source_code)
        self.processor = DocstringProcessor()
        self.cache = cache
        self.changes: List[str] = []
        self.markdown_generator = MarkdownGenerator(MarkdownConfig(include_source=True))
        self.code_extractor = CodeExtractor(ExtractionContext())
        self.metrics_calculator = Metrics()
        self._add_parents(self.tree)

    async def generate_documentation(self) -> str:
        """Generate complete documentation."""
        try:
            extraction_result = self.code_extractor.extract_code(self.context.source_code)
            
            # Get AI-generated docs from metadata
            ai_docs = self.context.metadata.get('ai_generated', '')
            
            sections = [
                self._create_module_section(ai_docs),  # Pass AI docs
                self._create_overview_section(),
                self._create_classes_section(extraction_result.classes),
                self._create_class_methods_section(extraction_result.classes),
                self._create_functions_section(extraction_result.functions),
                self._create_constants_section(extraction_result.constants),
                self._create_source_code_section(extraction_result.metrics)
            ]
            
            return self.markdown_generator.generate(sections, self.context.module_path)
            
        except Exception as e:
            logger.error("Failed to generate documentation: %s", str(e))
            raise DocumentationError("Documentation generation failed", {'error': str(e)})

    def _create_module_section(self, ai_docs: str) -> DocumentationSection:
        """Create module section with AI-generated documentation."""
        return DocumentationSection(
            title="Module Documentation",
            content=ai_docs or "No documentation available."
        )
    def _create_overview_section(self) -> DocumentationSection:
        """
        Create overview section.

        Returns:
            DocumentationSection: The overview section of the documentation.
        """
        file_path = self.context.module_path or Path('unknown')
        description = self.context.metadata.get('description', 'No description provided.')
        content = [
            "## Overview",
            f"**File:** `{file_path}`",
            f"**Description:** {description}"
        ]
        return DocumentationSection(title="Overview", content="\n".join(content))

    def _create_classes_section(self, classes: List[ExtractedClass]) -> DocumentationSection:
        """
        Create classes section.

        Args:
            classes (List[ExtractedClass]): List of extracted classes.

        Returns:
            DocumentationSection: The classes section of the documentation.
        """
        if not classes:
            return DocumentationSection("Classes", "")
        content = [
            "## Classes",
            "| Class | Inherits From | Complexity Score* |",
            "|-------|---------------|------------------|"
        ]
        for cls in classes:
            complexity_score = self.metrics_calculator.calculate_complexity(cls.node)
            warning = " ⚠️" if complexity_score > 10 else ""
            bases = ", ".join(cls.bases) if cls.bases else "None"
            row = f"| `{cls.name}` | `{bases}` | {complexity_score}{warning} |"
            content.append(row)
        return DocumentationSection(title="Classes", content="\n".join(content))

    def _create_class_methods_section(self, classes: List[ExtractedClass]) -> DocumentationSection:
        """
        Create class methods section.

        Args:
            classes (List[ExtractedClass]): List of extracted classes.

        Returns:
            DocumentationSection: The class methods section of the documentation.
        """
        if not classes:
            return DocumentationSection("Class Methods", "")
        content = [
            "### Class Methods",
            "| Class | Method | Parameters | Returns | Complexity Score* |",
            "|-------|--------|------------|---------|-------------------|"
        ]
        for cls in classes:
            for method in cls.methods:
                complexity_score = self.metrics_calculator.calculate_complexity(method.node)
                warning = " ⚠️" if complexity_score > 10 else ""
                params = self._format_parameters(method.args)
                returns = method.return_type or "None"
                row = f"| `{cls.name}` | `{method.name}` | `{params}` | `{returns}` | {complexity_score}{warning} |"
                content.append(row)
        return DocumentationSection(title="Class Methods", content="\n".join(content))

    def _create_functions_section(self, functions: List[ExtractedFunction]) -> DocumentationSection:
        """
        Create functions section.

        Args:
            functions (List[ExtractedFunction]): List of extracted functions.

        Returns:
            DocumentationSection: The functions section of the documentation.
        """
        if not functions:
            return DocumentationSection("Functions", "")
        content = [
            "## Functions",
            "| Function | Parameters | Returns | Complexity Score* |",
            "|----------|------------|---------|-------------------|"
        ]
        for func in functions:
            complexity_score = self.metrics_calculator.calculate_complexity(func.node)
            warning = " ⚠️" if complexity_score > 10 else ""
            params = self._format_parameters(func.args)
            returns = func.return_type or "None"
            row = f"| `{func.name}` | `{params}` | `{returns}` | {complexity_score}{warning} |"
            content.append(row)
        return DocumentationSection(title="Functions", content="\n".join(content))

    def _create_constants_section(self, constants: List[Dict[str, Any]]) -> DocumentationSection:
        """
        Create constants and variables section.

        Args:
            constants (List[Dict[str, Any]]): List of extracted constants.

        Returns:
            DocumentationSection: The constants and variables section of the documentation.
        """
        if not constants:
            return DocumentationSection("Constants and Variables", "")
        content = [
            "## Constants and Variables",
            "| Name | Type | Value |",
            "|------|------|--------|"
        ]
        for const in constants:
            row = f"| `{const['name']}` | `{const['type']}` | `{const['value']}` |"
            content.append(row)
        return DocumentationSection(title="Constants and Variables", content="\n".join(content))

    def _create_changes_section(self) -> DocumentationSection:
        """
        Create recent changes section.

        Returns:
            DocumentationSection: The recent changes section of the documentation.
        """
        content = ["## Recent Changes"]
        changes = self.context.metadata.get('changes', [])
        if changes:
            content.extend(f"- {change}" for change in changes)
        else:
            content.append("- No recent changes recorded.")
        return DocumentationSection(title="Recent Changes", content="\n".join(content))

    def _create_source_code_section(self, metrics: Dict[str, Any]) -> DocumentationSection:
        """
        Create source code section with complexity info.

        Args:
            metrics (Dict[str, Any]): Complexity metrics of the source code.

        Returns:
            DocumentationSection: The source code section of the documentation.
        """
        complexity_header = self._format_complexity_header(metrics)
        content = [
            "## Source Code",
            "```python",
            f'"""{complexity_header}"""',
            self.context.source_code,
            "```"
        ]
        return DocumentationSection(title="Source Code", content="\n".join(content))

    def _format_parameters(self, args: List[ExtractedArgument]) -> str:
        """
        Format function parameters.

        Args:
            args (List[ExtractedArgument]): List of function arguments.

        Returns:
            str: Formatted string of parameters.
        """
        params = []
        for arg in args:
            param = f"{arg.name}: {arg.type_hint or 'Any'}"
            if arg.default_value is not None:
                param += f" = {arg.default_value}"
            params.append(param)
        return f"({', '.join(params)})"

    def _format_complexity_header(self, metrics: Dict[str, Any]) -> str:
        """
        Format complexity information for module header.

        Args:
            metrics (Dict[str, Any]): Complexity metrics of the source code.

        Returns:
            str: Formatted complexity header.
        """
        lines = ["Module Complexity Information:"]
        for name, score in metrics.items():
            if isinstance(score, (int, float)):
                warning = " ⚠️" if score > 10 else ""
                lines.append(f"    {name}: {score}{warning}")
        return "\n".join(lines)

    def _add_parents(self, node: ast.AST) -> None:
        """
        Add parent references to AST nodes.

        Args:
            node (ast.AST): The AST node to process.
        """
        for child in ast.iter_child_nodes(node):
            child.parent = node
            self._add_parents(child)

    def _format_markdown(self, content: str) -> str:
        """Format documentation as markdown with enhanced structure."""
        sections = [
            "# API Documentation\n",
            "## Summary\n",
            f"{content}\n",
            "## Changelog\n",
            self._generate_changelog(),
            "## Classes\n",
            self._generate_class_documentation(),
            "## Functions\n", 
            self._generate_function_documentation(),
            "## Complexity Metrics\n",
            self._generate_complexity_metrics()
        ]

        return "\n".join(sections)

    def _extract_module_info(self) -> Dict[str, Any]:
        """Extract key information about the module."""
        info = {
            'name': getattr(self.context.module_path, 'stem', 'Unknown'),
            'version': '0.1.0',
            'author': 'Unknown',
            'description': self.context.metadata.get('description', '').strip()
        }
        return info

    def _generate_changelog(self) -> str:
        """Generate a changelog section for the documentation."""
        changes = self.context.metadata.get('changes', [])
        if not changes:
            return "No recent changes recorded."
        
        changelog_lines = ["## Changelog"]
        for change in changes:
            changelog_lines.append(f"- {change}")
        
        return "\n".join(changelog_lines)

    def _generate_class_documentation(self) -> str:
        """Generate class documentation section."""
        classes_section = self._create_classes_section(self.code_extractor.extract_code(self.context.source_code).classes)
        return classes_section.content

    def _generate_function_documentation(self) -> str:
        """Generate function documentation section."""
        functions_section = self._create_functions_section(self.code_extractor.extract_code(self.context.source_code).functions)
        return functions_section.content

    def _generate_complexity_metrics(self) -> str:
        """Generate complexity metrics section."""
        metrics = {'Module Complexity': self.metrics_calculator.calculate_complexity(self.tree)}
        return self._format_complexity_header(metrics)