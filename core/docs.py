"""
Documentation Management Module

Handles docstring operations and documentation generation with improved structure
and centralized processing.
"""

from datetime import datetime
import ast
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field
from core.logger import LoggerSetup
from core.docstring_processor import DocstringProcessor, DocumentationSection
from core.markdown_generator import MarkdownGenerator, MarkdownConfig
from core.code_extraction import CodeExtractor, ExtractionContext, ExtractedClass, ExtractedFunction, ExtractedArgument
from core.metrics import Metrics
# Remove circular import
# from ai_interaction import AIInteractionHandler
logger = LoggerSetup.get_logger(__name__)

class DocumentationError(Exception):
    """Custom exception for documentation generation errors."""

    def __init__(self, message: str, details: dict):
        self.details = details
        super().__init__(message)

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

    def __init__(self, context: DocumentationContext, ai_handler: AIInteractionHandler, cache: Optional[Any] = None) -> None:
        """
        Initialize DocStringManager with context, AI handler, and optional cache.

        Args:
            context (DocumentationContext): The context for documentation generation.
            ai_handler (AIInteractionHandler): The AI handler for generating docstrings.
            cache (Optional[Any]): Optional cache for storing intermediate results.
        """
        self.logger = LoggerSetup.get_logger(__name__)
        self.context = context
        self.ai_handler = ai_handler
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
            # Parse the source code
            tree = ast.parse(self.context.source_code)
            
            # Process all nodes for docstrings first
            modified_code = await self._process_all_nodes(tree)
            self.context.source_code = modified_code  # Update source code with new docstrings
            
            # Extract all code elements
            extraction_result = self.code_extractor.extract_code(modified_code)
            
            # Debug logging
            self.logger.debug(f"Found {len(extraction_result.classes)} classes")
            self.logger.debug(f"Found {len(extraction_result.functions)} functions")
            
            # Generate sections with the updated code
            sections = [
                self._create_module_section(self.context.metadata.get('ai_generated', '')),
                self._create_overview_section(),
                self._create_classes_section(extraction_result.classes),
                self._create_class_methods_section(extraction_result.classes),
                self._create_functions_section(extraction_result.functions),
                self._create_constants_section(extraction_result.constants),
                self._create_changes_section(),
                self._create_source_code_section(modified_code, extraction_result.metrics)
            ]
            
            return self.markdown_generator.generate(sections, self.context.module_path)
            
        except Exception as e:
            self.logger.error(f"Documentation generation failed: {str(e)}")
            raise DocumentationError("Documentation generation failed", {'error': str(e)})

    async def _process_all_nodes(self, tree: ast.AST) -> str:
        """Process all nodes in the AST and add docstrings where missing."""
        modified = False
        
        # Track all nodes that need docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                self.logger.debug(f"Processing node: {getattr(node, 'name', 'unknown')}")
                
                # Check if node needs a docstring
                if not ast.get_docstring(node):
                    self.logger.debug(f"Generating docstring for {getattr(node, 'name', 'unknown')}")
                    
                    # Generate docstring using AI
                    result = await self.ai_handler.process_code(
                        source_code=ast.unparse(node),
                        cache_key=f"doc:{getattr(node, 'name', 'unknown')}",
                    )
                    
                    if result:
                        _, docstring = result
                        # Insert the docstring
                        self._insert_docstring(node, docstring)
                        modified = True
                        self.logger.debug(f"Added docstring to {getattr(node, 'name', 'unknown')}")
        
        if modified:
            return ast.unparse(tree)
        return self.context.source_code

    def _insert_docstring(self, node: ast.AST, docstring: str) -> None:
        """Insert docstring into node."""
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            return

        docstring_node = ast.Expr(value=ast.Constant(value=docstring))
        
        # Insert at the beginning of the body
        if hasattr(node, 'body'):
            # Remove existing docstring if present
            if node.body and isinstance(node.body[0], ast.Expr) and \
               isinstance(node.body[0].value, ast.Constant):
                node.body.pop(0)
            node.body.insert(0, docstring_node)

    def _create_module_section(self, ai_docs: str) -> DocumentationSection:
        """Create module section with AI-generated documentation."""
        return DocumentationSection(
            title="Module Documentation",
            content=ai_docs or "No documentation available."
        )

    def _create_overview_section(self) -> DocumentationSection:
        """Create overview section."""
        file_path = self.context.module_path or Path('unknown')
        description = self.context.metadata.get('description', 'No description provided.')
        content = [
            "## Overview",
            f"**File:** `{file_path}`",
            f"**Description:** {description}"
        ]
        return DocumentationSection(title="Overview", content="\n".join(content))

    def _create_classes_section(self, classes: List[ExtractedClass]) -> DocumentationSection:
        """Create classes section."""
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
        """Create class methods section."""
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
        """Create functions section."""
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
        """Create constants and variables section."""
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
        """Create recent changes section."""
        content = ["## Recent Changes"]
        changes = self.context.metadata.get('changes', [])
        if changes:
            content.extend(f"- {change}" for change in changes)
        else:
            content.append("- No recent changes recorded.")
        return DocumentationSection(title="Recent Changes", content="\n".join(content))

    def _create_source_code_section(self, modified_code: str, metrics: Dict[str, Any]) -> DocumentationSection:
        """Create source code section with complexity info."""
        complexity_header = self._format_complexity_header(metrics)
        content = [
            "## Source Code",
            "```python",
            f'"""{complexity_header}"""',
            modified_code,
            "```"
        ]
        return DocumentationSection(title="Source Code", content="\n".join(content))

    def _format_parameters(self, args: List[ExtractedArgument]) -> str:
        """Format function parameters."""
        params = []
        for arg in args:
            param = f"{arg.name}: {arg.type_hint or 'Any'}"
            if arg.default_value is not None:
                param += f" = {arg.default_value}"
            params.append(param)
        return f"({', '.join(params)})"

    def _format_complexity_header(self, metrics: Dict[str, Any]) -> str:
        """Format complexity information for module header."""
        lines = ["Module Complexity Information:"]
        for name, score in metrics.items():
            if isinstance(score, (int, float)):
                warning = " ⚠️" if score > 10 else ""
                lines.append(f"    {name}: {score}{warning}")
        return "\n".join(lines)

    def _add_parents(self, node: ast.AST) -> None:
        """Add parent references to AST nodes."""
        for child in ast.iter_child_nodes(node):
            child.parent = node
            self._add_parents(child)