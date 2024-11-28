# docs.py
"""
Documentation Management Module

Handles docstring operations and documentation generation with improved structure
and centralized processing.
"""

import ast
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass

from core.logger import LoggerSetup, log_debug, log_error, log_info
from core.docstring_processor import DocstringProcessor, DocstringData, DocumentationSection
from markdown_generator import MarkdownGenerator

logger = LoggerSetup.get_logger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message: str, errors: List[str]) -> None:
        """
        Initialize ValidationError with a message and list of errors.

        Args:
            message (str): Error message.
            errors (List[str]): List of validation errors.
        """
        super().__init__(message)
        self.errors = errors


class DocumentationError(Exception):
    """Custom exception for documentation generation errors."""

    def __init__(self, message: str, details: Dict[str, Any]) -> None:
        """
        Initialize DocumentationError with a message and error details.

        Args:
            message (str): Error message.
            details (Dict[str, Any]): Additional error details.
        """
        super().__init__(message)
        self.details = details


@dataclass
class DocumentationContext:
    """Holds context for documentation generation."""
    source_code: str
    module_path: Optional[str] = None
    include_source: bool = True
    metadata: Optional[Dict[str, Any]] = None


class DocStringManager:
    """Manages docstring operations and documentation generation."""

    def __init__(self, context: DocumentationContext, cache: Optional[Any] = None) -> None:
        """
        Initialize DocStringManager with context and optional cache.

        Args:
            context (DocumentationContext): Documentation generation context.
            cache (Optional[Any]): Optional cache implementation.
        """
        self.context = context
        self.tree: ast.Module = ast.parse(context.source_code)
        self.processor = DocstringProcessor()
        self.cache = cache
        self.changes: List[str] = []
        self.markdown_generator = MarkdownGenerator()

    async def process_docstring(
        self,
        node: ast.AST,
        docstring_data: DocstringData
    ) -> bool:
        """
        Process and insert a docstring for an AST node.

        Args:
            node (ast.AST): AST node to process.
            docstring_data (DocstringData): Structured docstring data.

        Returns:
            bool: Success status of the operation.

        Raises:
            ValidationError: If docstring validation fails.
            DocumentationError: If docstring insertion fails.
        """
        try:
            cache_key = f"validation:{hash(str(docstring_data))}"

            # Check cache
            if self.cache:
                cached_result = await self.cache.get_cached_docstring(cache_key)
                if cached_result:
                    return await self._handle_cached_result(node, cached_result)

            # Validate docstring
            is_valid, errors = self.processor.validate(docstring_data)
            if not is_valid:
                raise ValidationError("Docstring validation failed", errors)

            # Format and insert
            docstring = self.processor.format(docstring_data)
            if self.processor.insert(node, docstring):
                node_name = getattr(node, 'name', 'unknown')
                self.changes.append(f"Updated docstring for {node_name}")

                # Cache successful result
                if self.cache:
                    await self.cache.save_docstring(cache_key, {
                        'docstring': docstring,
                        'valid': True
                    })

                return True

            return False

        except ValidationError as e:
            log_error(f"Docstring validation failed: {e.errors}")
            raise
        except Exception as e:
            log_error(f"Failed to process docstring: {e}")
            raise DocumentationError(
                "Failed to process docstring",
                {'node': getattr(node, 'name', 'unknown'), 'error': str(e)}
            )

    async def generate_documentation(self) -> str:
        """
        Generate complete documentation for the current context.

        Returns:
            str: Generated documentation in markdown format.

        Raises:
            DocumentationError: If documentation generation fails.
        """
        try:
            # Prepare documentation sections
            sections: List[DocumentationSection] = []

            # Module documentation
            if self.context.metadata:
                sections.append(self._create_module_section())

            # Classes documentation
            class_nodes = [n for n in ast.walk(self.tree)
                           if isinstance(n, ast.ClassDef)]
            for node in class_nodes:
                sections.append(await self._create_class_section(node))

            # Functions documentation
            function_nodes = [n for n in ast.walk(self.tree)
                              if isinstance(n, ast.FunctionDef)]
            for node in function_nodes:
                if not self._is_method(node):
                    sections.append(await self._create_function_section(node))

            # Generate markdown
            return self.markdown_generator.generate(
                sections,
                include_source=self.context.include_source,
                source_code=self.context.source_code if self.context.include_source else None,
                module_path=self.context.module_path
            )

        except Exception as e:
            log_error(f"Failed to generate documentation: {e}")
            raise DocumentationError(
                "Documentation generation failed",
                {'error': str(e)}
            )

    def _create_module_section(self) -> DocumentationSection:
        """
        Create module-level documentation section.

        Returns:
            DocumentationSection: Module documentation section.
        """
        return DocumentationSection(
            title="Module Overview",
            content=self.context.metadata.get('description', ''),
            subsections=[
                DocumentationSection(
                    title="Module Information",
                    content=f"Path: {self.context.module_path}\n"
                            f"Last Modified: {self.context.metadata.get('last_modified', 'Unknown')}"
                )
            ]
        )

    async def _create_class_section(self, node: ast.ClassDef) -> DocumentationSection:
        """
        Create class documentation section.

        Args:
            node (ast.ClassDef): Class definition node.

        Returns:
            DocumentationSection: Class documentation section.
        """
        docstring_data = self.processor.parse(ast.get_docstring(node) or '')

        methods_sections: List[DocumentationSection] = []
        for method in [n for n in node.body if isinstance(n, ast.FunctionDef)]:
            methods_sections.append(await self._create_function_section(method))

        return DocumentationSection(
            title=f"Class: {node.name}",
            content=docstring_data.description,
            subsections=[
                DocumentationSection(
                    title="Methods",
                    content="",
                    subsections=methods_sections
                )
            ]
        )

    async def _create_function_section(
        self,
        node: ast.FunctionDef
    ) -> DocumentationSection:
        """
        Create function documentation section.

        Args:
            node (ast.FunctionDef): Function definition node.

        Returns:
            DocumentationSection: Function documentation section.
        """
        docstring_data = self.processor.parse(ast.get_docstring(node) or '')

        return DocumentationSection(
            title=f"{'Method' if self._is_method(node) else 'Function'}: {node.name}",
            content=self.processor.format(docstring_data),
            subsections=[
                DocumentationSection(
                    title="Source",
                    content=f"```python\n{ast.unparse(node)}\n```"
                ) if self.context.include_source else None
            ]
        )

    def _is_method(self, node: ast.FunctionDef) -> bool:
        """
        Check if a function node is a method.

        Args:
            node (ast.FunctionDef): Function node to check.

        Returns:
            bool: True if the function is a method of a class, False otherwise.
        """
        return any(
            isinstance(parent, ast.ClassDef) and node in parent.body
            for parent in ast.walk(self.tree)
        )

    async def _handle_cached_result(
        self,
        node: ast.AST,
        cached_result: Dict[str, Any]
    ) -> bool:
        """
        Handle cached docstring result.

        Args:
            node (ast.AST): AST node to insert docstring into.
            cached_result (Dict[str, Any]): Cached docstring result.

        Returns:
            bool: True if insertion was successful, False otherwise.
        """
        if cached_result.get('valid'):
            return self.processor.insert(node, cached_result['docstring'])
        return False
