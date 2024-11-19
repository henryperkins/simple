"""
Documentation generation system for Python source code.
Handles docstring management and coordinates with MarkdownDocumentationGenerator.
"""

import ast
from typing import Optional, Dict, Any, List
from pathlib import Path
from markdown_generator import MarkdownDocumentationGenerator
from docstring_utils import DocstringValidator
from core.logger import log_info, log_error, log_debug

class DocStringManager:
    """
    Manages docstring operations and documentation generation.
    Works with MarkdownDocumentationGenerator for output generation.
    """

    def __init__(self, source_code: str):
        """
        Initialize with source code.

        Args:
            source_code: The source code to process
        """
        self.source_code = source_code
        self.tree = ast.parse(source_code)
        self.validator = DocstringValidator()
        self.changes = []

    def insert_docstring(self, node: ast.AST, docstring: str) -> bool:
        """
        Insert or update a docstring in an AST node.

        Args:
            node: The AST node to update
            docstring: The docstring to insert

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not isinstance(docstring, str):
                log_error(f"Invalid docstring type for {getattr(node, 'name', 'unknown')}")
                return False

            # Validate docstring
            is_valid, errors = self.validator.validate_docstring({
                'docstring': docstring,
                'summary': docstring.split('\n')[0],
                'parameters': [],
                'returns': {'type': 'None', 'description': 'No return value.'}
            })

            if not is_valid:
                log_error(f"Docstring validation failed: {errors}")
                return False

            # Insert docstring node
            node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
            
            # Record change
            node_name = getattr(node, 'name', 'unknown')
            self.changes.append(f"Updated docstring for {node_name}")
            log_info(f"Inserted docstring for {node_name}")
            
            return True

        except Exception as e:
            log_error(f"Failed to insert docstring: {e}")
            return False

    def update_source_code(self, documentation_entries: List[Dict]) -> str:
        """
        Update source code with new docstrings.

        Args:
            documentation_entries: List of documentation updates

        Returns:
            str: Updated source code
        """
        try:
            modified = False
            for entry in documentation_entries:
                node_type = entry.get('type', 'function')
                name = entry.get('name')
                docstring = entry.get('docstring')

                if not all([name, docstring]):
                    continue

                # Find and update matching nodes
                for node in ast.walk(self.tree):
                    if (isinstance(node, (ast.FunctionDef if node_type == 'function' else ast.ClassDef)) 
                            and node.name == name):
                        if self.insert_docstring(node, docstring):
                            modified = True

            return ast.unparse(self.tree) if modified else self.source_code

        except Exception as e:
            log_error(f"Failed to update source code: {e}")
            return self.source_code

    def generate_documentation(
        self,
        module_path: Optional[str] = None,
        include_source: bool = True
    ) -> str:
        """
        Generate documentation using MarkdownDocumentationGenerator.

        Args:
            module_path: Optional path to the module file
            include_source: Whether to include source code in documentation

        Returns:
            str: Generated documentation
        """
        try:
            generator = MarkdownDocumentationGenerator(
                source_code=self.source_code if include_source else None,
                module_path=module_path
            )

            # Add recorded changes
            for change in self.changes:
                generator.add_change(change)

            return generator.generate_markdown()

        except Exception as e:
            log_error(f"Failed to generate documentation: {e}")
            return f"# Documentation Generation Failed\n\nError: {str(e)}"

    def process_batch(
        self,
        entries: List[Dict],
        module_path: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
        """
        Process a batch of documentation entries.

        Args:
            entries: List of documentation entries
            module_path: Optional path to the module file

        Returns:
            Optional[Dict[str, str]]: Dictionary containing updated code and documentation,
                                    or None if processing failed
        """
        try:
            updated_code = self.update_source_code(entries)
            documentation = self.generate_documentation(
                module_path=module_path,
                include_source=True
            )

            if updated_code and documentation:
                return {
                    "code": updated_code,
                    "documentation": documentation
                }
            return None

        except Exception as e:
            log_error(f"Batch processing failed: {e}")
            return None

    @staticmethod
    def extract_docstring(node: ast.AST) -> Optional[str]:
        """
        Extract existing docstring from an AST node.

        Args:
            node: AST node to extract docstring from

        Returns:
            Optional[str]: Extracted docstring if found
        """
        try:
            return ast.get_docstring(node)
        except Exception as e:
            log_error(f"Failed to extract docstring: {e}")
            return None