"""
Docstring processing module.
"""

import ast
from typing import Optional, Dict, Any, List, Union
from docstring_parser import parse as parse_docstring
from core.logger import LoggerSetup
from core.metrics import Metrics
from core.types import DocstringData
from exceptions import DocumentationError

class DocstringProcessor:
    """Processes docstrings by parsing, validating, and formatting them."""

    def __init__(self, metrics: Optional[Metrics] = None) -> None:
        """
        Initialize docstring processor.

        Args:
            metrics: Optional metrics instance for complexity calculations
        """
        self.logger = LoggerSetup.get_logger(__name__)
        self.metrics = metrics or Metrics()

    def parse(self, docstring: Union[Dict[str, Any], str]) -> DocstringData:
        """Parse a raw docstring into structured format."""
        try:
            # If it's already a dict, convert directly
            if isinstance(docstring, dict):
                return DocstringData(
                    summary=docstring.get('summary', ''),
                    description=docstring.get('description', ''),
                    args=docstring.get('args', []),
                    returns=docstring.get('returns', {'type': 'Any', 'description': ''}),
                    raises=docstring.get('raises', []),
                    complexity=docstring.get('complexity', 1)
                )

            # If it's a string, try to parse as JSON first
            if isinstance(docstring, str) and docstring.strip().startswith('{'):
                try:
                    import json
                    doc_dict = json.loads(docstring)
                    return self.parse(doc_dict)
                except json.JSONDecodeError:
                    pass

            # Otherwise parse as regular docstring
            parsed = parse_docstring(docstring)
            
            return DocstringData(
                summary=parsed.short_description or '',
                description=parsed.long_description or '',
                args=[{
                    'name': param.arg_name,
                    'type': param.type_name or 'Any',
                    'description': param.description or ''
                } for param in parsed.params],
                returns={
                    'type': parsed.returns.type_name if parsed.returns else 'Any',
                    'description': parsed.returns.description if parsed.returns else ''
                },
                raises=[{
                    'exception': e.type_name or 'Exception',
                    'description': e.description or ''
                } for e in parsed.raises] if parsed.raises else []
            )

        except Exception as e:
            self.logger.error(f"Error parsing docstring: {e}")
            raise DocumentationError(f"Failed to parse docstring: {e}")

    def format(self, data: DocstringData) -> str:
        """Format structured docstring data into a string."""
        lines = []

        # Add summary
        if data.summary:
            lines.extend([data.summary, ""])

        # Add detailed description if different from summary
        if data.description and data.description != data.summary:
            lines.extend([data.description, ""])

        # Add arguments section if present
        if data.args:
            lines.append("Args:")
            for arg in data.args:
                arg_desc = f"    {arg['name']} ({arg['type']}): {arg['description']}"
                if arg.get('optional', False):
                    arg_desc += " (Optional)"
                if 'default_value' in arg and arg['default_value'] is not None:
                    arg_desc += f", default: {arg['default_value']}"
                lines.append(arg_desc)
            lines.append("")

        # Add returns section
        if data.returns:
            lines.append("Returns:")
            lines.append(f"    {data.returns['type']}: {data.returns['description']}")
            lines.append("")

        # Add raises section if present
        if data.raises:
            lines.append("Raises:")
            for exc in data.raises:
                lines.append(f"    {exc['exception']}: {exc['description']}")
            lines.append("")

        # Add complexity warning if high
        if data.complexity and data.complexity > 10:
            lines.append(f"Warning: High complexity score ({data.complexity}) ⚠️")

        return "\n".join(lines).strip()
    
    def extract_from_node(self, node: ast.AST) -> DocstringData:
        """
        Extract docstring from an AST node.

        Args:
            node (ast.AST): The AST node to extract from.

        Returns:
            DocstringData: The extracted docstring data.

        Raises:
            DocumentationError: If extraction fails.
        """
        try:
            raw_docstring = ast.get_docstring(node) or ""
            docstring_data = self.parse(raw_docstring)
            
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                complexity = self.metrics.calculate_complexity(node)
                docstring_data.complexity = complexity

            return docstring_data

        except Exception as e:
            self.logger.error(f"Error extracting docstring: {e}")
            raise DocumentationError(f"Failed to extract docstring: {e}")

    def insert_docstring(self, node: ast.AST, docstring: str) -> ast.AST:
        """
        Insert docstring into an AST node.

        Args:
            node (ast.AST): The AST node to update
            docstring (str): The docstring to insert

        Returns:
            ast.AST: The updated node

        Raises:
            DocumentationError: If insertion fails
        """
        try:
            # Handle module-level docstrings
            if isinstance(node, ast.Module):
                docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                # Remove existing docstring if present
                if node.body and isinstance(node.body[0], ast.Expr) and \
                isinstance(node.body[0].value, ast.Constant):
                    node.body.pop(0)
                node.body.insert(0, docstring_node)
                return node
                
            # Handle class and function docstrings
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                # Remove existing docstring if present
                if node.body and isinstance(node.body[0], ast.Expr) and \
                isinstance(node.body[0].value, ast.Constant):
                    node.body.pop(0)
                node.body.insert(0, docstring_node)
                return node
            else:
                raise ValueError(f"Invalid node type for docstring: {type(node)}")

        except Exception as e:
            self.logger.error(f"Error inserting docstring: {e}")
            raise DocumentationError(f"Failed to insert docstring: {e}")

    def update_docstring(self, existing: str, new_content: str) -> str:
        """
        Update an existing docstring with new content.

        Args:
            existing (str): Existing docstring.
            new_content (str): New content to merge.

        Returns:
            str: Updated docstring.
        """
        try:
            # Parse both docstrings
            existing_data = self.parse(existing)
            new_data = self.parse(new_content)

            # Merge data, preferring new content but keeping existing if new is empty
            merged = DocstringData(
                summary=new_data.summary or existing_data.summary,
                description=new_data.description or existing_data.description,
                args=new_data.args or existing_data.args,
                returns=new_data.returns or existing_data.returns,
                raises=new_data.raises or existing_data.raises,
                complexity=new_data.complexity or existing_data.complexity
            )

            return self.format(merged)

        except Exception as e:
            self.logger.error(f"Error updating docstring: {e}")
            raise DocumentationError(f"Failed to update docstring: {e}")