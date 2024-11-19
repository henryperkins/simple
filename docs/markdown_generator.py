"""
Markdown documentation generator module.

Generates standardized markdown documentation for Python modules
following a consistent template format.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import ast
from datetime import datetime
import re
from core.logger import log_info, log_error, log_debug

class MarkdownDocumentationGenerator:
    """Generates standardized markdown documentation for Python modules."""

    def __init__(self, source_code: str, module_path: Optional[str] = None):
        """
        Initialize the markdown generator.

        Args:
            source_code: Source code to document
            module_path: Optional path to the module file
        """
        self.source_code = source_code
        self.module_path = Path(module_path) if module_path else Path("module.py")
        self.tree = ast.parse(source_code)
        self.docstring = ast.get_docstring(self.tree) or ""
        self.changes: List[str] = []

    def generate_markdown(self) -> str:
        """
        Generate complete markdown documentation.
        
        Returns:
            str: Complete markdown documentation
        """
        sections = [
            self._generate_header(),
            self._generate_overview(),
            self._generate_classes_section(),
            self._generate_functions_section(),
            self._generate_constants_section(),
            self._generate_changes_section(),
        ]
        
        if self.source_code:
            sections.append(self._generate_source_section())
            
        return "\n\n".join(filter(None, sections))

    def _generate_header(self) -> str:
        """Generate module header section."""
        return f"# Module: {self.module_path.stem}"

    def _generate_overview(self) -> str:
        """Generate overview section."""
        description = self.docstring.split('\n')[0] if self.docstring else "No description available."
        
        return f"""## Overview
**File:** `{self.module_path}`
**Description:** {description}"""

    def _generate_classes_section(self) -> str:
        """Generate classes section with methods."""
        classes = [node for node in ast.walk(self.tree) if isinstance(node, ast.ClassDef)]
        if not classes:
            return ""

        # Classes overview table
        output = """## Classes

| Class | Inherits From | Complexity Score* |
|-------|---------------|------------------|"""

        for cls in classes:
            bases = ', '.join(ast.unparse(base) for base in cls.bases) or '-'
            score = self._get_complexity_score(cls)
            output += f"\n| `{cls.name}` | `{bases}` | {score} |"

        # Methods table
        output += """\n\n### Class Methods

| Class | Method | Parameters | Returns | Complexity Score* |
|-------|--------|------------|---------|------------------|"""

        for cls in classes:
            for method in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
                params = self._format_parameters(method)
                returns = self._get_return_annotation(method)
                score = self._get_complexity_score(method)
                
                output += (f"\n| `{cls.name}` | `{method.name}` | "
                          f"`{params}` | `{returns}` | {score} |")

        return output

    def _generate_functions_section(self) -> str:
        """Generate functions section."""
        functions = [
            node for node in ast.walk(self.tree) 
            if isinstance(node, ast.FunctionDef) 
            and isinstance(node.parent, ast.Module)
        ]
        
        if not functions:
            return ""

        output = """## Functions

| Function | Parameters | Returns | Complexity Score* |
|----------|------------|---------|------------------|"""

        for func in functions:
            params = self._format_parameters(func)
            returns = self._get_return_annotation(func)
            score = self._get_complexity_score(func)
            
            output += f"\n| `{func.name}` | `{params}` | `{returns}` | {score} |"

        return output

    def _generate_constants_section(self) -> str:
        """Generate constants section."""
        constants = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.parent, ast.Module):
                # Type-annotated assignments
                if isinstance(node.target, ast.Name) and node.target.id.isupper():
                    constants.append((
                        node.target.id,
                        ast.unparse(node.annotation),
                        ast.unparse(node.value) if node.value else "None"
                    ))
            elif isinstance(node, ast.Assign) and isinstance(node.parent, ast.Module):
                # Regular assignments
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        try:
                            value = ast.unparse(node.value)
                            type_name = type(eval(value)).__name__
                            constants.append((target.id, type_name, value))
                        except:
                            constants.append((target.id, "Any", ast.unparse(node.value)))

        if not constants:
            return ""

        output = """## Constants and Variables

| Name | Type | Value |
|------|------|-------|"""

        for name, type_name, value in constants:
            output += f"\n| `{name}` | `{type_name}` | `{value}` |"

        return output

    def _generate_changes_section(self) -> str:
        """Generate recent changes section."""
        if not self.changes:
            today = datetime.now().strftime('%Y-%m-%d')
            self.changes.append(f"[{today}] Initial documentation generated")

        return "## Recent Changes\n" + "\n".join(f"- {change}" for change in self.changes)

    def _generate_source_section(self) -> str:
        """Generate source code section."""
        return f"""## Source Code
```python
{self.source_code}
```"""

    def _format_parameters(self, node: ast.FunctionDef) -> str:
        """Format function parameters with types."""
        params = []
        
        for arg in node.args.args:
            if arg.arg == 'self':
                continue
                
            param_str = arg.arg
            if arg.annotation:
                param_str += f": {ast.unparse(arg.annotation)}"
            elif arg in node.args.defaults:
                # Has default value
                default_idx = len(node.args.args) - len(node.args.defaults)
                if arg_idx := node.args.args.index(arg) >= default_idx:
                    default = ast.unparse(node.args.defaults[arg_idx - default_idx])
                    param_str += f" = {default}"
                    
            params.append(param_str)
            
        return ", ".join(params)

    def _get_return_annotation(self, node: ast.FunctionDef) -> str:
        """Get function return type annotation."""
        if node.returns:
            return ast.unparse(node.returns)
        return "None"

    def _get_complexity_score(self, node: ast.AST) -> str:
        """Get complexity score from docstring or calculate it."""
        docstring = ast.get_docstring(node)
        if docstring:
            match = re.search(r'Complexity Score:\s*(\d+)', docstring)
            if match:
                score = int(match.group(1))
                return f"{score} ⚠️" if score > 10 else str(score)
        return "-"

    def add_change(self, description: str):
        """Add a change entry to the documentation."""
        date = datetime.now().strftime('%Y-%m-%d')
        self.changes.append(f"[{date}] {description}")

def generate_module_documentation(
    source_code: str,
    module_path: str,
    changes: Optional[List[str]] = None
) -> str:
    """
    Generate markdown documentation for a module.

    Args:
        source_code: Source code to document
        module_path: Path to the module file
        changes: Optional list of recent changes

    Returns:
        str: Generated markdown documentation
    """
    generator = MarkdownDocumentationGenerator(source_code, module_path)
    if changes:
        for change in changes:
            generator.add_change(change)
    return generator.generate_markdown()