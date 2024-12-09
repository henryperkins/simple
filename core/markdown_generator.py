"""
Markdown documentation generator module.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from core.logger import LoggerSetup, log_debug, log_error
from core.types import DocumentationData, ExtractedClass, ExtractedFunction
import json

class MarkdownGenerator:
    """Generates formatted markdown documentation."""

    def __init__(self) -> None:
        """Initialize the markdown generator."""
        self.logger = LoggerSetup.get_logger(__name__)
        self.search_index = []

    def generate(self, documentation_data: DocumentationData) -> str:
        """Generate markdown documentation."""
        try:
            log_debug("Generating markdown documentation.")

            # Create module info from DocumentationData fields
            module_info = {
                "module_name": documentation_data.module_name,
                "file_path": str(documentation_data.module_path),
                "description": documentation_data.module_summary
            }

            sections = [
                self._generate_header(module_info["module_name"]),
                self._generate_overview(module_info["file_path"], module_info["description"]),
                self._generate_ai_doc_section(documentation_data.ai_content),
                self._generate_class_tables(documentation_data.code_metadata.get("classes", [])),
                self._generate_function_tables(documentation_data.code_metadata.get("functions", [])),
                self._generate_constants_table(documentation_data.code_metadata.get("constants", [])),
                self._generate_source_code(documentation_data.source_code),
                self._generate_search_index()
            ]
            log_debug("Markdown generation completed successfully.")
            return "\n\n".join(filter(None, sections))
        except Exception as e:
            log_error(f"Error generating markdown: {e}", exc_info=True)
            return f"# Error Generating Documentation\n\nAn error occurred: {e}"

    def _generate_header(self, module_name: str) -> str:
        """Generate the module header."""
        log_debug(f"Generating header for module_name: {module_name}.")
        return f"# Module: {module_name}"

    def _generate_overview(self, file_path: str, description: str) -> str:
        """Generate the overview section."""
        log_debug(f"Generating overview for file_path: {file_path}")
        return "\n".join(
            [
                "## Overview",
                f"**File:** `{file_path}`",
                f"**Description:** {description}",
            ]
        )

    def _generate_ai_doc_section(self, ai_documentation: Dict[str, Any]) -> str:
        """
        Generate the AI documentation section using docstring data and AI enhancements.

        Args:
            ai_documentation: Dictionary containing AI-enhanced documentation

        Returns:
            str: Generated markdown documentation
        """
        if not ai_documentation:
            return ""

        sections = [
            "## AI-Generated Documentation\n\n",
            "**Summary:** "
            + (ai_documentation.get("summary", "No summary provided."))
            + "\n\n",
            "**Description:** "
            + (ai_documentation.get("description", "No description provided."))
            + "\n\n",
        ]

        # Format arguments section
        if args := ai_documentation.get("args"):
            sections.append("**Arguments:**")
            for arg in args:
                sections.append(
                    f"- **{arg.get('name', 'Unknown')}** "
                    f"({arg.get('type', 'Any')}): "
                    f"{arg.get('description', 'No description.')}"
                )
            sections.append("\n")

        # Format returns section
        if returns := ai_documentation.get("returns"):
            sections.append(
                f"**Returns:** {returns.get('type', 'Unknown Type')} - "
                f"{returns.get('description', 'No description.')}\n\n"
            )

        # Format raises section
        if raises := ai_documentation.get("raises"):
            sections.append("**Raises:**")
            for exc in raises:
                sections.append(
                    f"- **{exc.get('exception', 'Unknown Exception')}**: "
                    f"{exc.get('description', 'No description.')}"
                )
            sections.append("\n")

        return "\n".join(sections)

    def _generate_class_tables(self, classes: list) -> str:
        """Generate the classes section with tables."""
        try:
            if not classes:
                return ""

            # Initialize the markdown tables
            classes_table = [
                "## Classes",
                "",
                "| Class | Inherits From | Complexity Score* |",
                "|-------|---------------|-------------------|",
            ]

            methods_table = [
                "### Class Methods",
                "",
                "| Class | Method | Parameters | Returns | Complexity Score* |",
                "|-------|--------|------------|---------|-------------------|",
            ]

            for cls in classes:
                # Safely retrieve class properties
                class_name = cls.get("name", "Unknown")
                complexity = cls.get("metrics", {}).get("complexity", 0)
                warning = " ⚠️" if complexity > 10 else ""
                bases = ", ".join(cls.get("bases", []))

                # Add a row for the class
                classes_table.append(
                    f"| `{class_name}` | `{bases}` | {complexity}{warning} |"
                )

                # Add class to search index
                self._add_to_search_index(class_name, cls)

                # Check if the class has methods and iterate over them safely
                for method in cls.get("methods", []):
                    method_name = method.get("name", "Unknown")
                    method_complexity = method.get("metrics", {}).get("complexity", 0)
                    method_warning = " ⚠️" if method_complexity > 10 else ""
                    return_type = method.get("returns", {}).get("type", "Any")

                    # Generate parameters safely
                    params = ", ".join(
                        f"{arg.get('name', 'unknown')}: {arg.get('type', 'Any')}"
                        + (f" = {arg.get('default_value', '')}" if arg.get('default_value') else "")
                        for arg in method.get("args", [])
                    )

                    # Add a row for the method
                    methods_table.append(
                        f"| `{class_name}` | `{method_name}` | "
                        f"`({params})` | `{return_type}` | "
                        f"{method_complexity}{method_warning} |"
                    )

                    # Add method to search index
                    self._add_to_search_index(method_name, method)

            # Combine the tables and return the final markdown string
            return "\n".join(classes_table + [""] + methods_table)
        except Exception as e:
            log_error(f"Error generating class tables: {e}", exc_info=True)
            return "An error occurred while generating class documentation."

    def _generate_function_tables(self, functions: list) -> str:
        """Generate the functions section."""
        try:
            if not functions:
                return ""

            lines = [
                "## Functions",
                "",
                "| Function | Parameters | Returns | Complexity Score* |",
                "|----------|------------|---------|------------------|",
            ]

            for func in functions:
                # Safely get the complexity
                complexity = func.get("metrics", {}).get("complexity", 0)
                warning = " ⚠️" if complexity > 10 else ""

                # Generate parameters safely
                params = ", ".join(
                    f"{arg.get('name', 'unknown')}: {arg.get('type', 'Any')}"
                    + (f" = {arg.get('default_value', '')}" if arg.get('default_value') else "")
                    for arg in func.get("args", [])
                )

                # Safely get the return type
                return_type = func.get("returns", {}).get("type", "Any")

                lines.append(
                    f"| `{func.get('name', 'Unknown')}` | `({params})` | "
                    f"`{return_type}` | {complexity}{warning} |"
                )

                # Add function to search index
                self._add_to_search_index(func.get('name', 'Unknown'), func)

            return "\n".join(lines)
        except Exception as e:
            log_error(f"Error generating function tables: {e}", exc_info=True)
            return "An error occurred while generating function documentation."

    def _generate_constants_table(self, constants: list) -> str:
        """Generate the constants section."""
        try:
            if not constants:
                return ""

            lines = [
                "## Constants and Variables",
                "",
                "| Name | Type | Value |",
                "|------|------|-------|",
            ]

            for const in constants:
                lines.append(
                    f"| `{const.get('name', 'Unknown Name')}` | "
                    f"`{const.get('type', 'Unknown Type')}` | "
                    f"`{const.get('value', 'Unknown Value')}` |"
                )

                # Add constant to search index
                self._add_to_search_index(const.get('name', 'Unknown Name'), const)

            return "\n".join(lines)
        except Exception as e:
            log_error(f"Error generating constants table: {e}", exc_info=True)
            return "An error occurred while generating constants documentation."

    def _generate_source_code(self, source_code: Optional[str]) -> str:
        """Generate the source code section."""
        try:
            if not source_code:
                return ""

            return "\n".join(
                [
                    "## Source Code",
                    f"```python",
                    source_code,
                    "```",
                ]
            )
        except Exception as e:
            log_error(f"Error generating source code section: {e}", exc_info=True)
            return "An error occurred while generating source code documentation."

    def _add_to_search_index(self, name: str, element: Dict[str, Any]) -> None:
        """Add an element to the search index."""
        metadata = {
            "name": name,
            "description": element.get("docstring", ""),
            "tags": element.get("tags", []),
            "categories": element.get("categories", []),
            "keywords": element.get("keywords", []),
            "author": element.get("author", ""),
            "date": element.get("date", ""),
            "version": element.get("version", ""),
            "summary": element.get("summary", ""),
            "references": element.get("references", []),
            "examples": element.get("examples", [])
        }
        self.search_index.append(metadata)

    def _generate_search_index(self) -> str:
        """Generate the search index section."""
        try:
            search_index_json = json.dumps(self.search_index, indent=2)
            return "\n".join(
                [
                    "## Search Index",
                    f"```json",
                    search_index_json,
                    "```",
                ]
            )
        except Exception as e:
            log_error(f"Error generating search index: {e}", exc_info=True)
            return "An error occurred while generating search index."
