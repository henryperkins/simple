"""
Markdown documentation generator module.
"""

from typing import Optional, Dict, Any
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.types import DocumentationData
from core.exceptions import DocumentationError


class MarkdownGenerator:
    """Generates formatted markdown documentation."""

    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """
        Initialize the markdown generator.

        Args:
            correlation_id: Optional correlation ID for tracking related operations.
        """
        self.correlation_id = correlation_id
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__),
            extra={"correlation_id": self.correlation_id}
        )

    def generate(self, documentation_data: DocumentationData) -> str:
        """Generate markdown documentation."""
        try:
            self.logger.debug("Generating markdown documentation.")

            # Check for complete information
            if not documentation_data.source_code:
                self.logger.error(
                    "Source code is missing - cannot generate documentation")
                return "# Error: Missing Source Code\n\nDocumentation cannot be generated without source code."

            if not self._has_complete_information(documentation_data):
                self.logger.warning(
                    "Incomplete information received for markdown generation", extra={'correlation_id': self.correlation_id})
                # Continue with partial documentation but add warning header
                sections = [
                    "# ⚠️ Warning: Partial Documentation\n\nSome information may be missing or incomplete.\n"]
            else:
                sections = []

            # Create module info from DocumentationData fields
            module_info = {
                "module_name": documentation_data.module_name,
                "file_path": str(documentation_data.module_path),
                "description": documentation_data.module_summary
            }

            sections = [
                self._generate_header(module_info["module_name"]),
                self._generate_overview(
                    module_info["file_path"], module_info["description"]),
                self._generate_ai_doc_section(documentation_data.ai_content),
                self._generate_class_tables(
                    documentation_data.code_metadata.get("classes", [])),
                self._generate_function_tables(
                    documentation_data.code_metadata.get("functions", [])),
                self._generate_constants_table(
                    documentation_data.code_metadata.get("constants", [])),
                self._generate_source_code(documentation_data.source_code),
            ]
            markdown = "\n\n".join(filter(None, sections))
            if not self._has_complete_information(documentation_data):
                self.logger.warning(
                    "Generated partial documentation due to incomplete information")
            else:
                self.logger.debug("Generated complete documentation successfully")
            return markdown
        except DocumentationError as de:
            error_msg = f"DocumentationError: {de} in markdown generation with correlation ID: {self.correlation_id}"
            self.logger.error(error_msg, extra={'correlation_id': self.correlation_id})
            return f"# Error Generating Documentation\n\nDocumentationError: {de}"
        except Exception as e:
            error_msg = f"Unexpected error: {e} in markdown generation with correlation ID: {self.correlation_id}"
            self.logger.error(error_msg, exc_info=True, extra={'correlation_id': self.correlation_id})
            return f"# Error Generating Documentation\n\nAn error occurred: {e}"

    def _has_complete_information(self, documentation_data: DocumentationData) -> bool:
        """Check if the documentation data contains complete information."""
        missing_fields = []

        # Check required fields have content
        required_fields = {
            'module_name': documentation_data.module_name,
            'module_path': documentation_data.module_path,
            'source_code': documentation_data.source_code,
            'code_metadata': documentation_data.code_metadata
        }

        missing_fields = [
            field for field, value in required_fields.items()
            if not value or (isinstance(value, str) and not value.strip())
        ]

        # These fields are optional but we'll log if they're missing
        if not documentation_data.module_summary:
            self.logger.warning(
                f"Module {documentation_data.module_name} is missing a summary", extra={'correlation_id': self.correlation_id})
            documentation_data.module_summary = (
                documentation_data.ai_content.get('summary') or
                documentation_data.docstring_data.summary or
                "No module summary provided."
            )

        if not documentation_data.ai_content:
            self.logger.warning(
                f"Module {documentation_data.module_name} is missing AI-generated content", extra={'correlation_id': self.correlation_id})
            documentation_data.ai_content = {
                'summary': documentation_data.module_summary
            }

        # Only fail validation if critical fields are missing
        if missing_fields:
            self.logger.warning(
                f"Missing required fields: {', '.join(missing_fields)}", extra={'correlation_id': self.correlation_id})
            return False

        return True

    def _generate_header(self, module_name: str) -> str:
        """Generate the module header."""
        self.logger.debug(f"Generating header for module_name: {module_name}.")
        return f"# Module: {module_name}"

    def _generate_overview(self, file_path: str, description: str) -> str:
        """Generate the overview section."""
        self.logger.debug(f"Generating overview for file_path: {file_path}")

        # Use a default description if none provided
        if not description or description.isspace():
            description = "No description available."
            self.logger.warning(f"No description provided for {file_path}")

        # Log the description being used
        self.logger.debug(f"Using description: {description[:100]}...")

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

                # Check if the class has methods and iterate over them safely
                for method in cls.get("methods", []):
                    method_name = method.get("name", "Unknown")
                    method_complexity = method.get(
                        "metrics", {}).get("complexity", 0)
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

            # Combine the tables and return the final markdown string
            return "\n".join(classes_table + [""] + methods_table)
        except Exception as e:
            self.logger.error(f"Error generating class tables: {e}", exc_info=True)
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

            return "\n".join(lines)
        except Exception as e:
            self.logger.error(f"Error generating function tables: {e}", exc_info=True)
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

            return "\n".join(lines)
        except Exception as e:
            self.logger.error(f"Error generating constants table: {e}", exc_info=True)
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
            self.logger.error(
                f"Error generating source code section: {e}", exc_info=True)
            return "An error occurred while generating source code documentation."
