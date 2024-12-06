"""Markdown documentation generator module."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from core.logger import LoggerSetup  # Import the LoggerSetup utility

@dataclass
class MarkdownConfig:
    """Configuration for markdown generation."""

    include_toc: bool = True
    include_timestamp: bool = True
    code_language: str = "python"
    include_source: bool = True


class MarkdownGenerator:
    """Generates formatted markdown documentation."""

    def __init__(self, config: Optional[MarkdownConfig] = None):
        """Initialize the markdown generator."""
        self.config = config or MarkdownConfig()
        self.logger = LoggerSetup.get_logger(
            name=__name__
        )  # Use LoggerSetup to initialize logger

    def generate(self, context: Dict[str, Any]) -> str:
        """Generate markdown documentation."""
        try:
            self.logger.debug("Generating markdown documentation.")

            # Accessing context elements safely
            module_name = context.get("module_name", "Unknown Module")
            file_path = context.get("file_path", "Unknown File")
            description = context.get("description", "No description provided.")
            classes = context.get("classes", [])
            functions = context.get("functions", [])
            constants = context.get("constants", [])
            changes = context.get("changes", [])
            source_code = context.get("source_code", "")
            ai_documentation = context.get("ai_documentation", {})

            sections = [
                self._generate_header(module_name),
                self._generate_overview(file_path, description),
                self._generate_ai_doc_section(ai_documentation),
                self._generate_class_tables(classes),
                self._generate_function_tables(functions),
                self._generate_constants_table(constants),
                self._generate_changes(changes),
                self._generate_source_code(source_code, context),  # Pass the context
            ]
            self.logger.debug("Markdown generation completed successfully.")
            return "\n\n".join(filter(None, sections))
        except Exception as e:
            self.logger.error(f"Error generating markdown: {e}", exc_info=True)
            return f"# Error Generating Documentation\n\nAn error occurred: {e}"

    def _generate_header(self, module_name: str) -> str:
        """Generate the module header."""
        self.logger.debug(f"Generating header for module_name: {module_name}.")
        return f"# Module: {module_name}"

    def _generate_overview(self, file_path: str, description: str) -> str:
        """Generate the overview section."""
        self.logger.debug(f"Generating overview for file_path: {file_path}")
        return "\n".join(
            [
                "## Overview",
                f"**File:** `{file_path}`",
                f"**Description:** {description}",
            ]
        )

    def _generate_ai_doc_section(self, ai_documentation: Dict[str, Any]) -> str:
        """Generates the AI documentation section."""
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

        if ai_documentation.get("args"):
            sections.append("**Arguments:**")
            for arg in ai_documentation["args"]:
                sections.append(
                    f"- **{arg.get('name', 'Unknown Name')}** "
                    f"({arg.get('type', 'Unknown Type')}): "
                    f"{arg.get('description', 'No description.')}"
                )
            sections.append("\n")

        if ai_documentation.get("returns"):
            returns = ai_documentation["returns"]
            sections.append(
                f"**Returns:** {returns.get('type', 'Unknown Type')} - "
                f"{returns.get('description', 'No description.')}\n\n"
            )

        if ai_documentation.get("raises"):
            sections.append("**Raises:**")
            for raise_ in ai_documentation["raises"]:
                sections.append(
                    f"- **{raise_.get('exception', 'Unknown Exception')}**: "
                    f"{raise_.get('description', 'No description.')}"
                )
            sections.append("\n")

        return "\n".join(sections)

    def _generate_class_tables(self, classes: List[Any]) -> str:
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
                class_name = getattr(cls, "name", "Unknown Class")
                metrics = getattr(cls, "metrics", {})
                complexity = (
                    metrics.get("complexity", 0) if isinstance(metrics, dict) else 0
                )
                warning = " ⚠️" if complexity > 10 else ""
                bases = (
                    ", ".join(getattr(cls, "bases", []))
                    if isinstance(getattr(cls, "bases", None), list)
                    else "None"
                )

                # Add a row for the class
                classes_table.append(
                    f"| `{class_name}` | `{bases}` | {complexity}{warning} |"
                )

                # Check if the class has methods and iterate over them safely
                if hasattr(cls, "methods") and isinstance(cls.methods, list):
                    for method in cls.methods:
                        method_name = getattr(method, "name", "Unknown Method")
                        method_metrics = getattr(method, "metrics", {})
                        method_complexity = (
                            method_metrics.get("complexity", 0)
                            if isinstance(method_metrics, dict)
                            else 0
                        )
                        method_warning = " ⚠️" if method_complexity > 10 else ""
                        return_type = getattr(method, "return_type", "Any")

                        # Generate parameters safely
                        if hasattr(method, "args") and isinstance(method.args, list):
                            params = ", ".join(
                                f"{getattr(arg, 'name', 'Unknown')}: {getattr(arg, 'type', 'Any')}"
                                + (
                                    f" = {getattr(arg, 'default_value', '')}"
                                    if getattr(arg, "default_value", None)
                                    else ""
                                )
                                for arg in method.args
                            )
                        else:
                            params = "None"

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

    def _generate_function_tables(self, functions: List[Any]) -> str:
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
                complexity = 0
                warning = ""
                if hasattr(func, "metrics") and isinstance(func.metrics, dict):
                    complexity = func.metrics.get("complexity", 0)
                    warning = " ⚠️" if complexity > 10 else ""

                # Safely generate parameters
                params = ""
                if hasattr(func, "args") and isinstance(func.args, list):
                    param_list = []
                    for arg in func.args:
                        arg_name = getattr(arg, "name", "Unknown")
                        arg_type = getattr(arg, "type", "Any")
                        default_value = getattr(arg, "default_value", None)
                        param_str = f"{arg_name}: {arg_type}"
                        if default_value is not None:
                            param_str += f" = {default_value}"
                        param_list.append(param_str)
                    params = ", ".join(param_list)
                else:
                    params = "None"

                # Safely get the return type
                return_type = getattr(func, "return_type", "Any")

                lines.append(
                    f"| `{getattr(func, 'name', 'Unknown')}` | `({params})` | "
                    f"`{return_type}` | {complexity}{warning} |"
                )

            return "\n".join(lines)
        except Exception as e:
            self.logger.error(f"Error generating function tables: {e}", exc_info=True)
            return "An error occurred while generating function documentation."

    def _generate_constants_table(self, constants: List[Any]) -> str:
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

    def _generate_changes(self, changes: List[Any]) -> str:
        """Generate the recent changes section."""
        try:
            if not changes:
                return ""

            lines = ["## Recent Changes"]

            for change in changes:
                date = change.get("date", datetime.now().strftime("%Y-%m-%d"))
                description = change.get("description", "No description.")
                lines.append(f"- [{date}] {description}")

            return "\n".join(lines)
        except Exception as e:
            self.logger.error(f"Error generating changes section: {e}", exc_info=True)
            return "An error occurred while generating changes documentation."

    def _generate_source_code(self, source_code: str, context: Dict[str, Any]) -> str:
        """Generate the source code section."""
        try:
            if not self.config.include_source or not source_code:
                return ""

            complexity_scores = []

            # Access context elements safely and handle potential missing data
            functions = context.get("functions", [])
            classes = context.get("classes", [])
            description = context.get("description", "[description]")

            for func in functions:
                complexity = (
                    func.metrics.get("complexity", 0) if hasattr(func, "metrics") else 0
                )
                warning = " ⚠️" if complexity > 10 else ""
                complexity_scores.append(f"    {func.name}: {complexity}{warning}")

            for cls in classes:
                if hasattr(cls, "methods"):
                    for method in cls.methods:
                        complexity = (
                            method.metrics.get("complexity", 0)
                            if hasattr(method, "metrics")
                            else 0
                        )
                        warning = " ⚠️" if complexity > 10 else ""
                        complexity_scores.append(
                            f"    {method.name}: {complexity}{warning}"
                        )

            docstring = f'"""Module for handling {description}.\n\n'
            if complexity_scores:
                docstring += "Complexity Scores:\n" + "\n".join(complexity_scores) + "\n"
            docstring += '"""\n\n'

            return "\n".join(
                [
                    "## Source Code",
                    f"```{self.config.code_language}",
                    docstring + source_code,
                    "```",
                ]
            )
        except Exception as e:
            self.logger.error(f"Error generating source code section: {e}", exc_info=True)
            return "An error occurred while generating source code documentation."