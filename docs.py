#!/usr/bin/env python3
"""
docs.py - Documentation Generation System

This module provides a comprehensive system for generating documentation from Python source code,
including docstring management, markdown generation, and documentation workflow automation.

Classes:
    DocStringManager: Manages docstring operations for source code files.
    MarkdownGenerator: Generates markdown documentation from Python code elements.
    DocumentationManager: Manages the overall documentation generation process.

Functions:
    main(): Demonstrates usage of the documentation system.
"""

import ast
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from datetime import datetime
from docstring_utils import parse_docstring, parse_and_validate_docstring
from core.logger import log_error
from core.utils import handle_exceptions  # Import the decorator
from docstring_utils import DocstringValidator  # Assuming this is where the validator is defined

class DocStringManager:
    """
    Manages docstring operations for source code files.

    Attributes:
        source_code (str): The source code to manage docstrings for.
        tree (ast.AST): The abstract syntax tree of the source code.
    """

    def __init__(self, source_code: str):
        """
        Initialize with source code and validator.

        Args:
            source_code (str): The source code to manage docstrings for
        """
        self.source_code = source_code
        self.tree = ast.parse(source_code)
        self.validator = DocstringValidator()
        logging.debug("DocStringManager initialized with validator.")

    @handle_exceptions(log_error)
    def insert_docstring(self, node: ast.FunctionDef, docstring: str) -> None:
        """
        Insert or update docstring with validation.

        Args:
            node (ast.FunctionDef): The function node to update
            docstring (str): The new docstring to insert
        """
        if not isinstance(docstring, str):
            log_error(f"Invalid docstring for function '{node.name}'. Expected a string.")
            return

        # Validate before insertion
        parsed_docstring, validation_errors = parse_and_validate_docstring(docstring)
        if parsed_docstring:
            logging.debug(f"Inserting validated docstring into function '{node.name}'.")
            node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
        else:
            log_error(
                f"Docstring validation failed for {node.name}: {validation_errors}"
            )

    @handle_exceptions(log_error)
    def update_source_code(self, documentation_entries: List[Dict]) -> str:
        """
        Update source code with validated docstrings.

        Args:
            documentation_entries (List[Dict]): List of documentation updates

        Returns:
            str: Updated source code
        """
        logging.debug("Updating source code with validated docstrings.")
        for entry in documentation_entries:
            for node in ast.walk(self.tree):
                if isinstance(node, ast.FunctionDef) and node.name == entry['function_name']:
                    # Validate docstring before updating
                    parsed_docstring, validation_errors = parse_and_validate_docstring(
                        entry['docstring']
                    )
                    if parsed_docstring:
                        self.insert_docstring(node, entry['docstring'])
                    else:
                        log_error(
                            f"Skipping invalid docstring for {node.name}: {validation_errors}"
                        )

        updated_code = ast.unparse(self.tree)
        logging.info("Source code updated with new docstrings.")
        return updated_code

    @handle_exceptions(log_error)
    def generate_markdown_documentation(
        self,
        documentation_entries: List[Dict],
        module_name: str = "",
        file_path: str = "",
        description: str = ""
    ) -> str:
        """
        Generate markdown documentation for the code.

        Args:
            documentation_entries (List[Dict]): List of documentation updates
            module_name (str): Name of the module
            file_path (str): Path to the source file 
            description (str): Module description

        Returns:
            str: Generated markdown documentation
        """
        logging.debug("Generating markdown documentation.")
        markdown_gen = MarkdownGenerator()
        if module_name:
            markdown_gen.add_header(f"Module: {module_name}")
        if description:
            markdown_gen.add_section("Description", description)
    
        for entry in documentation_entries:
            if 'function_name' in entry and 'docstring' in entry:
                markdown_gen.add_section(
                    f"Function: {entry['function_name']}",
                    entry['docstring'] or "No documentation available"
                )

        markdown = markdown_gen.generate_markdown()
        logging.info("Markdown documentation generated.")
        return markdown
    

class MarkdownGenerator:
    """
    Generates markdown documentation from Python code elements.

    Attributes:
        output (List[str]): List of markdown lines to be generated.

    Methods:
        add_header(text: str, level: int = 1) -> None: Adds a header to the markdown document.
        add_code_block(code: str, language: str = "python") -> None: Adds a code block to the markdown document.
        add_section(title: str, content: str, level: int = 3) -> None: Adds a section with title and content.
        generate_markdown() -> str: Generates the final markdown document.
    """

    def __init__(self):
        """Initialize the MarkdownGenerator."""
        self.output = []
        logging.debug("MarkdownGenerator initialized.")

    def add_header(self, text: str, level: int = 1) -> None:
        """
        Add a header to the markdown document.

        Args:
            text (str): Header text
            level (int): Header level (1-6)
        """
        logging.debug(f"Adding header: {text}")
        self.output.append(f"{'#' * level} {text}\n")

    def add_code_block(self, code: str, language: str = "python") -> None:
        """
        Add a code block to the markdown document.

        Args:
            code (str): The code to include
            language (str): Programming language for syntax highlighting
        """
        logging.debug("Adding code block.")
        self.output.append(f"```{language}\n{code}\n```\n")

    def add_section(self, title: str, content: str) -> None:
        """
        Add a section with title and content.

        Args:
            title (str): Section title
            content (str): Section content
        """
        logging.debug(f"Adding section: {title}")
        self.output.append(f"### {title}\n\n{content}\n")

    def generate_markdown(self) -> str:
        """
        Generate the final markdown document.

        Returns:
            str: Complete markdown document
        """
        logging.debug("Generating final markdown document.")
        return "\n".join(self.output)


class DocumentationManager:
    """
    Manages the overall documentation generation process.

    Attributes:
        output_dir (Path): Directory for output documentation.
        logger (logging.Logger): Logger instance for logging.

    Methods:
        process_file(file_path: Union[str, Path]) -> Optional[str]: Processes a single Python file for documentation.
        process_directory(directory_path: Union[str, Path]) -> Dict[str, str]: Processes all Python files in a directory.
        save_documentation(content: str, output_file: Union[str, Path]) -> bool: Saves documentation content to a file.
        generate_index(docs_map: Dict[str, str]) -> str: Generates an index page for all documentation files.
    """

    def __init__(self, output_dir: str = "docs"):
        """
        Initialize the DocumentationManager.

        Args:
            output_dir (str): Directory for output documentation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logging()
        logging.debug("DocumentationManager initialized.")

    def _setup_logging(self) -> logging.Logger:
        """
        Set up logging configuration.

        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger('documentation_manager')
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def process_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Process a single Python file for documentation.

        Args:
            file_path (Union[str, Path]): Path to the Python file

        Returns:
            Optional[str]: Generated markdown documentation
        """
        logging.debug(f"Processing file: {file_path}")
        try:
            file_path = Path(file_path)
            if not file_path.exists() or file_path.suffix != '.py':
                self.logger.error(f"Invalid Python file: {file_path}")
                return None

            with open(file_path, 'r') as f:
                source = f.read()

            module_doc = parse_docstring(source)
            
            markdown_gen = MarkdownGenerator()
            markdown_gen.add_header(f"Documentation for {file_path.name}")
            if module_doc:
                markdown_gen.add_section("Module Description", module_doc.get('Description', ''))

            # Parse the source code
            tree = ast.parse(source)
            
            # Process classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._process_class(node, markdown_gen)
                elif isinstance(node, ast.FunctionDef):
                    self._process_function(node, markdown_gen)

            markdown = markdown_gen.generate_markdown()
            logging.info(f"Generated markdown for file: {file_path}")
            return markdown

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return None

    def _process_class(self, node: ast.ClassDef, markdown_gen: MarkdownGenerator) -> None:
        """
        Process a class definition node.

        Args:
            node (ast.ClassDef): AST node representing a class definition
            markdown_gen (MarkdownGenerator): Markdown generator instance
        """
        logging.debug(f"Processing class: {node.name}")
        try:
            class_doc = ast.get_docstring(node) or "No documentation available"
            markdown_gen.add_section(f"Class: {node.name}", 
                                    class_doc)
            
            # Process class methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    self._process_function(item, markdown_gen, is_method=True, class_name=node.name)
        except Exception as e:
            self.logger.error(f"Error processing class {node.name}: {e}")

    def _process_function(self, node: ast.FunctionDef, markdown_gen: MarkdownGenerator, is_method: bool = False, class_name: str = None) -> None:
        """
        Process a function definition node.

        Args:
            node (ast.FunctionDef): AST node representing a function definition
            markdown_gen (MarkdownGenerator): Markdown generator instance
            is_method (bool): Whether the function is a class method
            class_name (str): Name of the containing class if is_method is True
        """
        logging.debug(f"Processing function: {node.name}")
        try:
            func_doc = ast.get_docstring(node) or "No documentation available"
            section_title = f"{'Method' if is_method else 'Function'}: {node.name}"
            if is_method:
                section_title = f"Method: {class_name}.{node.name}"

            # Extract function signature
            args = [arg.arg for arg in node.args.args]
            signature = f"{node.name}({', '.join(args)})"

            content = [
                f"```python\n{signature}\n```\n",
                func_doc
            ]
            
            markdown_gen.add_section(section_title, "\n".join(content))
        except Exception as e:
            self.logger.error(f"Error processing function {node.name}: {e}")

    def process_directory(self, directory_path: Union[str, Path]) -> Dict[str, str]:
        """
        Process all Python files in a directory for documentation.

        Args:
            directory_path (Union[str, Path]): Path to the directory to process

        Returns:
            Dict[str, str]: Dictionary mapping file paths to their documentation
        """
        logging.debug(f"Processing directory: {directory_path}")
        directory_path = Path(directory_path)
        results = {}

        if not directory_path.is_dir():
            self.logger.error(f"Invalid directory path: {directory_path}")
            return results

        for file_path in directory_path.rglob("*.py"):
            try:
                doc_content = self.process_file(file_path)
                if doc_content:
                    results[str(file_path)] = doc_content
            except Exception as e:
                self.logger.error(f"Error processing directory {directory_path}: {e}")

        return results

    def save_documentation(self, content: str, output_file: Union[str, Path]) -> bool:
        """
        Save documentation content to a file.

        Args:
            content (str): Documentation content to save
            output_file (Union[str, Path]): Path to the output file

        Returns:
            bool: True if successful, False otherwise
        """
        logging.debug(f"Saving documentation to: {output_file}")
        try:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Documentation saved to {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving documentation: {e}")
            return False

    def generate_index(self, docs_map: Dict[str, str]) -> str:
        """
        Generate an index page for all documentation files.

        Args:
            docs_map (Dict[str, str]): Dictionary mapping file paths to their documentation

        Returns:
            str: Generated index page content
        """
        logging.debug("Generating documentation index.")
        index_content = [
            "# Documentation Index\n",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## Files\n"
        ]

        for file_path in sorted(docs_map.keys()):
            rel_path = Path(file_path).name
            doc_path = Path(file_path).with_suffix('.md').name
            index_content.append(f"- [{rel_path}]({doc_path})")

        logging.info("Documentation index generated.")
        return "\n".join(index_content)
