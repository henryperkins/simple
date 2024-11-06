import os
import ast
import shutil
import subprocess
import sentry_sdk
import aiofiles
from typing import Any, Dict, List, Union
from tqdm import tqdm
import asyncio

from extract.code import extract_classes_and_functions_from_ast
from api_interaction import analyze_function_with_openai
from logging_utils import setup_logger

# Initialize a logger for this module
logger = setup_logger("files")

async def clone_repo(repo_url: str, clone_dir: str) -> None:
    """Clone a GitHub repository into a specified directory."""
    logger.info("Cloning repository %s into %s", repo_url, clone_dir)
    remove_existing_directory(clone_dir)
    try:
        execute_git_clone(repo_url, clone_dir)
        set_directory_permissions(clone_dir)
        logger.info("Repository cloned successfully.")
    except Exception as e:
        logger.error("Failed to clone repository: %s", e)
        sentry_sdk.capture_exception(e)
        raise

def remove_existing_directory(clone_dir: str) -> None:
    """Remove an existing directory if it exists."""
    if os.path.exists(clone_dir):
        try:
            shutil.rmtree(clone_dir)
            logger.info("Removed existing directory: %s", clone_dir)
        except OSError as e:
            logger.error("Failed to remove directory %s: %s", clone_dir, e)
            raise

def execute_git_clone(repo_url: str, clone_dir: str) -> None:
    """Execute the git clone command."""
    try:
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', repo_url, clone_dir],
            capture_output=True,
            text=True,
            timeout=60,
            check=True
        )
        if result.stderr:
            logger.warning("Git clone stderr output: %s", result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"Git clone failed with return code {e.returncode}: {e.stderr}")
        remove_existing_directory(clone_dir)
        raise
    except subprocess.TimeoutExpired:
        logger.error("Git clone operation timed out.")
        remove_existing_directory(clone_dir)
        raise
    except OSError as e:
        logger.error(f"OS error during git clone: {e}")
        remove_existing_directory(clone_dir)
        raise

def set_directory_permissions(clone_dir: str) -> None:
    """Set directory permissions for cloned files."""
    try:
        for root, dirs, files in os.walk(clone_dir):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o755)
            for f in files:
                os.chmod(os.path.join(root, f), 0o644)
        logger.debug("Set directory permissions for %s", clone_dir)
    except Exception as e:
        logger.error(f"Error setting permissions in {clone_dir}: {e}")
        raise

def load_gitignore_patterns(repo_dir: str) -> List[str]:
    """Load .gitignore patterns from the repository directory."""
    gitignore_path = os.path.join(repo_dir, '.gitignore')
    try:
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            logger.info("Loaded %d patterns from .gitignore", len(patterns))
            return patterns
        else:
            logger.info(".gitignore file not found.")
            return []
    except OSError as e:
        logger.error(f"Error reading .gitignore file: {e}")
        return []

def get_all_files(directory: str, exclude_patterns: List[str] | None = None) -> List[str]:
    """Retrieve all Python files in the directory, excluding patterns."""
    exclude_patterns = exclude_patterns if exclude_patterns is not None else []
    python_files: List[str] = []

    try:
        with tqdm(desc="🔍 Discovering Python files", unit="dir") as pbar:
            for root, dirs, files in os.walk(directory):
                pbar.update(1)
                dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
                for file in files:
                    if file.endswith('.py') and not any(pattern in file for pattern in exclude_patterns):
                        filepath = os.path.join(root, file)
                        python_files.append(filepath)
        logger.info("Found %d Python files", len(python_files))
    except Exception as e:
        logger.error(f"Error discovering files in {directory}: {e}")
        sentry_sdk.capture_exception(e)
    return python_files

async def process_file(filepath: str, service: str) -> Dict[str, Any]:
    """Read and parse a Python file."""
    try:
        content = await read_file_content(filepath)
        logger.debug(f"Read {len(content)} characters from {filepath}")
        tree = ast.parse(content)
        extracted_data = extract_classes_and_functions_from_ast(tree, content)
        logger.info("Successfully extracted data for %s", filepath)
        return extracted_data
    except FileNotFoundError:
        logger.error("File not found: %s", filepath)
        sentry_sdk.capture_message(f"File not found: {filepath}")
        return {"error": "File not found", "file_content": [{"content": ""}]}
    except UnicodeDecodeError:
        logger.error("Unicode decode error in file %s", filepath)
        sentry_sdk.capture_message(f"Unicode decode error in file {filepath}")
        return {"error": "Unicode decode error", "file_content": [{"content": ""}]}
    except SyntaxError as e:
        logger.error("Syntax error in file %s: %s", filepath, e)
        sentry_sdk.capture_exception(e)
        return {"error": "Syntax error", "file_content": [{"content": ""}]}
    except Exception as e:
        logger.error("Unexpected error processing file %s: %s", filepath, e)
        sentry_sdk.capture_exception(e)
        return {"error": "Unexpected error", "file_content": [{"content": ""}]}

async def read_file_content(filepath: str) -> str:
    """Read the content of a file asynchronously."""
    try:
        async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
            content = await f.read()
        logger.debug(f"Read file content from {filepath}")
        return content
    except FileNotFoundError as e:
        logger.error(f"File not found: {filepath}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Unicode decode error in file {filepath}: {e}")
        raise
    except OSError as e:
        logger.error(f"OS error reading file {filepath}: {e}")
        raise

async def analyze_and_update_functions(extracted_data: Dict[str, Any], tree: ast.AST, content: str, service: str) -> str:
    """Analyze functions and update their docstrings."""
    file_content = content
    for function in extracted_data.get('functions', []):
        try:
            analysis = await analyze_function_with_openai(function, service)
            function.update(analysis)
            file_content = update_function_docstring(file_content, tree, function, analysis)
            logger.info(f"Updated docstring for function {function['name']}")
        except Exception as e:
            logger.error(f"Error analyzing function {function.get('name', 'unknown')}: {e}")
            sentry_sdk.capture_exception(e)
    logger.debug("Updated function docstrings")
    return file_content

def update_function_docstring(file_content: str, tree: ast.AST, function: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    """Update the docstring of a function in the file content."""
    try:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function['name']:
                docstring = analysis.get('docstring')
                if docstring:
                    return insert_docstring(file_content, node, docstring)
        logger.warning(f"Function {function['name']} not found in AST.")
    except Exception as e:
        logger.error(f"Error updating docstring for function {function['name']}: {e}")
    return file_content

def insert_docstring(
    source: str,
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    docstring: str
) -> str:
    """Insert or replace a docstring in a function or class definition."""
    try:
        if not hasattr(node, 'body') or not node.body:
            logger.warning(f"No body found for node {getattr(node, 'name', 'unknown')}.")
            return source

        # Determine the start of the body
        body_start = node.body[0].lineno - 1  # Adjust for 0-based index
        lines = source.splitlines()
        def_line = lines[node.lineno - 1]
        indent = " " * (len(def_line) - len(def_line.lstrip()))

        # Check if there's an existing docstring
        if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            # Replace existing docstring
            docstring_start = node.body[0].lineno - 1
            docstring_end = node.body[0].end_lineno if hasattr(node.body[0], 'end_lineno') else docstring_start
            lines = lines[:docstring_start] + lines[docstring_end:]
            body_start = docstring_start

        # Prepare the new docstring lines
        docstring_lines = [
            f'{indent}"""',
            *[f"{indent}{line}" for line in docstring.split('\n')],
            f'{indent}"""'
        ]

        # Insert the new docstring
        updated_lines = lines[:body_start] + docstring_lines + lines[body_start:]
        logger.debug(f"Inserted docstring for {node.name}")
        return '\n'.join(updated_lines)
    except Exception as e:
        logger.error(f"Error inserting docstring: {e}")
        sentry_sdk.capture_exception(e)
        return source