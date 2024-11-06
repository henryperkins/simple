import os
import ast
import shutil
import subprocess
import sentry_sdk
import aiofiles
from typing import Any, Dict, List, Union
from tqdm import tqdm

from extract.code import extract_classes_and_functions_from_ast
from api_interaction import analyze_function_with_openai
from logging_utils import setup_logger

# Initialize a logger for this module
logger = setup_logger("file")

async def clone_repo(repo_url: str, clone_dir: str) -> None:
    """Clone a GitHub repository into a specified directory."""
    logger.info("Cloning repository %s into %s", repo_url, clone_dir)
    remove_existing_directory(clone_dir)
    execute_git_clone(repo_url, clone_dir)
    set_directory_permissions(clone_dir)

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
        logger.info("Successfully cloned repository into %s", clone_dir)
    except subprocess.TimeoutExpired:
        logger.error("Git clone operation timed out")
        remove_existing_directory(clone_dir)
        raise
    except subprocess.CalledProcessError as e:
        logger.error("Git clone failed: %s", e.stderr)
        remove_existing_directory(clone_dir)
        raise
    except OSError as e:
        logger.error("Unexpected error during clone: %s", str(e))
        remove_existing_directory(clone_dir)
        raise

def set_directory_permissions(clone_dir: str) -> None:
    """Set directory permissions for cloned files."""
    for root, dirs, files in os.walk(clone_dir):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o755)
        for f in files:
            os.chmod(os.path.join(root, f), 0o644)
    logger.debug("Set directory permissions for %s", clone_dir)

def load_gitignore_patterns(repo_dir: str) -> List[str]:
    """Load .gitignore patterns from the repository directory."""
    gitignore_path = os.path.join(repo_dir, '.gitignore')
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        logger.info("Loaded %d patterns from .gitignore", len(patterns))
        return patterns
    logger.info(".gitignore file not found.")
    return []

def get_all_files(directory: str, exclude_dirs: List[str] | None = None) -> List[str]:
    """Retrieve all Python files in the directory."""
    exclude_dirs = exclude_dirs if exclude_dirs is not None else []
    python_files: List[str] = []
    
    with tqdm(desc="🔍 Discovering Python files", unit="dir") as pbar:
        for root, dirs, files in os.walk(directory):
            pbar.update(1)
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    python_files.append(filepath)
                    
    logger.info("Found %d Python files", len(python_files))
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
    except Exception as e:
        logger.error("Error processing file %s: %s", filepath, str(e))
        sentry_sdk.capture_exception(e)
        return {"file_content": [{"content": ""}]}

async def read_file_content(filepath: str) -> str:
    """Read the content of a file asynchronously."""
    async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
        content = await f.read()
    logger.debug(f"Read file content from {filepath}")
    return content

async def analyze_and_update_functions(extracted_data: Dict[str, Any], tree: ast.AST, content: str, service: str) -> str:
    """Analyze functions and update their docstrings."""
    file_content = content
    for function in extracted_data.get('functions', []):
        analysis = await analyze_function_with_openai(function, service)
        function.update(analysis)
        file_content = update_function_docstring(file_content, tree, function, analysis)
    logger.debug("Updated function docstrings")
    return file_content

def update_function_docstring(file_content: str, tree: ast.AST, function: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    """Update the docstring of a function in the file content."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function['name']:
            docstring = analysis.get('docstring')
            if docstring:
                return insert_docstring(file_content, node, docstring)
    return file_content

def insert_docstring(
    source: str, 
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef], 
    docstring: str
) -> str:
    """Insert a docstring into a function or class definition."""
    try:
        if not hasattr(node, 'body') or not node.body:
            return source
            
        body_start = node.body[0].lineno if node.body else node.lineno
        lines = source.splitlines()
        def_line = lines[node.lineno - 1]
        indent = " " * (len(def_line) - len(def_line.lstrip()))
        
        docstring_lines = [
            f'{indent}"""',
            *[f"{indent}{line}" for line in docstring.split('\n')],
            f'{indent}"""'
        ]
        
        result = lines[:body_start] + docstring_lines + lines[body_start:]
        logger.debug(f"Inserted docstring for {node.name}")
        return '\n'.join(result)
        
    except Exception as e:
        logger.error(f"Error inserting docstring: {e}")
        return source
