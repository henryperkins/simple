import os
import ast
import logging
import shutil
import subprocess
import sentry_sdk
import aiofiles
from typing import Any, Dict, List, Tuple, Union
from tqdm import tqdm

from code_extraction import CodeExtractor, add_parent_info
from api_interaction import analyze_function_with_openai

logger = logging.getLogger(__name__)


async def clone_repo(repo_url: str, clone_dir: str) -> None:
    """
    Clone a GitHub repository into a specified directory.
    Removes existing directory if it exists.
    """
    logger.info("Cloning repository %s into %s", repo_url, clone_dir)
    
    # Remove existing directory if it exists
    if os.path.exists(clone_dir):
        try:
            shutil.rmtree(clone_dir)
            logger.info("Removed existing directory: %s", clone_dir)
        except OSError as e:
            logger.error("Failed to remove existing directory %s: %s", clone_dir, e)
            raise

    try:
        # Use subprocess with timeout to prevent hanging
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', repo_url, clone_dir],
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout
            check=True
        )
            
        if result.stderr:
            logger.warning("Git clone stderr output: %s", result.stderr)
            
        logger.info("Successfully cloned repository into %s", clone_dir)
        
        # Fix file permissions
        for root, dirs, files in os.walk(clone_dir):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o755)
            for f in files:
                os.chmod(os.path.join(root, f), 0o644)
                
    except subprocess.TimeoutExpired:
        logger.error("Git clone operation timed out")
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)
        raise
    except subprocess.CalledProcessError as e:
        logger.error("Git clone failed: %s", e.stderr)
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)
        raise
    except OSError as e:
        logger.error("Unexpected error during clone: %s", str(e))
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)
        raise

def insert_docstring(source: str, node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef], docstring: str) -> str:
    """Insert a docstring into a function or class definition."""
    try:
        # Get the line number where the body starts
        if not hasattr(node, 'body') or not node.body:
            return source
            
        body_start = node.body[0].lineno if node.body else node.lineno
        
        # Split the source into lines
        lines = source.splitlines()
        
        # Get indentation from the definition line
        def_line = lines[node.lineno - 1]
        indent = " " * (len(def_line) - len(def_line.lstrip()))
        
        # Format the docstring with proper indentation
        docstring_lines = [f'{indent}"""'] + [f"{indent}{line}" for line in docstring.split('\n')] + [f'{indent}"""']
        
        # Insert the docstring after the definition line
        result = lines[:body_start] + docstring_lines + lines[body_start:]
        return '\n'.join(result)
        
    except Exception as e:
        logger.error(f"Error inserting docstring: {e}")
        return source

def load_gitignore_patterns(repo_dir: str) -> List[str]:
    """Load .gitignore patterns from the repository directory."""
    gitignore_path = os.path.join(repo_dir, '.gitignore')
    patterns: List[str] = []
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        logger.info("Loaded %d patterns from .gitignore", len(patterns))
    else:
        logger.info(".gitignore file not found.")
    return patterns

def get_all_files(directory: str, exclude_dirs: List[str] | None = None) -> List[str]:
    """Retrieve all Python files in the directory, excluding specified directories."""
    exclude_dirs = exclude_dirs if exclude_dirs is not None else []
    python_files: List[str] = []
    
    # Create progress bar for file discovery
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

def format_with_black(file_content: str) -> Tuple[bool, str]:
    """Attempt to format code using black."""
    try:
        process = subprocess.run(
            ['black', '-'],
            input=file_content.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        formatted_content = process.stdout.decode('utf-8')
        logger.info("Successfully formatted code with black.")
        return True, formatted_content
    except subprocess.CalledProcessError as e:
        logger.warning("Black formatting failed: %s", e.stderr.decode('utf-8'))
        return False, file_content
    except OSError as e:
        logger.error("Exception during black formatting: %s", e)
        return False, file_content

def create_complexity_indicator(complexity: int) -> str:
    """Create a visual indicator for code complexity."""
    if complexity <= 5:
        return "🟢"
    elif complexity <= 10:
        return "🟡"
    else:
        return "🔴"

async def process_file(filepath: str, service: str) -> Dict[str, Any]:
    """Read and parse a Python file, extract classes and functions, and analyze them."""
    try:
        async with aiofiles.open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = await f.read()
        
        # Parse the code
        tree = ast.parse(content)
        add_parent_info(tree)
        
        # Extract data
        extractor = CodeExtractor()
        extracted_data = extractor.extract_classes_and_functions_from_ast(tree, content)
        
        # Store original content
        file_content = content
        
        # Analyze functions and insert docstrings
        for function in extracted_data.get('functions', []):
            analysis = await analyze_function_with_openai(function, service)
            function.update(analysis)
            
            # Find the function node in the AST
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function['name']:
                    # Insert the docstring
                    docstring = analysis.get('docstring')
                    if docstring:
                        file_content = insert_docstring(file_content, node, docstring)
                    break
        
        # Store the updated content
        extracted_data['file_content'] = [{"content": file_content}]
        logger.info("Successfully extracted data and inserted docstrings for %s", filepath)
        
        return extracted_data
    except (OSError, SyntaxError) as e:
        logger.error("Error processing file %s: %s", filepath, e)
        sentry_sdk.capture_exception(e)
        return {}

def write_analysis_to_markdown(results: Dict[str, Dict[str, Any]], output_file_path: str, repo_dir: str) -> None:
    """Write the analysis results to a markdown file with improved readability."""
    logger.info("Writing analysis results to %s", output_file_path)
    try:
        with open(output_file_path, 'w', encoding='utf-8') as md_file:
            # Write Table of Contents
            md_file.write("# Code Analysis Report\n\n")
            md_file.write("## Table of Contents\n\n")
            for filepath in results:
                relative_path = os.path.relpath(filepath, repo_dir)
                anchor = relative_path.replace('/', '-').replace('.', '-').replace(' ', '-')
                md_file.write(f"- [{relative_path}](#{anchor})\n")
            md_file.write("\n---\n\n")
            
            # Create progress bar for file documentation
            with tqdm(total=len(results), desc="📝 Generating documentation", unit="file") as pbar:
                # Write details for each file
                for filepath, analysis in results.items():
                    relative_path = os.path.relpath(filepath, repo_dir)
                    anchor = relative_path.replace('/', '-').replace('.', '-').replace(' ', '-')
                    
                    # File header
                    md_file.write(f"# {relative_path}\n\n")
                    
                    # Function Analysis table
                    md_file.write("### Function Analysis\n\n")
                    if analysis.get("functions"):
                        md_file.write("| Function/Class/Method | Complexity Score | Summary |\n")
                        md_file.write("|-----------------------|-------------------|----------|\n")
                        
                        for func_analysis in analysis.get("functions", []):
                            if not func_analysis:
                                continue
                            name = func_analysis.get('name', 'Unknown')
                            complexity = func_analysis.get('complexity_score', None)
                            complexity_indicator = create_complexity_indicator(complexity) if isinstance(complexity, int) else ""
                            summary = func_analysis.get('summary', 'No documentation available.').replace("\n", " ")
                            md_file.write(f"| **{name}** | {complexity} {complexity_indicator} | {summary} |\n")
                        md_file.write("\n")
                    else:
                        md_file.write("No functions analyzed.\n\n")
                    
                    # Source Code section
                    md_file.write("### Source Code\n\n")
                    md_file.write("```python\n")
                    # Extract the actual content from the file_content structure
                    file_content = analysis.get('file_content', [{}])[0].get('content', '')
                    md_file.write(file_content)
                    md_file.write("\n```\n\n")
                    
                    # Update progress bar
                    pbar.update(1)
                
        logger.info("Successfully wrote analysis to markdown.")
    except OSError as e:
        logger.error("Error writing markdown file: %s", str(e))
        sentry_sdk.capture_exception(e)

async def process_files_concurrently(files_list: List[str], service: str) -> Dict[str, Dict[str, Any]]:
    """Process multiple files concurrently with progress bar."""
    logger.info("Starting to process %d files", len(files_list))
    processed_results = {}
    
    # Create progress bar for file processing
    with tqdm(total=len(files_list), desc="⚙️ Processing files", unit="file") as pbar:
        for filepath in files_list:
            try:
                result = await process_file(filepath, service)
                if result:  # Only add successful results
                    processed_results[filepath] = result
            except Exception as e:
                logger.error("Error processing file %s: %s", filepath, e)
            finally:
                pbar.update(1)
    
    logger.info("Completed processing files.")
    return processed_results

async def analyze_functions_concurrently(results: Dict[str, Dict[str, Any]], service: str) -> None:
    """Analyze multiple functions concurrently with progress bar."""
    logger.info("Starting function analysis using %s service", service)
    
    # Count total functions to analyze
    total_functions = sum(
        len(analysis.get("functions", [])) 
        for analysis in results.values()
    )
    
    # Create progress bar for function analysis
    with tqdm(total=total_functions, desc="🧮 Analyzing functions", unit="func") as pbar:
        for analysis in results.values():
            for func in analysis.get("functions", []):
                if func.get("name"):
                    try:
                        func_analysis = await analyze_function_with_openai(func, service)
                        func.update(func_analysis)
                    except Exception as e:
                        logger.error("Error analyzing function: %s", e)
                    finally:
                        pbar.update(1)
    
    logger.info("Completed function analysis.")
