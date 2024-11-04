# file_processing.py
import os
import ast
import aiofiles
import asyncio
import aiohttp
import random
import logging
import shutil
import subprocess
import sentry_sdk
from code_extraction import extract_classes_and_functions_from_ast

def clone_repo(repo_url, clone_dir):
    """Clone a GitHub repository into a specified directory.

    Args:
        repo_url (str): The URL of the GitHub repository to clone.
        clone_dir (str): The directory where the repository will be cloned.

    Raises:
        subprocess.CalledProcessError: If the cloning process fails.
    """
    try:
        if os.path.exists(clone_dir):
            logging.info(f"Removing existing directory: {clone_dir}")
            shutil.rmtree(clone_dir)

        logging.info(f"Cloning repository {repo_url} into {clone_dir}")
        subprocess.run(["git", "clone", repo_url, clone_dir], check=True)
        logging.info(f"Cloned repository from {repo_url} into {clone_dir}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to clone repository: {e}")
        sentry_sdk.capture_exception(e)
        raise

def get_all_files(directory, exclude_dirs=None):
    """Retrieve all Python files in the directory, excluding specified directories.

    Args:
        directory (str): The root directory to search for Python files.
        exclude_dirs (list, optional): A list of directories to exclude from the search.

    Returns:
        list: A list of file paths to Python files.
    """
    if exclude_dirs is None:
        exclude_dirs = []
    files_list = []
    for root, dirs, files in os.walk(directory, topdown=True):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith(".py"):
                files_list.append(os.path.join(root, file))
    return files_list

async def process_file(filepath):
    """Read and parse a Python file, extracting classes and functions.

    Args:
        filepath (str): The path to the Python file to process.

    Returns:
        tuple: A tuple containing the file path and a dictionary with extracted data.

    Raises:
        Exception: If an error occurs during file processing.
    """
    try:
        with sentry_sdk.start_span(op="process_file", description=filepath):
            async with aiofiles.open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = await f.read()

            logging.info(f"Processing file: {filepath}")

            try:
                tree = ast.parse(content)
                add_parent_info(tree)
                extracted_data = extract_classes_and_functions_from_ast(tree, content)
            except (IndentationError, SyntaxError) as e:
                logging.warning(f"Initial parsing failed for {filepath}, attempting black formatting: {e}")
                success, formatted_content = format_with_black(content)
                if success:
                    try:
                        tree = ast.parse(formatted_content)
                        add_parent_info(tree)
                        extracted_data = extract_classes_and_functions_from_ast(tree, formatted_content)
                        content = formatted_content
                    except (IndentationError, SyntaxError) as e:
                        logging.error(f"Parsing failed even after black formatting for {filepath}: {e}")
                        extracted_data = {"functions": [], "classes": []}
                else:
                    extracted_data = {"functions": [], "classes": []}

            extracted_data['file_content'] = content
            return filepath, extracted_data

    except Exception as e:
        logging.error(f"Error processing file {filepath}: {e}")
        sentry_sdk.capture_exception(e)
        return filepath, {"functions": [], "classes": [], "error": str(e)}

# file_processing.py
import os
import ast
import logging
import asyncio
import random
import aiohttp
from typing import Any, Callable, Coroutine

def add_parent_info(node, parent=None):
    """Add parent links to AST nodes."""
    for child in ast.iter_child_nodes(node):
        child.parent = node
        add_parent_info(child, node)

def format_with_black(file_content):
    """Attempt to format code using black.
    
    Args:
        file_content (str): The source code content to format.
        
    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if formatting succeeded, False otherwise
            - str: The formatted content if successful, original content if failed
    """
    try:
        import black
        mode = black.FileMode()
        formatted_content = black.format_file_contents(
            file_content, 
            fast=False,
            mode=mode
        )
        return True, formatted_content
        
    except ImportError:
        logging.warning("Black package not found. Run: pip install black")
        return False, file_content
        
    except Exception as e:
        logging.warning(f"Black formatting failed: {str(e)}")
        return False, file_content

async def exponential_backoff_with_jitter(func, max_retries=5, base_delay=1, max_delay=60):
    """Execute a coroutine with exponential backoff and jitter for handling rate limits/failures.
    
    Args:
        func (Coroutine): Async function to retry
        max_retries (int, optional): Maximum retry attempts. Defaults to 5.
        base_delay (int, optional): Initial delay between retries in seconds. Defaults to 1.
        max_delay (int, optional): Maximum delay between retries in seconds. Defaults to 60.
        
    Returns:
        Any: Result from the coroutine if successful
        
    Raises:
        Exception: If max retries exceeded or unrecoverable error occurs
    """
    retries = 0
    while retries < max_retries:
        try:
            return await func()
            
        except aiohttp.ClientResponseError as e:
            if e.status == 429:  # Rate limit
                retry_after = 5  # Default retry time
                try:
                    if 'retry after' in e.message.lower():
                        retry_after = int(e.message.split('retry after ')[1].split()[0])
                except (AttributeError, IndexError, ValueError):
                    pass
                
                logging.warning(f"Rate limit hit. Retrying in {retry_after}s...")
                await asyncio.sleep(retry_after)
                retries += 1
                continue
                
            logging.error(f"API client error: {e}")
            raise
            
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                logging.error(f"Max retries ({max_retries}) exceeded")
                raise
                
            delay = min(base_delay * (2 ** (retries - 1)), max_delay)
            jitter = random.uniform(0, 0.1 * delay)
            total_delay = delay + jitter
            
            logging.warning(
                f"Attempt {retries}/{max_retries} failed: {str(e)}. "
                f"Retrying in {total_delay:.2f}s"
            )
            
            await asyncio.sleep(total_delay)
            
    raise Exception(f"Failed after {max_retries} retries")

def write_analysis_to_markdown(results, output_file_path, repo_dir):
    """Write analysis results to a markdown file with consistent formatting.
    
    Args:
        results (dict): Analysis results containing file summaries and function details
        output_file_path (str): Path to output markdown file
        repo_dir (str): Root directory of repository being analyzed
    """
    try:
        with open(output_file_path, "w", encoding="utf-8") as md_file:
            # Write header and TOC
            md_file.write("# Code Documentation and Analysis\n\n")
            md_file.write("## Table of Contents\n\n")
            
            # Generate TOC entries
            for filepath in results.keys():
                relative_path = os.path.relpath(filepath, repo_dir)
                anchor = relative_path.replace('/', '-').replace('.', '-').replace(' ', '-')
                md_file.write(f"- [{relative_path}](#{anchor})\n")
            md_file.write("\n")

            # Process each file
            for filepath, analysis in results.items():
                relative_path = os.path.relpath(filepath, repo_dir)
                anchor = relative_path.replace('/', '-').replace('.', '-').replace(' ', '-')
                
                # File header
                md_file.write(f"## {relative_path}\n\n")

                # Summary section - AI generated overview
                md_file.write("### Summary\n\n")
                file_summary = analysis.get("file_analysis", {}).get("summary", "No summary available.")
                md_file.write(f"{file_summary}\n\n")

                # Recent Changes section
                md_file.write("### Recent Changes\n\n")
                changelog = analysis.get("file_analysis", {}).get("changelog", "No recent changes.")
                md_file.write(f"{changelog}\n\n")

                # Function Analysis table
                md_file.write("### Function Analysis\n\n")
                if analysis.get("functions"):
                    md_file.write("| Function/Class/Method | Complexity Score | Summary |\n")
                    md_file.write("|---------------------|-----------------|----------|\n")
                    
                    for func_analysis in analysis.get("functions", []):
                        if not func_analysis:
                            continue
                        
                        name = func_analysis.get('name', 'Unknown')
                        complexity = func_analysis.get('complexity_score', None)
                        complexity_indicator = create_complexity_indicator(complexity)
                        summary = func_analysis.get('summary', 'No documentation available.').replace("\n", " ")
                        
                        md_file.write(f"| **{name}** | {complexity} {complexity_indicator} | {summary} |\n")
                    md_file.write("\n")
                else:
                    md_file.write("No functions analyzed.\n\n")

                # Source code with docstrings
                md_file.write("### Source Code\n\n")
                md_file.write("```python\n")
                source_code = analysis.get("source_code", "# No source code available")
                md_file.write(f"{source_code}\n")
                md_file.write("```\n\n")

    except Exception as e:
        error_msg = f"Error writing markdown file: {str(e)}"
        logging.error(error_msg)
        sentry_sdk.capture_exception(e)
        raise
    
def create_complexity_indicator(complexity):
    """Create a visual indicator based on the complexity score.

    Args:
        complexity (int): The complexity score to evaluate.

    Returns:
        str: A visual indicator representing the complexity level.
    """
    if complexity is None:
        return "❓"

    try:
        complexity = float(complexity)
    except (ValueError, TypeError):
        return "❓"

    if complexity < 3:
        return "🟢 Low"
    elif complexity < 6:
        return "🟡 Medium"
    elif complexity < 8:
        return "🟠 High"
    else:
        return "🔴 Very High"