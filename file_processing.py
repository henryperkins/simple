import os
import ast
import aiofiles
import asyncio
import aiohttp
import logging
import shutil
import subprocess
import sentry_sdk
from typing import Any, Dict, List, Tuple
from pathlib import Path
from code_extraction import CodeExtractor
from cache import initialize_cache

def clone_repo(repo_url: str, clone_dir: str) -> None:
    """Clone a GitHub repository into a specified directory."""
    try:
        subprocess.run(['git', 'clone', repo_url, clone_dir], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"Cloned repository {repo_url} into {clone_dir}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error cloning repository: {e.stderr.decode('utf-8')}")
        sentry_sdk.capture_exception(e)
        raise

def load_gitignore_patterns(repo_dir: str) -> List[str]:
    """Load .gitignore patterns from the repository directory."""
    gitignore_path = os.path.join(repo_dir, '.gitignore')
    patterns = []
    if os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, 'r') as f:
                patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            logging.info(f"Loaded gitignore patterns from {gitignore_path}")
        except Exception as e:
            logging.warning(f"Failed to load .gitignore: {e}")
            sentry_sdk.capture_exception(e)
    return patterns

def get_all_files(directory: str, exclude_dirs: List[str] = None) -> List[str]:
    """Retrieve all Python files in the directory, excluding specified directories."""
    if exclude_dirs is None:
        exclude_dirs = []
    files_list = []
    for root, dirs, files in os.walk(directory, topdown=True):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith(".py"):
                files_list.append(os.path.join(root, file))
    logging.info(f"Found {len(files_list)} Python files in {directory}")
    return files_list

def format_with_black(file_content: str) -> Tuple[bool, str]:
    """Attempt to format code using black."""
    try:
        import black
        mode = black.FileMode()
        formatted_content = black.format_file_contents(
            file_content,
            fast=False,
            mode=mode
        )
        logging.info("Formatted code with Black")
        return True, formatted_content
    except ImportError:
        logging.warning("Black package not found. Run: pip install black")
        return False, file_content
    except Exception as e:
        logging.warning(f"Black formatting failed: {str(e)}")
        return False, file_content

def create_complexity_indicator(complexity: int) -> str:
    """Create a visual indicator for code complexity."""
    if complexity is None:
        return "❓"
    elif complexity <= 5:
        return "🟢"  # Low complexity
    elif complexity <= 10:
        return "🟡"  # Medium complexity
    else:
        return "🔴"  # High complexity

async def process_file(filepath: str) -> Tuple[str, Dict[str, Any]]:
    """Read and parse a Python file, extracting classes and functions."""
    try:
        async with aiofiles.open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = await f.read()
        logging.info(f"Processing file: {filepath}")
        extractor = CodeExtractor()
        try:
            tree = ast.parse(content)
            add_parent_info(tree)
            extracted_data = extractor.extract_classes_and_functions_from_ast(tree, content)
            logging.info(f"Successfully extracted classes and functions from {filepath}")
        except (IndentationError, SyntaxError) as e:
            logging.warning(f"Initial parsing failed for {filepath}, attempting black formatting: {e}")
            success, formatted_content = format_with_black(content)
            if success:
                try:
                    tree = ast.parse(formatted_content)
                    add_parent_info(tree)
                    extracted_data = extractor.extract_classes_and_functions_from_ast(tree, formatted_content)
                    content = formatted_content
                    logging.info(f"Successfully extracted classes and functions from {filepath} after black formatting")
                except (IndentationError, SyntaxError) as e:
                    logging.error(f"Parsing failed even after black formatting for {filepath}: {e}")
                    extracted_data = {"functions": [], "classes": []}
            else:
                logging.error(f"Black formatting failed for {filepath}")
                extracted_data = {"functions": [], "classes": []}
        extracted_data['file_content'] = content
        return filepath, extracted_data
    except Exception as e:
        logging.error(f"Error processing file {filepath}: {e}")
        sentry_sdk.capture_exception(e)
        return filepath, {"functions": [], "classes": [], "error": str(e)}

def write_analysis_to_markdown(results: Dict[str, Dict[str, Any]], output_file_path: str, repo_dir: str) -> None:
    """Write the analysis results to a markdown file with improved readability."""
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
            
            # Write details for each file
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
                    md_file.write("|-----------------------|-------------------|----------|\n")
                    
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
                source_code = analysis.get("file_content", "# No source code available")
                md_file.write(f"{source_code}\n")
                md_file.write("```\n\n")
    except Exception as e:
        logging.error(f"Error writing markdown file: {str(e)}")
        sentry_sdk.capture_exception(e)