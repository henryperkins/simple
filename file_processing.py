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
from code_extraction import CodeExtractor, add_parent_info
from cache import cache_response, get_cached_response
from config import Config
import hashlib

logger = logging.getLogger(__name__)

async def clone_repo(repo_url: str, clone_dir: str) -> None:
    """
    Clone a GitHub repository into a specified directory.
    Removes existing directory if it exists.
    """
    logger.info(f"Cloning repository {repo_url} into {clone_dir}")
    
    # Remove existing directory if it exists
    if os.path.exists(clone_dir):
        try:
            shutil.rmtree(clone_dir)
            logger.info(f"Removed existing directory: {clone_dir}")
        except Exception as e:
            logger.error(f"Failed to remove existing directory {clone_dir}: {e}")
            raise

    try:
        # Use subprocess with timeout to prevent hanging
        process = subprocess.run(
            ['git', 'clone', '--depth', '1', repo_url, clone_dir],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode,
                process.args,
                process.stdout,
                process.stderr
            )
            
        logger.info(f"Successfully cloned repository into {clone_dir}")
        
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
        logger.error(f"Git clone failed: {e.stderr}")
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during clone: {str(e)}")
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)
        raise

def load_gitignore_patterns(repo_dir: str) -> List[str]:
    """Load .gitignore patterns from the repository directory."""
    gitignore_path = os.path.join(repo_dir, '.gitignore')
    patterns = []
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        logger.info(f"Loaded {len(patterns)} patterns from .gitignore.")
    else:
        logger.info(".gitignore file not found.")
    return patterns

def get_all_files(directory: str, exclude_dirs: List[str] = None) -> List[str]:
    """Retrieve all Python files in the directory, excluding specified directories."""
    if exclude_dirs is None:
        exclude_dirs = []
    python_files = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                python_files.append(filepath)
    logger.info(f"Found {len(python_files)} Python files.")
    return python_files

def format_with_black(file_content: str) -> Tuple[bool, str]:
    """Attempt to format code using black."""
    try:
        process = subprocess.run(
            ['black', '-'],
            input=file_content.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if process.returncode == 0:
            formatted_content = process.stdout.decode('utf-8')
            logger.info("Successfully formatted code with black.")
            return True, formatted_content
        else:
            logger.warning(f"Black formatting failed: {process.stderr.decode('utf-8')}")
            return False, file_content
    except Exception as e:
        logger.error(f"Exception during black formatting: {e}")
        return False, file_content

def create_complexity_indicator(complexity: int) -> str:
    """Create a visual indicator for code complexity."""
    if complexity <= 5:
        return "🟢"
    elif complexity <= 10:
        return "🟡"
    else:
        return "🔴"

async def process_file(filepath: str) -> Tuple[str, Dict[str, Any]]:
    """Read and parse a Python file, extracting classes and functions."""
    try:
        async with aiofiles.open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = await f.read()
        extractor = CodeExtractor()
        tree = ast.parse(content)
        add_parent_info(tree)
        extracted_data = extractor.extract_classes_and_functions_from_ast(tree, content)
        extracted_data['file_content'] = [{"content": content}]
        logger.info(f"Successfully extracted data from {filepath}")
        return filepath, extracted_data
    except (IndentationError, SyntaxError) as e:
        logger.warning(f"Parsing failed for {filepath}, attempting black formatting: {e}")
        success, formatted_content = format_with_black(content)
        if success:
            try:
                tree = ast.parse(formatted_content)
                add_parent_info(tree)
                extracted_data = extractor.extract_classes_and_functions_from_ast(tree, formatted_content)
                extracted_data['file_content'] = [{"content": formatted_content}]
                logger.info(f"Successfully extracted data from {filepath} after formatting.")
                return filepath, extracted_data
            except (IndentationError, SyntaxError) as e:
                logger.error(f"Parsing failed even after black formatting for {filepath}: {e}")
                sentry_sdk.capture_exception(e)
                return filepath, {"functions": [], "classes": [], "file_content": []}
        else:
            logger.error(f"Black formatting failed for {filepath}")
            sentry_sdk.capture_message(f"Black formatting failed for {filepath}")
            return filepath, {"functions": [], "classes": [], "file_content": []}

def write_analysis_to_markdown(results: Dict[str, Dict[str, Any]], output_file_path: str, repo_dir: str) -> None:
    """Write the analysis results to a markdown file with improved readability."""
    logger.info(f"Writing analysis results to {output_file_path}")
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
                        complexity_indicator = create_complexity_indicator(complexity) if isinstance(complexity, int) else ""
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
                
        logger.info("Successfully wrote analysis to markdown.")
    except Exception as e:
        logger.error(f"Error writing markdown file: {str(e)}")
        sentry_sdk.capture_exception(e)