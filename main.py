import argparse
import asyncio
import os
import ast
import aiofiles
import sys
import shutil
import logging
import sentry_sdk
from typing import List, Dict, Any, Tuple
from code_extraction import CodeExtractor, add_parent_info
from file_processing import clone_repo, load_gitignore_patterns, get_all_files, process_file, create_complexity_indicator
from api_interaction import analyze_function_with_openai
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Add after imports in main.py
ALWAYS_EXCLUDE_DIRS = [
    'venv',
    'node_modules',
    '.git',
    '__pycache__',
    '.pytest_cache',
    'dist',
    'build',
    '.vscode',
    '.idea'
]

def validate_repo_url(url: str) -> bool:
    """
    Validate if the given URL is a valid GitHub repository URL.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if valid GitHub repo URL, False otherwise
    """
    try:
        # Parse URL 
        parsed = urlparse(url)
        
        # Check if hostname is github.com or www.github.com
        if parsed.netloc not in ['github.com', 'www.github.com']:
            logger.debug(f"Invalid hostname: {parsed.netloc}")
            return False
            
        # Check path format (should be /{user}/{repo})
        path_parts = [p for p in parsed.path.split('/') if p]
        if len(path_parts) < 2:
            logger.debug(f"Invalid path format: {parsed.path}")
            return False
        
        # Valid GitHub URL format
        logger.info(f"Valid GitHub URL: {url}")
        return True
        
    except Exception as e:
        logger.error(f"URL validation error: {str(e)}")
        return False
    
async def process_files_concurrently(files_list: List[str]) -> Dict[str, Dict[str, Any]]:
    """Process multiple files concurrently."""
    logger.info(f"Starting to process {len(files_list)} files.")
    tasks = [process_file(filepath) for filepath in files_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    processed_results = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Error processing file: {result}")
            continue
        filepath, data = result
        processed_results[filepath] = data
    logger.info("Completed processing files.")
    return processed_results

async def analyze_functions_concurrently(results: Dict[str, Dict[str, Any]], service: str) -> None:
    """Analyze multiple functions concurrently using the selected AI service."""
    logger.info(f"Starting function analysis using {service} service.")
    tasks = []
    for analysis in results.values():
        for func in analysis.get("functions", []):
            if func.get("name"):
                tasks.append(analyze_function_with_openai(func, service))
    analyzed_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for func_analysis in analyzed_results:
        if isinstance(func_analysis, Exception):
            logger.error(f"Error analyzing function: {func_analysis}")
            continue
        func_name = func_analysis.get("name")
        for file_analysis in results.values():
            for func in file_analysis.get("functions", []):
                if func.get("name") == func_name:
                    func.update(func_analysis)
    logger.info("Completed function analysis.")

def write_analysis_to_markdown(results: Dict[str, Dict[str, Any]], output_file: str, repo_dir: str) -> None:
    """Write the analysis results to a Markdown file."""
    with open(output_file, 'w', encoding='utf-8') as md_file:
        for filepath, analysis in results.items():
            md_file.write(f"# {os.path.relpath(filepath, repo_dir)}\n\n")
            
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

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze code and generate documentation.")
    parser.add_argument("input_path", help="Path to the input directory or repository URL")
    parser.add_argument("output_file", help="Path to the output markdown file")
    parser.add_argument("--service", choices=["azure", "openai"], required=True, help="AI service to use")
    args = parser.parse_args()
    
    try:
        input_path = args.input_path
        output_file = args.output_file
        cleanup_needed = False
        if input_path.startswith(('http://', 'https://')):
            if not validate_repo_url(input_path):
                logger.error(f"Invalid GitHub repository URL: {input_path}")
                sys.exit(1)
            
            repo_url = input_path
            repo_dir = 'cloned_repo'
            logger.info(f"Input is a GitHub repository URL: {repo_url}")
            
            try:
                await clone_repo(repo_url, repo_dir)  # Await the coroutine
                cleanup_needed = True
            except Exception as e:
                logger.error(f"Failed to clone repository: {e}")
                sentry_sdk.capture_exception(e)
                sys.exit(1)
        else:
            repo_dir = input_path
            logger.info(f"Using local directory: {repo_dir}")
        
        if not os.path.isdir(repo_dir):
            logger.error(f"The directory {repo_dir} does not exist.")
            sys.exit(1)
        logger.info(f"Scanning directory {repo_dir} for Python files.")
        
        gitignore_patterns = load_gitignore_patterns(repo_dir)
        files_list = get_all_files(repo_dir, exclude_dirs=ALWAYS_EXCLUDE_DIRS + gitignore_patterns)
        if not files_list:
            logger.error("No Python files found to analyze.")
            sys.exit(1)
        logger.info(f"Found {len(files_list)} Python files to analyze.")
        
        # Process files concurrently
        results = await process_files_concurrently(files_list)
        
        # Analyze functions concurrently
        await analyze_functions_concurrently(results, args.service)
        
        # Write analysis to markdown
        write_analysis_to_markdown(results, output_file, repo_dir)
        logger.info(f"Documentation successfully written to {output_file}")
        
        if cleanup_needed:
            try:
                shutil.rmtree(repo_dir)
                logger.info(f"Cleaned up cloned repository at {repo_dir}")
            except Exception as e:
                logger.error(f"Failed to clean up cloned repository: {e}")
                sentry_sdk.capture_exception(e)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sentry_sdk.capture_exception(e)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())