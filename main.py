import argparse
import asyncio
import logging
import sys
import shutil
import os
from typing import List, Dict, Any
from file_processing import (
    clone_repo, 
    get_all_files,
    process_file, 
    load_gitignore_patterns,
    write_analysis_to_markdown
)
from api_interaction import analyze_function_with_openai
from config import Config
from monitoring import initialize_sentry
from cache import initialize_cache

# Initialize Sentry
initialize_sentry()

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize cache
initialize_cache()

# Define directories to always exclude
always_exclude_dirs = ['.git', '__pycache__', 'venv', 'node_modules']

def validate_repo_url(url: str) -> bool:
    """Validate repository URL format."""
    return url.startswith(('https://github.com/', 'https://www.github.com/')) and len(url.split('/')) >= 5

async def process_files_concurrently(files_list: List[str]) -> Dict[str, Dict[str, Any]]:
    """Process multiple files concurrently."""
    tasks = [process_file(filepath) for filepath in files_list]
    results = await asyncio.gather(*tasks)
    return dict(results)

async def analyze_functions_concurrently(results: Dict[str, Dict[str, Any]], service: str) -> None:
    """Analyze multiple functions concurrently using the selected AI service."""
    tasks = []
    for analysis in results.values():
        for func in analysis.get("functions", []):
            if func.get("name"):
                tasks.append(analyze_function_with_openai(func, service))
    analyzed_results = await asyncio.gather(*tasks)
    
    # Update results with analysis
    for func_analysis in analyzed_results:
        func_name = func_analysis.get("name")
        for file_analysis in results.values():
            for func in file_analysis.get("functions", []):
                if func.get("name") == func_name:
                    func.update(func_analysis)

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
                logging.error(f"Invalid GitHub repository URL: {input_path}")
                sys.exit(1)
            
            repo_url = input_path
            repo_dir = 'cloned_repo'
            logging.info(f"Input is a GitHub repository URL: {repo_url}")
            
            try:
                clone_repo(repo_url, repo_dir)
                cleanup_needed = True
            except Exception as e:
                logging.error(f"Failed to clone repository: {e}")
                sys.exit(1)
        else:
            repo_dir = input_path

        if not os.path.isdir(repo_dir):
            logging.error(f"The directory {repo_dir} does not exist.")
            sys.exit(1)
        logging.info(f"Using local directory {repo_dir}")
        gitignore_patterns = load_gitignore_patterns(repo_dir)
        files_list = get_all_files(repo_dir, exclude_dirs=always_exclude_dirs + gitignore_patterns)
        if not files_list:
            logging.error("No Python files found to analyze.")
            sys.exit(1)
        logging.info(f"Found {len(files_list)} files after applying ignore patterns and exclusions.")
        # Process files concurrently
        results = await process_files_concurrently(files_list)
        # Analyze functions concurrently
        await analyze_functions_concurrently(results, args.service)
        # Write analysis to markdown
        write_analysis_to_markdown(results, output_file, repo_dir)
        logging.info(f"Documentation written to {output_file}")
        if cleanup_needed:
            try:
                shutil.rmtree(repo_dir)
                logging.info(f"Cleaned up cloned repository at {repo_dir}")
            except Exception as e:
                logging.error(f"Failed to clean up cloned repository: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sentry_sdk.capture_exception(e)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())