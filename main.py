import argparse
import asyncio
import os
import sys
import shutil
import logging
import sentry_sdk
from typing import List, Dict, Any
from urllib.parse import urlparse

from code_extraction import CodeExtractor, add_parent_info
from file_processing import (
    clone_repo,
    load_gitignore_patterns,
    get_all_files,
    process_file,
    write_analysis_to_markdown
)
from api_interaction import analyze_function_with_openai
from monitoring import initialize_sentry
from config import Config
from cache import initialize_cache

logger = logging.getLogger(__name__)

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
            logger.debug("Invalid hostname: %s", parsed.netloc)
            return False
            
        # Check path format (should be /{user}/{repo})
        path_parts = [p for p in parsed.path.split('/') if p]
        if len(path_parts) < 2:
            logger.debug("Invalid path format: %s", parsed.path)
            return False
        
        # Valid GitHub URL format
        logger.info("Valid GitHub URL: %s", url)
        return True
        
    except ValueError as e:
        logger.error("URL validation error: %s", str(e))
        return False
    
async def process_files_concurrently(files_list: List[str], service: str) -> Dict[str, Dict[str, Any]]:
    """Process multiple files concurrently."""
    logger.info("Starting to process %d files", len(files_list))
    tasks = [process_file(filepath, service) for filepath in files_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    processed_results = {}
    for filepath, result in zip(files_list, results):
        if isinstance(result, Exception):
            logger.error("Error processing file %s: %s", filepath, result)
            continue
        processed_results[filepath] = result
    logger.info("Completed processing files.")
    return processed_results

async def analyze_functions_concurrently(results: Dict[str, Dict[str, Any]], service: str) -> None:
    """Analyze multiple functions concurrently using the selected AI service."""
    logger.info("Starting function analysis using %s service", service)
    tasks = []
    for analysis in results.values():
        for func in analysis.get("functions", []):
            if func.get("name"):
                tasks.append(analyze_function_with_openai(func, service))
    analyzed_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for func_analysis in analyzed_results:
        if isinstance(func_analysis, Exception):
            logger.error("Error analyzing function: %s", func_analysis)
            continue
        func_name = func_analysis.get("name")
        for file_analysis in results.values():
            for func in file_analysis.get("functions", []):
                if func.get("name") == func_name:
                    func.update(func_analysis)
    logger.info("Completed function analysis.")

async def main():
    """Main function to analyze code and generate documentation."""
    parser = argparse.ArgumentParser(description="Analyze code and generate documentation.")
    parser.add_argument("input_path", help="Path to the input directory or repository URL")
    parser.add_argument("output_file", help="Path to the output markdown file")
    parser.add_argument("--service", choices=["azure", "openai"], required=True, help="AI service to use")
    args = parser.parse_args()
    repo_dir = None
    try:
        # Initialize services
        initialize_sentry()
        Config.validate()
        initialize_cache()

        input_path = args.input_path
        output_file = args.output_file

        # Handle repository URL
        if input_path.startswith(('http://', 'https://')):
            if not validate_repo_url(input_path):
                logger.error("Invalid GitHub repository URL: %s", input_path)
                sys.exit(1)

            repo_dir = 'cloned_repo'
            await clone_repo(input_path, repo_dir)
            input_path = repo_dir

        # Process files and generate analysis
        exclude_patterns = load_gitignore_patterns(input_path)
        python_files = get_all_files(input_path, exclude_patterns + ALWAYS_EXCLUDE_DIRS)

        if not python_files:
            logger.error("No Python files found to analyze")
            sys.exit(1)

        results = await process_files_concurrently(python_files, args.service)
        await analyze_functions_concurrently(results, args.service)

        # Generate documentation
        write_analysis_to_markdown(results, output_file, input_path)
        logger.info("Analysis complete. Documentation written to %s", output_file)

    except (ValueError, OSError) as e:
        logger.error("Error during execution: %s", str(e))
        sentry_sdk.capture_exception(e)
        sys.exit(1)

    finally:
        # Cleanup temporary files
        if repo_dir and os.path.exists(repo_dir):
            try:
                shutil.rmtree(repo_dir)
                logger.info("Cleaned up temporary repository files")
            except OSError as e:
                logger.error("Error cleaning up repository: %s", str(e))

if __name__ == "__main__":
    asyncio.run(main())
