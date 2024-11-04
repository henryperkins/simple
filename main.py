# main.py

import argparse
import asyncio
import logging
import sys
import shutil
import os
import sentry_sdk
from monitoring import initialize_sentry
from file_processing import (
    clone_repo,
    get_all_files,
    process_file,
    write_analysis_to_markdown,
    load_gitignore_patterns  # Add this import
)
from api_interaction import analyze_function_with_openai
from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    OPENAI_API_KEY
)

# Initialize Sentry
initialize_sentry()

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define directories to always exclude
always_exclude_dirs = ['.git', '__pycache__', 'venv', 'node_modules']

async def process_files_concurrently(files_list):
    """Process multiple files concurrently.
    
    Args:
        files_list (list): A list of file paths to process.
        
    Returns:
        list: A list of tuples containing file paths and extracted data.
    """
    tasks = [process_file(filepath) for filepath in files_list]
    return await asyncio.gather(*tasks)

async def analyze_functions_concurrently(functions, service):
    """Analyze multiple functions concurrently using the selected AI service.
    
    Args:
        functions (list): A list of function details to analyze.
        service (str): The service to use ('openai' or 'azure').
        
    Returns:
        list: A list of analysis results for each function.
    """
    tasks = [analyze_function_with_openai(func, service) for func in functions]
    return await asyncio.gather(*tasks)

async def main():
    parser = argparse.ArgumentParser(description="Analyze code and generate documentation.")
    parser.add_argument("input_path", help="Path to the input directory or repository URL")
    parser.add_argument("output_file", help="Path to the output markdown file")
    parser.add_argument("--service", help="The service to use ('openai' or 'azure')", required=True)
    args = parser.parse_args()

    try:
        input_path = args.input_path
        output_file = args.output_file

        if input_path.startswith('http://') or input_path.startswith('https://'):
            repo_url = input_path
            repo_dir = 'cloned_repo'
            logging.debug(f"Input is a GitHub repository URL: {repo_url}")
            clone_repo(repo_url, repo_dir)
            cleanup_needed = True
        else:
            repo_dir = input_path
            if not os.path.isdir(repo_dir):
                logging.error(f"The directory {repo_dir} does not exist.")
                sys.exit(1)
            logging.info(f"Using local directory {repo_dir}")

        spec = load_gitignore_patterns(repo_dir)

        files_list = get_all_files(repo_dir, exclude_dirs=always_exclude_dirs)
        logging.info(f"Found {len(files_list)} files after applying ignore patterns and exclusions.")

        results = await process_files_concurrently(files_list)
        write_analysis_to_markdown(results, output_file, repo_dir)
        logging.info(f"Documentation written to {output_file}")

        if cleanup_needed:
            try:
                shutil.rmtree(repo_dir)
                logging.info(f"Cleaned up cloned repository at {repo_dir}")
            except Exception as e:
                logging.error(f"Failed to clean up cloned repository: {e}")
                sentry_sdk.capture_exception(e)

    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sentry_sdk.capture_exception(e)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())