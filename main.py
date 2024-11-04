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
    write_analysis_to_markdown,
    load_gitignore_patterns,
    FileProcessor  # Import the FileProcessor
)
from api_interaction import analyze_function_with_openai, RateLimiter  # Import RateLimiter
from config import Config
from code_extraction import CodeExtractor

# Initialize Sentry
initialize_sentry()

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define directories to always exclude
always_exclude_dirs = ['.git', '__pycache__', 'venv', 'node_modules']

# Initialize instances for FileProcessor, CodeExtractor, and RateLimiter
file_processor = FileProcessor()
code_extractor = CodeExtractor()
rate_limiter = RateLimiter(tokens_per_second=0.5, bucket_size=10)  # Adjust as needed

async def process_files_concurrently(files_list):
    """Process multiple files concurrently using FileProcessor."""
    tasks = [file_processor.process_large_file(filepath) for filepath in files_list]
    results = await asyncio.gather(*tasks)

    # Filter out failed results and log errors
    successful_results = {}
    for result, filepath in zip(results, files_list):
        if result.success:
            # Process the file content here using CodeExtractor
            try:
                async with aiofiles.open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = await f.read()
                tree = ast.parse(content)
                extracted_data = code_extractor.extract_classes_and_functions_from_ast(tree, content)
                successful_results[filepath] = extracted_data
            except Exception as e:
                logging.error(f"Error processing {filepath}: {e}")
                sentry_sdk.capture_exception(e)
        else:
            logging.error(f"Failed to process file {filepath}: {result.error}")

    return successful_results

async def analyze_functions_concurrently(functions, service):
    """Analyze multiple functions concurrently using the selected AI service and rate limiting."""
    tasks = []
    for func in functions:
        await rate_limiter.acquire()  # Apply rate limiting before each API call
        tasks.append(analyze_function_with_openai(func, service))
    return await asyncio.gather(*tasks)

async def main():
    """Main function to orchestrate code analysis and documentation generation."""
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
            clone_result = await clone_repo(repo_url, repo_dir)
            if not clone_result.success:
                logging.error(f"Failed to clone repository: {clone_result.error}")
                sys.exit(1)
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

        # Analyze functions using the specified AI service
        for filepath, extracted_data in results.items():
            functions = extracted_data.get("functions", [])
            function_analysis_results = await analyze_functions_concurrently(functions, args.service)
            extracted_data["functions"] = function_analysis_results

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
