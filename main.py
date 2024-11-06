import argparse
import asyncio
import os
import sys
import shutil
import logging
from datetime import datetime
from typing import List, Dict, Any
from urllib.parse import urlparse

from files import (
    clone_repo,
    load_gitignore_patterns,
    get_all_files,
    process_file
)
from docs import write_analysis_to_markdown
from api_interaction import analyze_function_with_openai
from monitoring import initialize_sentry, capture_exception, capture_message
from config import Config
from cache import initialize_cache
from logging_utils import setup_logger
import sentry_sdk

# Initialize logger for the main module
logger = setup_logger("main")

def validate_repo_url(url: str) -> bool:
    """
    Validate if the given URL is a valid GitHub repository URL.

    Args:
        url (str): The repository URL.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        parsed = urlparse(url)

        if parsed.netloc not in ['github.com', 'www.github.com']:
            logger.debug("Invalid hostname: %s", parsed.netloc)
            return False

        path_parts = [p for p in parsed.path.split('/') if p]
        if len(path_parts) < 2:
            logger.debug("Invalid path format: %s", parsed.path)
            return False

        logger.info("Valid GitHub URL: %s", url)
        return True

    except ValueError as e:
        logger.error("URL validation error: %s", str(e))
        return False

async def process_files_concurrently(files_list: List[str], service: str) -> Dict[str, Dict[str, Any]]:
    """
    Process multiple files concurrently.

    Args:
        files_list (List[str]): List of file paths to process.
        service (str): The AI service to use ('azure' or 'openai').

    Returns:
        Dict[str, Dict[str, Any]]: Processed results mapped by file path.
    """
    logger.info("Starting to process %d files", len(files_list))
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent tasks

    async def process_with_semaphore(filepath):
        async with semaphore:
            return await process_file(filepath, service)

    tasks = [process_with_semaphore(filepath) for filepath in files_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    processed_results = {}

    for filepath, result in zip(files_list, results):
        if isinstance(result, Exception):
            logger.error("Error processing file %s: %s", filepath, result)
            sentry_sdk.capture_exception(result)
            continue
        if result and ('classes' in result or 'functions' in result):
            processed_results[filepath] = result
        else:
            logger.warning(f"No valid data extracted for {filepath}")

    logger.info("Completed processing files.")
    return processed_results

async def analyze_functions_concurrently(results: Dict[str, Dict[str, Any]], service: str) -> None:
    """
    Analyze multiple functions concurrently using the selected AI service.

    Args:
        results (Dict[str, Dict[str, Any]]): The extracted data from files.
        service (str): The AI service to use ('azure' or 'openai').
    """
    logger.info("Starting function analysis using %s service", service)
    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent API calls

    async def analyze_with_semaphore(func, service):
        async with semaphore:
            return await analyze_function_with_openai(func, service)

    tasks = []

    for analysis in results.values():
        functions = analysis.get("functions", [])
        for func in functions:
            if isinstance(func, dict) and func.get("name"):
                tasks.append(analyze_with_semaphore(func, service))

    if not tasks:
        logger.warning("No functions found to analyze")
        return

    analyzed_results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in analyzed_results:
        if isinstance(result, Exception):
            logger.error("Error analyzing function: %s", result)
            sentry_sdk.capture_exception(result)
            continue
        if isinstance(result, dict):
            func_name = result.get("name")
            if func_name:
                for analysis in results.values():
                    for func in analysis.get("functions", []):
                        if isinstance(func, dict) and func.get("name") == func_name:
                            func.update(result)

    logger.info("Completed function analysis.")

async def main():
    """
    Main function to analyze code and generate documentation.
    """
    parser = argparse.ArgumentParser(description="Analyze code and generate documentation.")
    parser.add_argument("input_path", help="Path to the input directory or repository URL")
    parser.add_argument("output_path", help="Path to the output directory for markdown files")
    parser.add_argument("--service", choices=["azure", "openai"], required=True, help="AI service to use")

    args = parser.parse_args()
    repo_dir = None

    summary_data = {
        'files_processed': 0,
        'errors_encountered': 0,
        'start_time': datetime.now(),
        'end_time': None
    }

    try:
        # Initialize monitoring (Sentry)
        initialize_sentry()
        
        # Load environment variables and validate
        Config.load_environment()
        
        # Initialize cache
        initialize_cache()

        input_path = args.input_path
        output_path = args.output_path

        if input_path.startswith(('http://', 'https://')):
            if not validate_repo_url(input_path):
                logger.error("Invalid GitHub repository URL: %s", input_path)
                sys.exit(1)

            repo_dir = 'cloned_repo'
            await clone_repo(input_path, repo_dir)
            input_path = repo_dir

        exclude_patterns = load_gitignore_patterns(input_path)
        python_files = get_all_files(input_path, exclude_patterns)

        if not python_files:
            logger.error("No Python files found to analyze")
            sys.exit(1)

        results = await process_files_concurrently(python_files, args.service)
        summary_data['files_processed'] = len(results)

        if not results:
            logger.error("No valid results from file processing")
            sys.exit(1)

        await analyze_functions_concurrently(results, args.service)

        # Await the coroutine
        await write_analysis_to_markdown(results, output_path, input_path)
        logger.info("Analysis complete. Documentation written to %s", output_path)

    except Exception as e:
        summary_data['errors_encountered'] += 1
        logger.error("Error during execution: %s", str(e))
        sentry_sdk.capture_exception(e)
        sys.exit(1)

    finally:
        summary_data['end_time'] = datetime.now()
        if repo_dir and os.path.exists(repo_dir):
            try:
                shutil.rmtree(repo_dir)
                logger.info("Cleaned up temporary repository files")
            except OSError as e:
                logger.error("Error cleaning up repository: %s", str(e))

        logger.info(
            "Summary: Files processed: %d, Errors: %d, Start: %s, End: %s, Duration: %s",
            summary_data['files_processed'],
            summary_data['errors_encountered'],
            summary_data['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
            summary_data['end_time'].strftime('%Y-%m-%d %H:%M:%S'),
            str(summary_data['end_time'] - summary_data['start_time'])
        )

if __name__ == "__main__":
    asyncio.run(main())
