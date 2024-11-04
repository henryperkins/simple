# main.py
import argparse
import asyncio
import logging
import sys
import shutil
import os
import black
import sentry_sdk
from monitoring import initialize_sentry
from file_processing import (
    clone_repo,
    get_all_files,
    process_file,
    write_analysis_to_markdown
)
from api_interaction import analyze_function_with_openai
from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    OPENAI_API_KEY
)

# Initialize logging
logging.basicConfig(level=logging.INFO)

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
    """Main entry point for the documentation generator."""
    parser = argparse.ArgumentParser(description="Analyze a GitHub repository or local directory.")
    parser.add_argument("input_path", help="GitHub Repository URL or Local Directory Path")
    parser.add_argument("output_file", help="File to save Markdown output")
    parser.add_argument("--environment", default="development", help="Environment (development/production)")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent requests")
    parser.add_argument("--service", choices=["openai", "azure"], default="openai", help="AI service to use")
    
    args = parser.parse_args()
    
    # Initialize Sentry
    initialize_sentry(environment=args.environment)
    
    try:
        with sentry_sdk.start_transaction(op="process_repository", name=f"Process Repository: {args.input_path}"):
            # Validate service configuration
            if args.service == "azure":
                if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME]):
                    logging.error("Azure OpenAI environment variables are not fully set.")
                    sys.exit(1)
            elif not OPENAI_API_KEY:
                logging.error("OpenAI API key is not set.")
                sys.exit(1)

            # Handle repository input
            cleanup_needed = False
            if args.input_path.startswith(("http://", "https://")):
                repo_url = args.input_path
                repo_dir = "cloned_repo"
                clone_repo(repo_url, repo_dir)
                cleanup_needed = True
            else:
                repo_dir = args.input_path
                if not os.path.isdir(repo_dir):
                    logging.error(f"Directory does not exist: {repo_dir}")
                    sys.exit(1)

            try:
                # Process files
                files_list = get_all_files(repo_dir, exclude_dirs=[".git", ".github"])
                file_results = await process_files_concurrently(files_list)

                # Analyze and document code
                results = {}
                for filepath, extracted_data in file_results:
                    functions = extracted_data.get("functions", [])
                    if functions:
                        analysis_results = await analyze_functions_concurrently(functions, args.service)
                        results[filepath] = {
                            "functions": analysis_results,
                            "source_code": extracted_data.get("file_content", "")
                        }

                # Generate documentation
                write_analysis_to_markdown(results, args.output_file, repo_dir)
                logging.info(f"Documentation written to {args.output_file}")

            finally:
                if cleanup_needed and os.path.exists(repo_dir):
                    try:
                        shutil.rmtree(repo_dir)
                        logging.info(f"Cleaned up repository directory: {repo_dir}")
                    except Exception as e:
                        logging.error(f"Error cleaning up repository directory: {e}")
                        sentry_sdk.capture_exception(e)

    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sentry_sdk.capture_exception(e)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())