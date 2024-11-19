# main.py
"""
Main module for docstring generation workflow.
Handles file processing, caching, and documentation generation.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List

from core.ai_interaction import AIInteractionHandler
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.logger import log_info, log_error, log_debug
from core.utils import ensure_directory
from core.exceptions import CacheError, DocumentationError

async def process_file(
    file_path: Path,
    handler: AIInteractionHandler,
    output_dir: Path
) -> bool:
    """
    Process a single Python file to generate documentation.

    Args:
        file_path (Path): Path to the Python file to process
        handler (AIInteractionHandler): Handler instance for AI interactions
        output_dir (Path): Directory to save processed files and documentation

    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        log_debug(f"Processing file: {file_path}")
        source_code = file_path.read_text(encoding='utf-8')
        
        # Process code and generate documentation
        updated_code, documentation = await handler.process_code(source_code)

        if documentation:
            # Ensure output directory exists
            ensure_directory(str(output_dir))
            
            # Save processed files
            output_path = output_dir / file_path.name
            doc_path = output_dir / f"{file_path.stem}_docs.md"
            
            output_path.write_text(updated_code, encoding='utf-8')
            doc_path.write_text(documentation, encoding='utf-8')
            
            log_info(f"Successfully processed {file_path}")
            return True
        else:
            log_error(f"Failed to generate documentation for {file_path}")
            return False

    except DocumentationError as e:
        log_error(f"Documentation error for {file_path}: {str(e)}")
        return False
    except Exception as e:
        log_error(f"Error processing {file_path}: {str(e)}")
        return False

async def process_directory(
    directory: Path,
    handler: AIInteractionHandler,
    output_dir: Path
) -> int:
    """
    Process all Python files in a directory concurrently.

    Args:
        directory (Path): Directory containing Python files
        handler (AIInteractionHandler): Handler instance for AI interactions
        output_dir (Path): Directory to save processed files and documentation

    Returns:
        int: Number of successfully processed files
    """
    try:
        # Create tasks for all Python files
        tasks = [
            process_file(file_path, handler, output_dir)
            for file_path in directory.rglob("*.py")
        ]
        
        if not tasks:
            log_error(f"No Python files found in {directory}")
            return 0

        # Process files concurrently
        results: List[bool] = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful operations (excluding exceptions)
        success_count = sum(1 for result in results if isinstance(result, bool) and result)
        
        # Log any exceptions that occurred
        for result in results:
            if isinstance(result, Exception):
                log_error(f"Error during batch processing: {str(result)}")
        
        return success_count

    except Exception as e:
        log_error(f"Error processing directory {directory}: {str(e)}")
        return 0

async def run_workflow(args: argparse.Namespace) -> None:
    """
    Run the docstring generation workflow.

    Args:
        args (argparse.Namespace): Parsed command line arguments
    """
    source_path = Path(args.source_path)
    output_dir = Path(args.output_dir)
    
    try:
        # Initialize configuration
        config = AzureOpenAIConfig.from_env()
        
        # Prepare cache configuration
        cache_config = {
            'host': args.redis_host,
            'port': args.redis_port,
            'db': args.redis_db,
            'password': args.redis_password,
            'enabled': args.enable_cache
        }

        # Initialize AI interaction handler with cache config
        async with AIInteractionHandler(
            config=config,
            cache=Cache(**cache_config) if args.enable_cache else None,
            batch_size=config.batch_size
        ) as handler:
            if source_path.is_file():
                success = await process_file(source_path, handler, output_dir)
                if not success:
                    log_error(f"Failed to process file: {source_path}")
                    
            elif source_path.is_dir():
                processed_count = await process_directory(source_path, handler, output_dir)
                log_info(f"Successfully processed {processed_count} files")
                
            else:
                log_error(f"Invalid source path: {source_path}")
                return

            # Log final metrics
            metrics = await handler.get_metrics_summary()
            log_info(f"Final metrics: {metrics}")

    except CacheError as e:
        log_error(f"Cache error: {e}")
    except Exception as e:
        log_error(f"Workflow error: {str(e)}")
        raise

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate docstrings for Python code using AI."
    )
    
    # Required arguments
    parser.add_argument(
        "source_path",
        help="Path to the Python file or directory to process"
    )
    parser.add_argument(
        "output_dir",
        help="Directory to save the processed files and documentation"
    )
    
    # Cache configuration
    cache_group = parser.add_argument_group('Cache Configuration')
    cache_group.add_argument(
        "--enable-cache",
        action="store_true",
        help="Enable Redis caching"
    )
    cache_group.add_argument(
        "--redis-host",
        default="localhost",
        help="Redis host address (default: localhost)"
    )
    cache_group.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port number (default: 6379)"
    )
    cache_group.add_argument(
        "--redis-db",
        type=int,
        default=0,
        help="Redis database number (default: 0)"
    )
    cache_group.add_argument(
        "--redis-password",
        help="Redis password (optional)"
    )
    
    # Processing options
    process_group = parser.add_argument_group('Processing Options')
    process_group.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of files to process in parallel (default: 5)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the docstring generation tool."""
    args = parse_arguments()
    
    try:
        asyncio.run(run_workflow(args))
        log_info("Workflow completed successfully")
        
    except KeyboardInterrupt:
        log_info("Operation cancelled by user")
        sys.exit(1)
        
    except Exception as e:
        log_error(f"Workflow failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()