"""
Simplified main module for docstring generation workflow.
"""

import asyncio
import argparse
import os
import signal
from pathlib import Path
from typing import Optional, List
from core.logger import log_info, log_error, log_debug
from core.config import AzureOpenAIConfig
from core.utils import ensure_directory
from simplified_interaction import AIInteractionHandler

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Docstring Generation System'
    )
    
    parser.add_argument(
        'source_path',
        help='Path to Python source file or directory'
    )
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory for documentation'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Number of functions to process concurrently'
    )
    
    # Cache settings
    parser.add_argument('--redis-host', default='localhost')
    parser.add_argument('--redis-port', type=int, default=6379)
    parser.add_argument('--redis-db', type=int, default=0)
    parser.add_argument('--redis-password', default=None)
    
    return parser.parse_args()

async def process_file(
    file_path: Path,
    handler: AIInteractionHandler,
    output_dir: Path
) -> bool:
    """Process a single Python file."""
    try:
        log_debug(f"Processing file: {file_path}")
        
        # Read source code
        source_code = file_path.read_text(encoding='utf-8')
        
        # Process code
        updated_code, documentation = await handler.process_code(source_code)
        if not updated_code or not documentation:
            log_error(f"Failed to process {file_path}")
            return False
            
        # Save outputs
        output_path = output_dir / file_path.name
        doc_path = output_dir / f"{file_path.stem}_docs.md"
        
        output_path.write_text(updated_code, encoding='utf-8')
        doc_path.write_text(documentation, encoding='utf-8')
        
        log_info(f"Successfully processed {file_path}")
        return True
        
    except Exception as e:
        log_error(f"Error processing {file_path}: {str(e)}")
        return False

async def process_directory(
    directory: Path,
    handler: AIInteractionHandler,
    output_dir: Path
) -> List[Path]:
    """Process all Python files in a directory."""
    processed_files = []
    
    for file_path in directory.rglob("*.py"):
        if await process_file(file_path, handler, output_dir):
            processed_files.append(file_path)
    
    return processed_files

async def run_workflow(args: argparse.Namespace) -> None:
    """Run the docstring generation workflow."""
    source_path = Path(args.source_path)
    output_dir = Path(args.output_dir)
    
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Initialize handler
    config = AzureOpenAIConfig.from_env()
    cache_config = {
        'host': args.redis_host,
        'port': args.redis_port,
        'db': args.redis_db,
        'password': args.redis_password
    }
    
    async with AIInteractionHandler(
        config=config,
        cache_config=cache_config,
        batch_size=args.batch_size
    ) as handler:
        if source_path.is_file():
            await process_file(source_path, handler, output_dir)
        elif source_path.is_dir():
            await process_directory(source_path, handler, output_dir)
        else:
            log_error(f"Invalid source path: {source_path}")

def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        asyncio.run(run_workflow(args))
        log_info("Workflow completed successfully")
    except KeyboardInterrupt:
        log_info("Operation cancelled by user")
    except Exception as e:
        log_error(f"Workflow failed: {str(e)}")

if __name__ == "__main__":
    main()