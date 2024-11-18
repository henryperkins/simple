"""
Main Module for Docstring Workflow System

This module serves as the entry point for the docstring generation workflow
using Azure OpenAI. It handles command-line arguments, initializes necessary
components, and orchestrates the processing of Python source files to generate
and update docstrings.

Version: 1.0.1
Author: Development Team
"""

import argparse
import asyncio
import os
import aiofiles
import shutil
import tempfile
import subprocess
import time
import signal
from contextlib import contextmanager, AsyncExitStack
from typing import Optional, Tuple, Dict, Any
from dotenv import load_dotenv
from interaction_handler import InteractionHandler
from core.logger import log_info, log_error, log_debug, log_warning
from core.monitoring import SystemMonitor
from core.utils import ensure_directory
from api.models.api_client import AzureOpenAIClient
from core.config import AzureOpenAIConfig
from pathlib import Path
import ast

# Load environment variables from .env file
load_dotenv()

monitor = SystemMonitor()


def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    log_info("Received shutdown signal, cleaning up...")
    raise KeyboardInterrupt


# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


@contextmanager
def cleanup_context(temp_dir: Optional[str] = None):
    """Context manager for cleanup operations."""
    try:
        yield
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                log_debug(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir)
                log_info("Cleanup completed successfully")
            except Exception as e:
                log_error(f"Error during cleanup: {e}")


async def initialize_client() -> AzureOpenAIClient:
    """
    Initialize the Azure OpenAI client with proper retry logic.

    Returns:
        AzureOpenAIClient: Configured client instance
    """
    log_debug("Initializing Azure OpenAI client")
    config = AzureOpenAIConfig.from_env()
    client = AzureOpenAIClient(config)
    log_info("Azure OpenAI client initialized successfully")
    return client


def load_source_file(file_path: str) -> str:
    """
    Load the source code from a specified file path.

    Args:
        file_path (str): The path to the source file.

    Returns:
        str: The content of the source file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there is an error reading the file.
    """
    try:
        log_debug(f"Attempting to load source file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            source_code = file.read()
            log_info(f"Successfully loaded source code from '{file_path}'")
            return source_code
    except FileNotFoundError:
        log_error(f"File '{file_path}' not found.")
        raise
    except IOError as e:
        log_error(f"Failed to read file '{file_path}': {e}")
        raise


def save_updated_source(file_path: str, updated_code: str) -> None:
    """
    Save the updated source code to a specified file path.

    Args:
        file_path (str): The path to save the updated source code.
        updated_code (str): The updated source code to save.

    Raises:
        IOError: If there is an error writing to the file.
    """
    try:
        log_debug(f"Attempting to save updated source code to: {file_path}")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(updated_code)
            log_info(f"Successfully saved updated source code to '{file_path}'")
    except IOError as e:
        log_error(f"Failed to save updated source code to '{file_path}': {e}")
        raise


class SourceCodeHandler:
    """Handles source code preprocessing and validation."""

    @staticmethod
    def preprocess_source(source_code: str) -> Tuple[str, bool]:
        """
        Preprocess source code to handle common issues.

        Args:
            source_code: Raw source code

        Returns:
            Tuple[str, bool]: (processed_code, was_modified)
        """
        modified = False
        lines = source_code.splitlines()
        processed_lines = []
        current_indent = 0

        for line in lines:
            # Skip blank lines
            if not line.strip():
                processed_lines.append('')
                continue

            # Count leading spaces and tabs
            leading_spaces = len(line) - len(line.lstrip(' '))
            leading_tabs = len(line) - len(line.lstrip('\t'))

            # Handle mixed indentation
            if '\t' in line and ' ' in line[:line.index('\t')]:
                # Convert to spaces
                processed_line = line.replace('\t', '    ')
                modified = True
            else:
                processed_line = line

            # Handle inconsistent indentation
            stripped_line = line.strip()
            if stripped_line:
                # Calculate proper indentation
                if stripped_line.startswith(('def ', 'class ', 'if ', 'elif ',
                                             'else:', 'try:', 'except',
                                             'finally:', 'with ')):
                    next_indent = current_indent + 4
                elif stripped_line == 'else:' or stripped_line.startswith(
                        ('elif ', 'except', 'finally:')):
                    current_indent = max(0, current_indent - 4)
                    next_indent = current_indent + 4
                else:
                    next_indent = current_indent

                # Apply proper indentation
                if leading_spaces > 0 or leading_tabs > 0:
                    processed_line = ' ' * current_indent + stripped_line
                    modified = True
                current_indent = next_indent

            processed_lines.append(processed_line)

        return '\n'.join(processed_lines), modified

    @staticmethod
    def validate_and_fix(source_code: str) -> Tuple[str, bool, Optional[str]]:
        """
        Validate and attempt to fix source code.

        Args:
            source_code: Source code to validate

        Returns:
            Tuple[str, bool, Optional[str]]: (processed_code, is_valid, error_message)
        """
        try:
            # First try to compile the original code
            ast.parse(source_code)
            return source_code, True, None
        except (SyntaxError, IndentationError, TabError) as e:
            log_warning(f"Initial validation failed: {str(e)}")

            try:
                # Attempt to fix the code
                processed_code, was_modified = SourceCodeHandler.preprocess_source(source_code)

                if was_modified:
                    # Validate the processed code
                    ast.parse(processed_code)
                    log_info("Successfully fixed source code issues")
                    return processed_code, True, None
                else:
                    return source_code, False, str(e)

            except Exception as fix_error:
                log_error(f"Failed to fix source code: {str(fix_error)}")
                return source_code, False, str(fix_error)

    @staticmethod
    def extract_metadata(source_code: str) -> Optional[Dict[str, Any]]:
        """
        Safely extract metadata from source code.

        Args:
            source_code: Source code to analyze

        Returns:
            Optional[Dict[str, Any]]: Extracted metadata or None if failed
        """
        try:
            # Try to fix any issues first
            processed_code, is_valid, error = SourceCodeHandler.validate_and_fix(source_code)

            if not is_valid:
                log_error(f"Could not process source code: {error}")
                return None

            tree = ast.parse(processed_code)

            metadata = {
                'classes': [],
                'functions': []
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node) or '',
                        'methods': [],
                        'is_exception': any(
                            base.id in ('Exception', 'BaseException')
                            for base in node.bases
                            if isinstance(base, ast.Name)
                        )
                    }
                    metadata['classes'].append(class_info)

                elif isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node) or '',
                        'args': [],
                        'return_type': 'Any'
                    }
                    metadata['functions'].append(func_info)

            return metadata

        except Exception as e:
            log_error(f"Error extracting metadata: {str(e)}")
            return None


def process_source_safely(file_path: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Process source code file with comprehensive error handling.

    Args:
        file_path: Path to source file

    Returns:
        Tuple[Optional[str], Optional[Dict[str, Any]]]: (processed_code, metadata)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()

        # Process the source code
        handler = SourceCodeHandler()
        processed_code, is_valid, error = handler.validate_and_fix(source_code)

        if not is_valid:
            log_error(f"Source code validation failed: {error}")
            return None, None

        # Extract metadata
        metadata = handler.extract_metadata(processed_code)
        if metadata is None:
            log_error("Failed to extract metadata")
            return None, None

        return processed_code, metadata

    except Exception as e:
        log_error(f"Error processing source file: {str(e)}")
        return None, None


async def process_file(file_path: str, args: argparse.Namespace, client: AzureOpenAIClient) -> None:
    """
    Process a single Python file with proper resource management.

    Args:
        file_path (str): Path to the Python file.
        args (argparse.Namespace): Command-line arguments.
        client (AzureOpenAIClient): Azure OpenAI client instance.
    """
    log_debug(f"Processing file: {file_path}")
    start_time = time.time()
    file_handle = None
    temp_resources = []

    try:
        # Load and preprocess the source code
        async with AsyncExitStack() as stack:
            file_handle = await stack.enter_async_context(
                aiofiles.open(file_path, mode='r', encoding='utf-8')
            )
            await file_handle.read()  # Removed unused variable

            processed_code, metadata = process_source_safely(file_path)

            if not processed_code or not metadata:
                log_error(f"Could not process {file_path}: Invalid source code or metadata")
                return

            # Initialize the interaction handler with proper cleanup
            interaction_handler = InteractionHandler(
                client=client,
                cache_config={
                    'host': args.redis_host,
                    'port': args.redis_port,
                    'db': args.redis_db,
                    'password': args.redis_password
                }
            )

            # Process functions and classes
            updated_code, documentation = await interaction_handler.process_all_functions(processed_code)

            if updated_code and documentation:
                # Ensure the output directory exists
                ensure_directory(args.output_dir)

                # Save outputs with proper resource management
                doc_path = Path(args.output_dir) / f"{Path(file_path).stem}_docs.md"
                output_path = Path(args.output_dir) / Path(file_path).name

                async with aiofiles.open(doc_path, 'w', encoding='utf-8') as doc_file:
                    await doc_file.write(documentation)
                async with aiofiles.open(output_path, 'w', encoding='utf-8') as out_file:
                    await out_file.write(updated_code)

                log_info(f"Documentation saved to: {doc_path}")
                log_info(f"Updated source code saved to: {output_path}")
            else:
                log_warning(f"No updates generated for {file_path}")

    except Exception as e:
        log_error(f"Failed to process file {file_path}: {e}")
        monitor.log_api_request(
            endpoint=file_path,
            tokens=0,
            response_time=time.time() - start_time,
            status="error",
            prompt_tokens=0,
            completion_tokens=0,
            estimated_cost=0.0,
            error=str(e)
        )
        raise
    finally:
        # Cleanup any remaining resources
        for resource in temp_resources:
            try:
                if hasattr(resource, 'close'):
                    await resource.close()
                elif hasattr(resource, '__aexit__'):
                    await resource.__aexit__(None, None, None)
            except Exception as e:
                log_error(f"Error cleaning up resource: {e}")


async def run_workflow(args: argparse.Namespace) -> None:
    """
    Run the docstring generation workflow with enhanced error handling.

    Args:
        args: Command-line arguments
    """
    source_path = args.source_path
    temp_dir = None
    client = None  # Ensure client is initialized

    try:
        # Initialize client
        client = await asyncio.wait_for(
            initialize_client(),
            timeout=30
        )

        if source_path.startswith(('http://', 'https://')):
            # Handle Git repository
            temp_dir = tempfile.mkdtemp()
            try:
                log_debug(f"Cloning repository: {source_path}")
                process = await asyncio.create_subprocess_exec(
                    'git', 'clone', source_path, temp_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                try:
                    await asyncio.wait_for(process.communicate(), timeout=300)
                    if process.returncode is None or process.returncode != 0:
                        raise subprocess.CalledProcessError(process.returncode or -1, ['git', 'clone'])
                    source_path = temp_dir
                except asyncio.TimeoutError:
                    log_error("Repository cloning timed out")
                    if process:
                        process.terminate()
                    return

            except Exception as e:
                log_error(f"Failed to clone repository: {str(e)}")
                return

        with cleanup_context(temp_dir):
            if os.path.isdir(source_path):
                # Process directory
                tasks = []
                for root, _, files in os.walk(source_path):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            task = asyncio.create_task(
                                process_file(file_path, args, client)
                            )
                            tasks.append(task)

                if tasks:
                    # Process all files with proper error handling
                    try:
                        await asyncio.gather(*tasks)
                    except asyncio.CancelledError:
                        log_error("Tasks cancelled - attempting graceful shutdown")
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        raise
                else:
                    log_warning(f"No Python files found in {source_path}")

            else:
                # Process single file
                await process_file(source_path, args, client)

    except asyncio.CancelledError:
        log_error("Workflow cancelled - performing cleanup")
        raise
    except Exception as e:
        log_error(f"Workflow error: {str(e)}")
        raise
    finally:
        if client:
            await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='DocStrings Workflow System with Azure OpenAI'
    )

    parser.add_argument(
        'source_path',
        help='Path to Python source file, directory, or Git repository'
    )
    parser.add_argument(
        '--endpoint',
        default='https://your-azure-openai-endpoint.openai.azure.com/',
        help='Azure OpenAI endpoint'
    )
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory for documentation and updated source'
    )
    parser.add_argument(
        '--redis-host',
        default='localhost',
        help='Redis server host'
    )
    parser.add_argument(
        '--redis-port',
        type=int,
        default=6379,
        help='Redis server port'
    )
    parser.add_argument(
        '--redis-db',
        type=int,
        default=0,
        help='Redis database number'
    )
    parser.add_argument(
        '--redis-password',
        help='Redis server password'
    )
    parser.add_argument(
        '--cache-ttl',
        type=int,
        default=86400,
        help='Cache TTL in seconds'
    )

    args = parser.parse_args()

    try:
        # Set up event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(run_workflow(args))
        except (asyncio.CancelledError, KeyboardInterrupt):
            log_error("Program interrupted")
        except Exception as e:
            log_error(f"Program error: {e}")
        finally:
            # Clean up pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()

            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()

    finally:
        log_info("Program shutdown complete")
