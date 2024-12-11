"""Rich console utilities for enhanced visual feedback."""
from typing import Any
from rich.console import Console
from rich.progress import Progress
from rich.syntax import Syntax
from rich.logging import RichHandler
import logging

# Initialize rich console
console = Console()


def display_code_snippet(
    code: str,
    language: str = "python",
    theme: str = "monokai",
    line_numbers: bool = True
) -> None:
    """Display a code snippet with syntax highlighting.

    Args:
        code: The code string to display
        language: Programming language for syntax highlighting
        theme: Color theme to use
        line_numbers: Whether to show line numbers
    """
    syntax = Syntax(code, language, theme=theme, line_numbers=line_numbers)
    console.print(syntax)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging with rich handler and specified level.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


def print_status(message: str, style: str = "bold blue") -> None:
    """Print a status message with styling.

    Args:
        message: The message to display
        style: Rich style string for formatting
    """
    console.print(f"[{style}]{message}[/{style}]")


def print_error(message: str) -> None:
    """Print an error message in red.

    Args:
        message: The error message to display
    """
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_success(message: str) -> None:
    """Print a success message in green.

    Args:
        message: The success message to display
    """
    console.print(f"[bold green]Success:[/bold green] {message}")


def print_warning(message: str) -> None:
    """Print a warning message in yellow.

    Args:
        message: The warning message to display
    """
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message in blue.

    Args:
        message: The info message to display
    """
    console.print(f"[bold blue]Info:[/bold blue] {message}")


def print_debug(message: str) -> None:
    """Print a debug message in gray.

    Args:
        message: The debug message to display
    """
    console.print(f"[bold gray]Debug:[/bold gray] {message}")


def display_metrics(metrics: dict, title: str = "Metrics") -> None:
    """Display metrics in a formatted table."""
    console.print(f"[bold magenta]{title}[/bold magenta]")
    for key, value in metrics.items():
        console.print(f"[bold]{key}:[/bold] {value}")

def create_progress() -> Progress:
    """Create and return a Rich Progress instance."""
    return Progress()
if __name__ == "__main__":
    # Set up logging
    setup_logging(logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Example code snippet display
    code = '''
    def example_function(param: str) -> None:
        """Example function with syntax highlighting."""
        print(f"Parameter: {param}")
    '''
    display_code_snippet(code)

    # Example status messages
    print_info("Starting process...")
    print_status("Processing items", "bold cyan")
    print_warning("Some items were skipped")
    print_error("Failed to process item")
    print_success("Process completed successfully")
