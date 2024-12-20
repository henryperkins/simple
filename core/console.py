"""Console utilities for clean output formatting."""

from typing import Any, Optional
import logging
from rich.progress import Progress

# Configure basic logging - remove this as LoggerSetup handles it
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_live_layout() -> None:
    """Placeholder for setup_live_layout."""
    print("Live layout setup (placeholder)")

def stop_live_layout() -> None:
    """Placeholder for stop_live_layout."""
    print("Live layout stopped (placeholder)")

def format_error_output(error_message: str) -> str:
    """Format error messages for clean console output."""
    lines = error_message.split("\n")
    formatted_lines = []
    indent = "  "

    for line in lines:
        # Remove excessive whitespace
        line = " ".join(line.split())
        if line:
            formatted_lines.append(f"{indent}{line}")

    return "\n".join(formatted_lines)

def print_phase_header(title: str) -> None:
    """Print a section header with formatting."""
    print(f"--- {title} ---")  # Use consistent formatting for section headers

def print_error(message: str, correlation_id: str | None = None, details: dict[str, Any] | None = None) -> None:
    """Display formatted error messages."""
    print_section_break()
    print(f"🔥 Error: {message}")  # Highlight errors with a flame emoji
    if details:
        for key, value in details.items():
            print(f"  {key}: {value}")
    if correlation_id:
        print(f"  Correlation ID: {correlation_id}")
    print("Suggested Fix: Check the logs for more details or retry the operation.")
    if details:
        for key, value in details.items():
            print(f"  {key}: {value}")
    if correlation_id:
        print(f"  Correlation ID: {correlation_id}")
    print_section_break()

def print_status(message: str, details: dict[str, Any] | None = None) -> None:
    """Display formatted status messages with optional details."""
    print(f"Status: {message}")
    if details:
        for key, value in details.items():
            if isinstance(value, dict):  # Handle nested dictionaries
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")

def display_metrics(metrics: dict[str, Any], title: str = "Metrics") -> None:
    """Display metrics in a formatted table."""
    print(f"{title}:")
    print("+----------------------+----------------+")
    print(f"| {'Metric':<20} | {'Value':<14} |")
    print("+----------------------+----------------+")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"| {key:<20} | {value:>14.2f} |")
        else:
            print(f"| {key:<20} | {value:>14} |")
    print("+----------------------+----------------+")

def print_success(message: str, details: dict[str, Any] | None = None) -> None:
    """Display success messages."""
    print(f"✅ Success: {message}")  # Highlight successes with a checkmark emoji
    if details:
        for key, value in details.items():
            print(f"  {key}: {value}")

def print_info(message: str, details: Any = None) -> None:
    """Display info messages with optional details."""
    print(f"Info: {message}")
    if isinstance(details, dict):
        for key, value in details.items():
            print(f"  {key}: {value}")
    elif details is not None:
        print(format_error_output(str(details)))

def update_header(text: str) -> None:

    """Placeholder for update_header."""
    print(f"Header: {text}")

def update_footer(text: str) -> None:

    """Placeholder for update_footer."""
    print(f"Footer: {text}")

def update_left_panel(renderable: Any) -> None:

    """Placeholder for update_left_panel."""
    print(f"Left Panel: {renderable}")

def update_right_panel(renderable: Any) -> None:

    """Placeholder for update_right_panel."""
    print(f"Right Panel: {renderable}")

def display_progress(task_description: str) -> None:

    """Placeholder for display_progress."""
    print(f"Progress: {task_description}")

def display_code_snippet(

    code: str,
    language: str = "python",
    theme: str = "monokai",
    line_numbers: bool = True
) -> None:
    """Display a code snippet."""
    print(f"Code Snippet ({language}):\n{code}")

def print_warning(message: str) -> None:

    """Print a warning message."""
    print(f"⚠️ Warning: {message}")  # Highlight warnings with a warning emoji

def print_debug(message: str) -> None:

    """Print a debug message."""
    print(f"Debug: {message}")

def display_metrics_report(metrics: dict[str, Any]) -> None:

    """Display a formatted metrics report."""
    print("Metrics Report:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

def create_progress() -> Progress:

    """Create a progress object."""
    return Progress()

def print_phase_header(title: str) -> None:

    """Print a section header with formatting."""
    print(f"--- {title} ---")

def create_status_table(title: str, data: dict[str, Any]) -> None:

    """Create and display a status table."""
    print(f"{title}:")
    for key, value in data.items():
        print(f"  {key}: {value}")

def format_validation_status(success: bool, errors: list[str] | None = None) -> None:

    """Display validation status with optional errors."""
    status = "Passed" if success else "Failed"
    print(f"Validation Status: {status}")

    if not success and errors:
        for error in errors:
            print(f"  - {error}")

def display_processing_phase(title: str, content: dict[str, Any]) -> None:

    """Display a processing phase with formatted content."""
    print(f"--- {title} ---")
    for key, value in content.items():
        print(f"  {key}: {value}")

def display_api_metrics(response_data: dict[str, Any]) -> None:

    """Display API response metrics in a structured format."""
    print("API Response Metrics")
    for key, value in response_data.items():
        print(f"  {key}: {value}")

def display_validation_results(

    results: dict[str, bool],
    details: Optional[dict[str, Any]] = None
) -> None:
    """Display validation results with details."""
    print("Validation Results")
    for key, value in results.items():
        print(f"  {key}: {value} Details: {details.get(key, '') if details else ''}")

def display_progress_summary(summary: dict[str, Any]) -> None:

    """Display a summary of the processing progress."""
    print("Processing Summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")
def print_section_break() -> None:
    """Print a visual section break."""
    print("-" * 60)
