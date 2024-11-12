# response_parser.py

import re
from typing import Any, Dict, List

from core.logger import LoggerSetup
from utils import format_response

logger = LoggerSetup.get_logger("response_parser")


def extract_section(text: str, section_name: str) -> str:
    """Extract a section from the response text."""
    pattern = rf"{section_name}:\s*(.*?)(?=\n\n|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_parameter_section(text: str) -> List[Dict[str, str]]:
    """Extract parameter information from the response text."""
    params_section = extract_section(text, "Parameters")
    params = []
    param_pattern = r"(\w+)\s*\(([^)]+)\):\s*(.+?)(?=\n\w+\s*\(|\Z)"
    for match in re.finditer(param_pattern, params_section, re.DOTALL):
        params.append({
            "name": match.group(1),
            "type": match.group(2).strip(),
            "description": match.group(3).strip()
        })
    return params


def extract_return_section(text: str) -> Dict[str, str]:
    """Extract return information from the response text."""
    returns_section = extract_section(text, "Returns")
    type_pattern = r"(\w+):\s*(.+)"
    match = re.search(type_pattern, returns_section)
    return {
        "type": match.group(1) if match else "None",
        "description": match.group(2).strip() if match else ""
    }


def extract_code_examples(text: str) -> List[str]:
    """Extract code examples from the response text."""
    examples_section = extract_section(text, "Examples")
    examples = []
    for match in re.finditer(r"```python\s*(.*?)\s*```", examples_section, re.DOTALL):
        examples.append(match.group(1).strip())
    return examples


class ClaudeResponseParser:
    """Handles response parsing and formatting."""

    @staticmethod
    def parse_function_analysis(response: str) -> Dict[str, Any]:
        """Parse the natural language response into structured format."""
        try:
            logger.debug(f"Parsing function analysis response: {response}")
            # Extract key sections from the response
            sections = {
                'summary': extract_section(response, 'Summary'),
                'params': extract_parameter_section(response),
                'returns': extract_return_section(response),
                'examples': extract_code_examples(response),
                'classes': [],  # Ensure classes is included
                'functions': []  # Ensure functions is included
            }

            # Validate and format response
            return format_response(sections)

        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return ClaudeResponseParser.get_default_response()

    @staticmethod
    def get_default_response() -> Dict[str, Any]:
        """Return a default response in case of parsing errors."""
        logger.debug("Returning default response due to parsing error")
        return {
            "summary": "Error parsing response",
            "docstring": "Error occurred while parsing the documentation.",
            "params": [],
            "returns": {"type": "None", "description": ""},
            "examples": [],
            "classes": [],
            "functions": []
        }