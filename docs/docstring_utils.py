# docstring_utils.py
"""
Docstring Utilities Module

Provides utilities for parsing, validating, and formatting docstrings.
Acts as a facade for the core docstring processor.
"""

from typing import Dict, List, Any, Tuple, Optional
import re
from deprecated import deprecated

from core.logger import LoggerSetup
from core.docstring_processor import DocstringProcessor, DocstringData

logger = LoggerSetup.get_logger(__name__)

class DocstringUtils:
    """Utility class for docstring operations."""

    def __init__(self):
        """Initialize DocstringUtils with a processor instance."""
        self.processor = DocstringProcessor()

    def parse_docstring(self, docstring: str) -> DocstringData:
        """
        Parse a docstring into structured format.

        Args:
            docstring: Raw docstring text

        Returns:
            DocstringData: Structured docstring data
        """
        return self.processor.parse(docstring)

    def validate_docstring(
        self,
        docstring_data: DocstringData
    ) -> Tuple[bool, List[str]]:
        """
        Validate docstring content and structure.

        Args:
            docstring_data: Structured docstring data

        Returns:
            Tuple containing validation status and error messages
        """
        return self.processor.validate(docstring_data)

    def format_docstring(self, docstring_data: DocstringData) -> str:
        """
        Format structured docstring data into string.

        Args:
            docstring_data: Structured docstring data

        Returns:
            str: Formatted docstring
        """
        return self.processor.format(docstring_data)

    @staticmethod
    def extract_type_hints(docstring: str) -> Dict[str, str]:
        """
        Extract type hints from docstring.

        Args:
            docstring: Raw docstring text

        Returns:
            Dict mapping parameter names to their type hints
        """
        type_hints = {}
        pattern = r':param\s+(\w+):\s*$([^)]+)$'
        matches = re.finditer(pattern, docstring)
        
        for match in matches:
            param_name, type_hint = match.groups()
            type_hints[param_name] = type_hint.strip()
            
        return type_hints

    @staticmethod
    def extract_return_type(docstring: str) -> Optional[str]:
        """
        Extract return type from docstring.

        Args:
            docstring: Raw docstring text

        Returns:
            Optional[str]: Return type if found
        """
        pattern = r':return:\s*$([^)]+)$'
        match = re.search(pattern, docstring)
        return match.group(1).strip() if match else None

# Deprecated functions for backward compatibility
@deprecated(reason="Use DocstringUtils class instead")
def parse_docstring(docstring: str) -> Dict[str, Any]:
    """Deprecated: Use DocstringUtils.parse_docstring instead."""
    return DocstringUtils().parse_docstring(docstring).__dict__

@deprecated(reason="Use DocstringUtils class instead")
def validate_docstring(docstring_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Deprecated: Use DocstringUtils.validate_docstring instead."""
    return DocstringUtils().validate_docstring(DocstringData(**docstring_data))