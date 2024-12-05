"""Docstring processing module."""

import json
from typing import Optional, Dict, Any, List, Union, Tuple
from docstring_parser import parse as parse_docstring
from core.logger import LoggerSetup
from core.metrics import Metrics
from core.types import DocstringData
from exceptions import DocumentationError
from utils import FormattingUtils, ValidationUtils

class DocstringProcessor:
    """Processes docstrings by parsing and validating them."""

    def __init__(self, metrics: Optional[Metrics] = None) -> None:
        """Initialize docstring processor."""
        self.logger = LoggerSetup.get_logger(__name__)
        self.metrics = metrics or Metrics()

    def parse(self, docstring: Union[Dict[str, Any], str]) -> DocstringData:
        """Parse a raw docstring into structured format."""
        try:
            if isinstance(docstring, dict):
                # Ensure 'returns' is a dictionary, providing a default if needed
                returns = docstring.get('returns')
                if not isinstance(returns, dict):
                    returns = {'type': 'Any', 'description': ''}
                    docstring['returns'] = returns

                return DocstringData(
                    summary=docstring.get('summary', ''),
                    description=docstring.get('description', ''),
                    args=docstring.get('args', []),
                    returns=returns,
                    raises=docstring.get('raises', []),
                    complexity=docstring.get('complexity', 1)
                )

            # If it's a string, try to parse as JSON
            if isinstance(docstring, str) and docstring.strip().startswith('{'):
                try:
                    doc_dict = json.loads(docstring)
                    return self.parse(doc_dict)
                except json.JSONDecodeError:
                    pass

            # Parse as a regular docstring string
            parsed = parse_docstring(docstring)
            return DocstringData(
                summary=parsed.short_description or '',
                description=parsed.long_description or '',
                args=[{
                    'name': param.arg_name,
                    'type': param.type_name or 'Any',
                    'description': param.description or ''
                } for param in parsed.params],
                returns={
                    'type': parsed.returns.type_name if parsed.returns else 'Any',
                    'description': parsed.returns.description if parsed.returns else ''
                },
                raises=[{
                    'exception': e.type_name or 'Exception',
                    'description': e.description or ''
                } for e in parsed.raises] if parsed.raises else []
            )

        except Exception as e:
            self.logger.error(f"Error parsing docstring: {e}")
            raise DocumentationError(f"Failed to parse docstring: {e}")

    def format(self, data: DocstringData) -> str:
        """Format structured docstring data into a string."""
        return FormattingUtils.format_docstring(data.__dict__)

    def validate(self, data: DocstringData) -> Tuple[bool, List[str]]:
        """Validate docstring data against requirements."""
        return ValidationUtils.validate_docstring(data.__dict__)