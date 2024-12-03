"""
Response parsing service with consistent error handling and validation.
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from jsonschema import validate, ValidationError
from core.logger import LoggerSetup
from core.schema_loader import load_schema
from core.docstring_processor import DocstringProcessor
from core.types import ParsedResponse, DocstringData
from exceptions import ValidationError as CustomValidationError

logger = LoggerSetup.get_logger(__name__)

class ResponseParsingService:
    """Centralized service for parsing and validating AI responses."""

    def __init__(self):
        """Initialize the response parsing service."""
        self.logger = LoggerSetup.get_logger(__name__)
        self.docstring_processor = DocstringProcessor()
        self.docstring_schema = load_schema('docstring_schema')
        self.function_schema = load_schema('function_tools_schema')
        self._parsing_stats = {
            'total_processed': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'validation_failures': 0
        }

    async def parse_response(
        self, 
        response: str,
        expected_format: str = 'json',
        max_retries: int = 3,
        validate_schema: bool = True
    ) -> ParsedResponse:
        """
        Parse and validate an AI response.

        Args:
            response: Raw response string to parse
            expected_format: Expected format ('json', 'markdown', 'docstring')
            max_retries: Maximum number of parsing attempts
            validate_schema: Whether to validate against schema

        Returns:
            ParsedResponse: Structured response data with metadata

        Raises:
            CustomValidationError: If validation fails
        """
        start_time = datetime.now()
        errors = []
        parsed_content = None

        self._parsing_stats['total_processed'] += 1

        try:
            # Try parsing with retries
            for attempt in range(max_retries):
                try:
                    if expected_format == 'json':
                        parsed_content = await self._parse_json_response(response)
                    elif expected_format == 'markdown':
                        parsed_content = await self._parse_markdown_response(response)
                    elif expected_format == 'docstring':
                        parsed_content = await self._parse_docstring_response(response)
                    else:
                        raise ValueError(f"Unsupported format: {expected_format}")

                    if parsed_content:
                        break

                except Exception as e:
                    errors.append(f"Parsing attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    continue

            # Validate if requested and content was parsed
            validation_success = False
            if parsed_content and validate_schema:
                validation_success = await self._validate_response(
                    parsed_content, 
                    expected_format
                )
                if not validation_success:
                    errors.append("Schema validation failed")
                    self._parsing_stats['validation_failures'] += 1
                    # Provide default values if validation fails
                    parsed_content = self._create_fallback_response()

            # Update success/failure stats
            if parsed_content:
                self._parsing_stats['successful_parses'] += 1
            else:
                self._parsing_stats['failed_parses'] += 1
                parsed_content = self._create_fallback_response()

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            return ParsedResponse(
                content=parsed_content,
                format_type=expected_format,
                parsing_time=processing_time,
                validation_success=validation_success,
                errors=errors,
                metadata={
                    'attempts': len(errors) + 1,
                    'timestamp': datetime.now().isoformat(),
                    'response_size': len(response)
                }
            )

        except Exception as e:
            self.logger.error(f"Response parsing failed: {e}")
            self._parsing_stats['failed_parses'] += 1
            raise CustomValidationError(f"Failed to parse response: {str(e)}")

    async def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response, handling code blocks and cleaning."""
        try:
            # Clean up response
            response = response.strip()
            
            # Extract JSON from code blocks if present
            if '```json' in response and '```' in response:
                start = response.find('```json') + 7
                end = response.rfind('```')
                if start > 7 and end > start:
                    response = response[start:end].strip()
            
            # Remove any non-JSON content
            if not response.startswith('{') and not response.endswith('}'):
                start = response.find('{')
                end = response.rfind('}')
                if start >= 0 and end >= 0:
                    response = response[start:end+1]

            parsed_content = json.loads(response.strip())
            
            # Ensure the parsed content has the expected structure
            required_fields = {'summary', 'description', 'args', 'returns', 'raises'}
            if not all(field in parsed_content for field in required_fields):
                self.logger.warning("Parsed JSON missing required fields")
                # Add missing fields with default values
                parsed_content.update({
                    field: parsed_content.get(field, [] if field in {'args', 'raises'} else 
                                            {'type': 'Any', 'description': ''} if field == 'returns' else '')
                    for field in required_fields
                })

            return parsed_content

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}")
            return None

    async def _parse_markdown_response(self, response: str) -> Dict[str, Any]:
        """Parse markdown response into structured format."""
        sections = {
            'summary': '',
            'description': '',
            'args': [],
            'returns': {'type': 'Any', 'description': ''},
            'raises': []
        }

        current_section = None
        section_content = []

        for line in response.split('\n'):
            line = line.strip()
            
            if line.lower().startswith('# '):
                sections['summary'] = line[2:].strip()
            elif line.lower().startswith('## arguments') or line.lower().startswith('## args'):
                current_section = 'args'
                section_content = []
            elif line.lower().startswith('## returns'):
                if section_content and current_section == 'args':
                    sections['args'].extend(self._parse_arg_section(section_content))
                current_section = 'returns'
                section_content = []
            elif line.lower().startswith('## raises'):
                if current_section == 'returns':
                    sections['returns'] = self._parse_return_section(section_content)
                current_section = 'raises'
                section_content = []
            elif line:
                section_content.append(line)

        # Process final section
        if section_content:
            if current_section == 'args':
                sections['args'].extend(self._parse_arg_section(section_content))
            elif current_section == 'returns':
                sections['returns'] = self._parse_return_section(section_content)
            elif current_section == 'raises':
                sections['raises'].extend(self._parse_raises_section(section_content))

        return sections

    async def _parse_docstring_response(self, response: str) -> Dict[str, Any]:
        """Parse docstring response using DocstringProcessor."""
        docstring_data = self.docstring_processor.parse(response)
        return {
            'summary': docstring_data.summary,
            'description': docstring_data.description,
            'args': docstring_data.args,
            'returns': docstring_data.returns,
            'raises': docstring_data.raises,
            'complexity': docstring_data.complexity
        }

    def _parse_arg_section(self, lines: List[str]) -> List[Dict[str, str]]:
        """Parse argument section content."""
        args = []
        current_arg = None

        for line in lines:
            if line.startswith('- ') or line.startswith('* '):
                if current_arg:
                    args.append(current_arg)
                parts = line[2:].split(':')
                if len(parts) >= 2:
                    name = parts[0].strip()
                    description = ':'.join(parts[1:]).strip()
                    current_arg = {
                        'name': name,
                        'type': self._extract_type(name),
                        'description': description
                    }
            elif current_arg and line:
                current_arg['description'] += ' ' + line

        if current_arg:
            args.append(current_arg)

        return args

    def _parse_return_section(self, lines: List[str]) -> Dict[str, str]:
        """Parse return section content."""
        if not lines:
            return {'type': 'None', 'description': ''}

        return_text = ' '.join(lines)
        if ':' in return_text:
            type_str, description = return_text.split(':', 1)
            return {
                'type': type_str.strip(),
                'description': description.strip()
            }
        return {
            'type': 'Any',
            'description': return_text.strip()
        }

    def _parse_raises_section(self, lines: List[str]) -> List[Dict[str, str]]:
        """Parse raises section content."""
        raises = []
        current_exception = None

        for line in lines:
            if line.startswith('- ') or line.startswith('* '):
                if current_exception:
                    raises.append(current_exception)
                parts = line[2:].split(':')
                if len(parts) >= 2:
                    exception = parts[0].strip()
                    description = ':'.join(parts[1:]).strip()
                    current_exception = {
                        'exception': exception,
                        'description': description
                    }
            elif current_exception and line:
                current_exception['description'] += ' ' + line

        if current_exception:
            raises.append(current_exception)

        return raises

    def _extract_type(self, text: str) -> str:
        """Extract type hints from text."""
        if '(' in text and ')' in text:
            type_hint = text[text.find('(') + 1:text.find(')')]
            return type_hint
        return 'Any'

    def _create_fallback_response(self) -> Dict[str, Any]:
        """Create a fallback response when parsing fails."""
        return {
            'summary': 'AI-generated documentation not available',
            'description': 'Documentation could not be generated by AI service',
            'args': [],
            'returns': {
                'type': 'Any',
                'description': 'Return value not documented'
            },
            'raises': [],
            'complexity': 1
        }

    async def _validate_response(
        self,
        content: Dict[str, Any],
        format_type: str
    ) -> bool:
        """Validate response against appropriate schema."""
        try:
            if format_type == 'docstring':
                # Ensure the schema includes the required keys
                schema = self.docstring_schema['schema']
                required_keys = {'summary', 'description', 'args', 'returns', 'raises'}
                if not all(key in schema['properties'] for key in required_keys):
                    raise CustomValidationError("Schema does not include all required keys")

                validate(instance=content, schema=schema)
            elif format_type == 'function':
                validate(instance=content, schema=self.function_schema['schema'])
            return True
        except ValidationError as e:
            self.logger.error(f"Schema validation failed: {e}")
            return False

    def get_stats(self) -> Dict[str, int]:
        """Get current parsing statistics."""
        return self._parsing_stats.copy()

    async def __aenter__(self) -> 'ResponseParsingService':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass  # Cleanup if needed