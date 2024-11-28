"""
Response Parser Module

Handles parsing and validation of Azure OpenAI API responses.
Ensures consistent and reliable output formatting.
"""

import json
from typing import Optional, Dict, Any
from jsonschema import validate, ValidationError

from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class ResponseParser:
    """Parses and validates Azure OpenAI API responses."""

    # Define response schema
    RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "documentation": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "description": {"type": "string"},
                    "parameters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["name", "type", "description"]
                        }
                    },
                    "returns": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["type", "description"]
                    }
                },
                "required": ["summary", "description"]
            }
        },
        "required": ["documentation"]
    }

    def __init__(self):
        """Initialize response parser."""
        self._validation_cache = {}
        logger.info("Response parser initialized")

    async def parse_response(
        self,
        response: str,
        expected_format: str = 'json'
    ) -> Optional[Dict[str, Any]]:
        """
        Parse and validate API response.

        Args:
            response: Raw API response
            expected_format: Expected response format ('json' or 'markdown')

        Returns:
            Optional[Dict[str, Any]]: Parsed response
        """
        try:
            if expected_format == 'json':
                return await self._parse_json_response(response)
            return await self._parse_markdown_response(response)
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None

    async def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response."""
        try:
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]

            # Parse JSON
            data = json.loads(response.strip())
            
            # Validate against schema
            if not self._validate_response(data):
                logger.error("Response validation failed")
                return None

            return data

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing JSON response: {e}")
            return None

    async def _parse_markdown_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse markdown response into structured format."""
        try:
            sections = self._split_markdown_sections(response)
            
            return {
                'documentation': {
                    'summary': sections.get('summary', ''),
                    'description': sections.get('description', ''),
                    'parameters': self._parse_parameters(sections.get('parameters', '')),
                    'returns': self._parse_returns(sections.get('returns', ''))
                }
            }

        except Exception as e:
            logger.error(f"Error parsing markdown response: {e}")
            return None

    def _validate_response(self, data: Dict[str, Any]) -> bool:
        """Validate response against schema."""
        try:
            validate(instance=data, schema=self.RESPONSE_SCHEMA)
            return True
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            return False

    def _split_markdown_sections(self, markdown: str) -> Dict[str, str]:
        """Split markdown into sections."""
        sections = {}
        current_section = 'description'
        current_content = []

        for line in markdown.split('\n'):
            if line.startswith('#'):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                    current_content = []

                # Update current section
                section_name = line.lstrip('#').strip().lower()
                current_section = section_name
            else:
                current_content.append(line)

        # Save final section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()

        return sections

    def _parse_parameters(self, params_text: str) -> List[Dict[str, str]]:
        """Parse parameter section from markdown."""
        params = []
        current_param = None

        for line in params_text.split('\n'):
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                # New parameter
                if ':' in line:
                    name, rest = line[2:].split(':', 1)
                    current_param = {
                        'name': name.strip(),
                        'type': 'Any',
                        'description': rest.strip()
                    }
                    params.append(current_param)
            elif current_param and line:
                # Continue previous parameter description
                current_param['description'] += ' ' + line

        return params

    def _parse_returns(self, returns_text: str) -> Dict[str, str]:
        """Parse returns section from markdown."""
        if ':' in returns_text:
            type_str, description = returns_text.split(':', 1)
            return {
                'type': type_str.strip(),
                'description': description.strip()
            }
        return {
            'type': 'None',
            'description': returns_text.strip() or 'No return value.'
        }