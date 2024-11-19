"""
Docstring Utilities Module

Provides essential validation and parsing for Python docstrings.
Simplified but maintains core functionality needed by other components.
"""

from typing import Dict, List, Any, Tuple, Optional
import ast
import re
from core.logger import log_info, log_error, log_debug

class DocstringValidator:
    """Validates docstring content and structure."""
    
    def __init__(self):
        """Initialize the validator."""
        self.required_sections = ['summary', 'parameters', 'returns']
        self.min_length = {
            'summary': 10,
            'description': 10
        }

    def validate_docstring(self, docstring_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate docstring content and structure.

        Args:
            docstring_data: Dictionary containing docstring sections

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []

        # Check required sections
        for section in self.required_sections:
            if section not in docstring_data:
                errors.append(f"Missing required section: {section}")
                continue

        # Check content
        if 'summary' in docstring_data:
            if len(docstring_data['summary'].strip()) < self.min_length['summary']:
                errors.append("Summary too short (minimum 10 characters)")

        # Validate parameters if present
        if 'parameters' in docstring_data:
            param_errors = self._validate_parameters(docstring_data['parameters'])
            errors.extend(param_errors)

        # Validate return value
        if 'returns' in docstring_data:
            return_errors = self._validate_return(docstring_data['returns'])
            errors.extend(return_errors)

        is_valid = len(errors) == 0
        if not is_valid:
            log_error(f"Docstring validation failed: {errors}")
        
        return is_valid, errors

    def _validate_parameters(self, parameters: List[Dict[str, Any]]) -> List[str]:
        """
        Validate parameter documentation.

        Args:
            parameters: List of parameter dictionaries

        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        if not isinstance(parameters, list):
            return ["Parameters must be provided as a list"]

        for param in parameters:
            if not isinstance(param, dict):
                errors.append("Invalid parameter format")
                continue

            # Check required fields
            if 'name' not in param:
                errors.append("Parameter missing name")
                continue

            param_name = param.get('name', '')
            
            if 'type' not in param:
                errors.append(f"Parameter {param_name} missing type")

            if 'description' not in param:
                errors.append(f"Parameter {param_name} missing description")
            elif len(param.get('description', '').strip()) < self.min_length['description']:
                errors.append(f"Parameter {param_name} description too short")

        return errors

    def _validate_return(self, returns: Dict[str, Any]) -> List[str]:
        """
        Validate return value documentation.

        Args:
            returns: Return value documentation dictionary

        Returns:
            List[str]: List of validation errors
        """
        errors = []

        if not isinstance(returns, dict):
            return ["Return value must be provided as a dictionary"]

        if 'type' not in returns:
            errors.append("Return missing type")

        if 'description' not in returns:
            errors.append("Return missing description")
        elif len(returns.get('description', '').strip()) < self.min_length['description']:
            errors.append("Return description too short")

        return errors

def parse_docstring(docstring: str) -> Dict[str, Any]:
    """
    Parse a docstring into structured sections.

    Args:
        docstring: Raw docstring text

    Returns:
        Dict[str, Any]: Parsed docstring sections
    """
    if not docstring:
        return {
            "docstring": "",
            "summary": "",
            "parameters": [],
            "returns": {"type": "None", "description": "No return value."}
        }

    # Initialize structure
    sections = {
        "docstring": docstring.strip(),
        "summary": "",
        "parameters": [],
        "returns": {"type": "None", "description": "No return value."}
    }

    lines = docstring.split('\n')
    current_section = 'summary'
    current_content = []

    for line in lines:
        line = line.strip()
        
        # Check for section headers
        if line.lower().startswith(('args:', 'arguments:', 'parameters:', 'returns:', 'raises:')):
            # Save previous section content
            if current_content:
                if current_section == 'summary':
                    sections['summary'] = '\n'.join(current_content).strip()
                current_content = []

            # Update current section
            section_name = line.lower().split(':')[0]
            if section_name in ('args', 'arguments', 'parameters'):
                current_section = 'parameters'
            else:
                current_section = section_name

        # Add content to current section
        elif line:
            current_content.append(line)

    # Process final section
    if current_content:
        if current_section == 'summary':
            sections['summary'] = '\n'.join(current_content).strip()
        elif current_section == 'parameters':
            sections['parameters'] = _parse_parameters('\n'.join(current_content))
        elif current_section == 'returns':
            sections['returns'] = _parse_return('\n'.join(current_content))

    return sections

def _parse_parameters(params_str: str) -> List[Dict[str, Any]]:
    """Parse parameter section into structured format."""
    params = []
    current_param = None

    for line in params_str.split('\n'):
        line = line.strip()
        if not line:
            continue

        # New parameter definition
        if not line.startswith(' '):
            if ':' in line:
                name, rest = line.split(':', 1)
                current_param = {
                    "name": name.strip(),
                    "type": "Any",
                    "description": rest.strip()
                }
                params.append(current_param)
        # Parameter description continuation
        elif current_param:
            current_param["description"] = f"{current_param['description']} {line}"

    return params

def _parse_return(return_str: str) -> Dict[str, str]:
    """Parse return section into structured format."""
    return_info = {
        "type": "None",
        "description": "No return value."
    }

    if ':' in return_str:
        type_str, desc = return_str.split(':', 1)
        return_info.update({
            "type": type_str.strip(),
            "description": desc.strip()
        })
    else:
        return_info["description"] = return_str.strip()

    return return_info