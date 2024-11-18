"""
Docstring Utilities Module

This module provides utilities for parsing and validating docstrings, including schema validation
and completeness checks. It ensures that docstrings conform to a specified structure and contain
necessary information about function parameters, return values, and exceptions.

Version: 1.0.0
Author: Development Team
"""

from typing import Dict, List, Any, Optional, Tuple
from jsonschema import validate, ValidationError, Draft7Validator
import ast
import re
import logging
from extract.extraction_manager import ExtractionManager  # Ensure this import is added

# Set up module-level logger
logger = logging.getLogger(__name__)

# Enhanced JSON schema for comprehensive docstring validation
DOCSTRING_SCHEMA = {
    "type": "object",
    "properties": {
        "docstring": {
            "type": "string",
            "minLength": 10,
            "description": "The complete generated docstring."
        },
        "summary": {
            "type": "string",
            "minLength": 10,
            "maxLength": 1000,
            "description": "A concise summary of the function/class/module."
        },
        "description": {
            "type": "string",
            "minLength": 20,
            "description": "Detailed description of the functionality."
        },
        "parameters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$"},
                    "type": {"type": "string"},
                    "description": {"type": "string", "minLength": 10},
                    "optional": {"type": "boolean"},
                    "default": {"type": ["string", "number", "boolean", "null"]}
                },
                "required": ["name", "type", "description"]
            }
        },
        "returns": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "description": {"type": "string", "minLength": 10}
            },
            "required": ["type", "description"]
        },
        "raises": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "exception": {"type": "string"},
                    "description": {"type": "string", "minLength": 10}
                },
                "required": ["exception", "description"]
            }
        },
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["code"]
            }
        },
        "notes": {
            "type": "array",
            "items": {"type": "string", "minLength": 10}
        },
        "version": {
            "type": "string",
            "pattern": "^\\d+\\.\\d+\\.\\d+$"
        }
    },
    "required": ["docstring", "summary", "parameters", "returns"]
}

class DocstringValidator:
    """Validates and processes docstrings with comprehensive checks."""

    def __init__(self):
        """Initialize the validator with schema."""
        self.validator = Draft7Validator(DOCSTRING_SCHEMA)
        self.type_pattern = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*($[^$]+$)?)(,\s*[A-Za-z_][A-Za-z0-9_]*($[^$]+$)?)*$')

    def validate_docstring(self, docstring_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a docstring against the schema with detailed error reporting.

        Args:
            docstring_data (Dict[str, Any]): The structured docstring data to validate.

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        try:
            self.validator.validate(docstring_data)
            
            # Additional validation checks
            errors.extend(self._validate_parameters(docstring_data.get('parameters', [])))
            errors.extend(self._validate_return_type(docstring_data.get('returns', {})))
            errors.extend(self._validate_examples(docstring_data.get('examples', [])))
            
            if 'raises' in docstring_data:
                errors.extend(self._validate_exceptions(docstring_data['raises']))

            return len(errors) == 0, errors

        except ValidationError as e:
            error_path = ' -> '.join(str(p) for p in e.path)
            errors.append(f"Schema validation error at {error_path}: {e.message}")
            return False, errors
        except Exception as e:
            errors.append(f"Unexpected validation error: {str(e)}")
            return False, errors

    def _validate_parameters(self, parameters: List[Dict[str, Any]]) -> List[str]:
        """
        Validate parameter specifications.

        Args:
            parameters (List[Dict[str, Any]]): List of parameter specifications.

        Returns:
            List[str]: List of validation errors.
        """
        errors = []
        param_names = set()

        for param in parameters:
            # Check for duplicate parameter names
            if param['name'] in param_names:
                errors.append(f"Duplicate parameter name: {param['name']}")
            param_names.add(param['name'])

            # Validate parameter type syntax
            if not self.type_pattern.match(param['type']):
                errors.append(f"Invalid type format for parameter {param['name']}: {param['type']}")

            # Check description quality
            if len(param['description'].split()) < 3:
                errors.append(f"Description too brief for parameter {param['name']}")

            # Validate default value if present
            if 'default' in param and param['type'] != self._infer_type(param['default']):
                errors.append(
                    f"Default value type mismatch for {param['name']}: "
                    f"expected {param['type']}, got {self._infer_type(param['default'])}"
                )

        return errors

    def _validate_return_type(self, returns: Dict[str, Any]) -> List[str]:
        """
        Validate return type specification.

        Args:
            returns (Dict[str, Any]): Return type specification.

        Returns:
            List[str]: List of validation errors.
        """
        errors = []
        
        if not returns:
            return ["Missing return type specification"]

        if not self.type_pattern.match(returns['type']):
            errors.append(f"Invalid return type format: {returns['type']}")

        if len(returns['description'].split()) < 3:
            errors.append("Return description too brief")

        return errors

    def _validate_exceptions(self, exceptions: List[Dict[str, Any]]) -> List[str]:
        """
        Validate exception specifications.

        Args:
            exceptions (List[Dict[str, Any]]): List of exception specifications.

        Returns:
            List[str]: List of validation errors.
        """
        errors = []
        exception_names = set()

        for exc in exceptions:
            # Check for duplicate exceptions
            if exc['exception'] in exception_names:
                errors.append(f"Duplicate exception: {exc['exception']}")
            exception_names.add(exc['exception'])

            # Validate exception name format
            if not exc['exception'].endswith('Error') and not exc['exception'].endswith('Exception'):
                errors.append(f"Invalid exception name format: {exc['exception']}")

            # Check description quality
            if len(exc['description'].split()) < 3:
                errors.append(f"Description too brief for exception {exc['exception']}")

        return errors

    def _validate_examples(self, examples: List[Dict[str, Any]]) -> List[str]:
        """
        Validate code examples.

        Args:
            examples (List[Dict[str, Any]]): List of code examples.

        Returns:
            List[str]: List of validation errors.
        """
        errors = []

        for i, example in enumerate(examples, 1):
            try:
                # Check if code is valid Python
                ast.parse(example['code'])
            except SyntaxError as e:
                errors.append(f"Invalid Python syntax in example {i}: {str(e)}")
            except Exception as e:
                errors.append(f"Error validating example {i}: {str(e)}")

            # Check example quality
            if len(example['code'].strip().split('\n')) < 2:
                errors.append(f"Example {i} too brief (should be at least 2 lines)")

            if 'description' in example and len(example['description'].split()) < 3:
                errors.append(f"Description too brief for example {i}")

        return errors

    def _infer_type(self, value: Any) -> str:
        """
        Infer the type of a value.

        Args:
            value: Value to infer type from.

        Returns:
            str: Inferred type name.
        """
        if value is None:
            return "None"
        return value.__class__.__name__

def validate_and_fix_docstring(docstring: str) -> Tuple[str, List[str]]:
    """
    Validate and attempt to fix common docstring issues.

    Args:
        docstring (str): The docstring to validate and fix.

    Returns:
        Tuple[str, List[str]]: (fixed_docstring, list_of_changes)
    """
    changes = []
    fixed_docstring = docstring

    # Fix common formatting issues
    fixes = [
        (r'\n\s*\n\s*\n', '\n\n'),  # Fix multiple blank lines
        (r'^(\s*[A-Za-z]+:)\s*([A-Z])', r'\1\n    \2'),  # Fix section headers
        (r'(?<=\n)(\s{4,})', '    '),  # Normalize indentation
        (r'"""[\s\n]*', '"""'),  # Fix opening quotes
        (r'[\s\n]*"""$', '"""'),  # Fix closing quotes
    ]

    for pattern, replacement in fixes:
        new_docstring = re.sub(pattern, replacement, fixed_docstring)
        if new_docstring != fixed_docstring:
            changes.append(f"Fixed pattern: {pattern}")
            fixed_docstring = new_docstring

    # Ensure sections are properly formatted
    sections = ['Args:', 'Returns:', 'Raises:', 'Examples:', 'Notes:']
    for section in sections:
        if section in fixed_docstring and not re.search(f"\n{section}", fixed_docstring):
            fixed_docstring = fixed_docstring.replace(section, f"\n{section}")
            changes.append(f"Fixed section formatting: {section}")

    return fixed_docstring, changes

def parse_and_validate_docstring(docstring: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Parse and validate a docstring comprehensively.

    Args:
        docstring (str): The docstring to parse and validate.

    Returns:
        Tuple[Optional[Dict[str, Any]], List[str]]: (parsed_docstring, validation_errors)
    """
    if not docstring:
        return None, ["Empty docstring"]

    try:
        # First fix any formatting issues
        fixed_docstring, fix_changes = validate_and_fix_docstring(docstring)
        
        # Parse the fixed docstring
        parsed_data = parse_docstring(fixed_docstring)
        if not parsed_data:
            return None, ["Failed to parse docstring"]

        # Validate the parsed data
        validator = DocstringValidator()
        is_valid, validation_errors = validator.validate_docstring(parsed_data)

        if fix_changes:
            validation_errors.extend([f"Applied fix: {change}" for change in fix_changes])

        return parsed_data if is_valid else None, validation_errors

    except Exception as e:
        return None, [f"Error processing docstring: {str(e)}"]

def parse_docstring(docstring: str) -> Dict[str, Any]:
    """Parse a docstring into structured sections.

    Args:
        docstring (str): The docstring to parse.

    Returns:
        Dict[str, Any]: A dictionary with parsed sections.
    """
    if not docstring:
        return {}

    sections = {}
    current_section = 'Description'
    sections[current_section] = []

    for line in docstring.split('\n'):
        line = line.strip()
        if line.endswith(':') and line[:-1] in ['Args', 'Returns', 'Raises', 'Yields', 'Examples', 'Attributes']:
            current_section = line[:-1]
            sections[current_section] = []
        else:
            sections[current_section].append(line)

    for key in sections:
        sections[key] = '\n'.join(sections[key]).strip()

    return sections

def analyze_code_element_docstring(node: ast.AST) -> List[str]:
    """Analyze the docstring of a code element for completeness.

    Args:
        node (ast.AST): The AST node representing the code element.

    Returns:
        List[str]: A list of issues found in the docstring.
    """
    issues = []
    docstring = ast.get_docstring(node)
    if not docstring:
        issues.append("Missing docstring.")
        return issues

    parsed_docstring = parse_docstring(docstring)
    if not validate_docstring(parsed_docstring):
        issues.append("Docstring does not conform to schema.")

    # Additional checks for completeness
    issues.extend(check_parameter_descriptions(parsed_docstring, node))
    issues.extend(check_return_description(parsed_docstring, node))
    issues.extend(check_exception_details(parsed_docstring, node))

    return issues

def _extract_documented_args(args_section: str) -> List[str]:
    """Extract parameter names from the Args section.

    Args:
        args_section (str): The Args section of the docstring.

    Returns:
        List[str]: A list of documented argument names.
    """
    documented_args = []
    for line in args_section.split('\n'):
        if ':' in line:
            arg_name = line.split(':')[0].strip()
            documented_args.append(arg_name)
    return documented_args

def check_parameter_descriptions(docstring_data: Dict[str, Any], function_node: ast.FunctionDef) -> List[str]:
    """Check for the presence and quality of parameter descriptions.

    Args:
        docstring_data (Dict[str, Any]): The structured docstring data.
        function_node (ast.FunctionDef): The function node to analyze.

    Returns:
        List[str]: A list of issues found with parameter descriptions.
    """
    issues = []
    args_section = docstring_data.get('Args', '')
    
    if isinstance(args_section, str):
        documented_params = _extract_documented_args(args_section)
    else:
        documented_params = args_section.keys()

    function_params = [arg.arg for arg in function_node.args.args]

    for param in function_params:
        if param not in documented_params:
            issues.append(f"Parameter '{param}' is not documented.")
        elif isinstance(args_section, dict):
            description = args_section.get(param, '')
            if len(description) < 10:  # Example threshold for quality
                issues.append(f"Description for parameter '{param}' is too short.")

    return issues

def check_return_description(docstring_data: Dict[str, Any], function_node: ast.FunctionDef) -> List[str]:
    """Check for the presence and quality of return value descriptions.

    Args:
        docstring_data (Dict[str, Any]): The structured docstring data.
        function_node (ast.FunctionDef): The function node to analyze.

    Returns:
        List[str]: A list of issues found with return descriptions.
    """
    issues = []
    if function_node.returns and 'Returns' not in docstring_data:
        issues.append("Missing 'Returns' section in docstring.")
    elif 'Returns' in docstring_data:
        description = docstring_data['Returns']
        if len(description) < 10:  # Example threshold for quality
            issues.append("Return description is too short.")

    return issues

def check_exception_details(docstring_data: Dict[str, Any], function_node: ast.FunctionDef, extraction_manager: Optional[ExtractionManager] = None) -> List[str]:
    """Enhanced exception checking with extraction manager integration.

    Args:
        docstring_data (Dict[str, Any]): The structured docstring data.
        function_node (ast.FunctionDef): The function node to analyze.
        extraction_manager (Optional[ExtractionManager]): An optional extraction manager for detailed exception info.

    Returns:
        List[str]: A list of issues found with exception details.
    """
    issues = []
    raises_exceptions = any(isinstance(node, ast.Raise) for node in ast.walk(function_node))
    
    if extraction_manager:
        # Use extraction manager to get detailed exception info
        detected_exceptions = extraction_manager._detect_exceptions(function_node)
        documented_exceptions = set(
            exc['exception'] for exc in docstring_data.get('raises', [])
        )
        
        # Check for undocumented exceptions
        for exc in detected_exceptions:
            if exc not in documented_exceptions:
                issues.append(f"Detected exception '{exc}' is not documented")

    # Existing checks
    if raises_exceptions and 'Raises' not in docstring_data:
        issues.append("Missing 'Raises' section in docstring.")
    elif 'Raises' in docstring_data:
        for exception in docstring_data['Raises']:
            if len(exception['description']) < 10:
                issues.append(
                    f"Description for exception '{exception['exception']}' is too short"
                )

    return issues
