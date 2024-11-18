"""
Response Parser Module

This module provides functionality to parse and validate responses from Azure OpenAI,
focusing on extracting docstrings, summaries, and other metadata from API responses.

Version: 1.2.0
Author: Development Team
"""

import json
from typing import Optional, Dict, Any
from jsonschema import validate, ValidationError
from core.logger import log_info, log_error, log_debug

# Define JSON schema for API response validation
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "docstring": {
            "type": "string",
            "minLength": 1
        },
        "summary": {
            "type": "string",
            "minLength": 1
        },
        "changelog": {
            "type": "string"
        },
        "complexity_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100
        }
    },
    "required": ["docstring", "summary"],
    "additionalProperties": False
}

class ResponseParser:
    """
    Parses and validates responses from Azure OpenAI API.

    Methods:
        parse_json_response: Parses the Azure OpenAI response to extract generated docstring and related details.
        validate_response: Validates the response to ensure it contains required fields and proper content.
        _parse_plain_text_response: Fallback parser for plain text responses from Azure OpenAI.
    """

    def parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse the Azure OpenAI response to extract the generated docstring and related details.

        Args:
            response (str): The JSON response string to parse.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing parsed response data or None if parsing fails.
        """
        log_debug("Parsing JSON response.")
        try:
            response_json = json.loads(response)
            log_info("Successfully parsed Azure OpenAI response.")
            log_debug(f"Parsed JSON response: {response_json}")

            # Validate against JSON schema
            validate(instance=response_json, schema=JSON_SCHEMA)

            # Extract fields
            parsed_response = {
                "docstring": response_json["docstring"].strip(),
                "summary": response_json["summary"].strip(),
                "changelog": response_json.get("changelog", "Initial documentation").strip(),
                "complexity_score": response_json.get("complexity_score", 0)
            }

            return parsed_response

        except json.JSONDecodeError as e:
            log_error(f"Failed to parse response as JSON: {e}")
            return self._parse_plain_text_response(response)
        except ValidationError as e:
            log_error(f"Response validation error: {e.message}")
            log_error(f"Schema path: {' -> '.join(str(p) for p in e.schema_path)}")
            return None
        except Exception as e:
            log_error(f"Unexpected error during JSON response parsing: {e}")
            return None

    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate the response from the API to ensure it contains required fields and proper content.

        Args:
            response (Dict[str, Any]): The response from the API containing content and usage information.

        Returns:
            bool: True if the response is valid and contains all required fields with proper content.
        """
        try:
            if not isinstance(response, dict) or "content" not in response:
                log_error("Response missing basic structure")
                return False

            content = response["content"]

            # Validate required fields
            required_fields = ["docstring", "summary", "complexity_score", "changelog"]
            missing_fields = [field for field in required_fields if field not in content]
            if missing_fields:
                log_error(f"Response missing required fields: {missing_fields}")
                return False

            # Validate usage information if present
            if "usage" in response:
                usage = response["usage"]
                required_usage_fields = ["prompt_tokens", "completion_tokens", "total_tokens"]
                
                if not all(field in usage for field in required_usage_fields):
                    log_error("Missing usage information fields")
                    return False
                
                if not all(isinstance(usage[field], int) and usage[field] >= 0 
                        for field in required_usage_fields):
                    log_error("Invalid token count in usage information")
                    return False

                if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
                    log_error("Inconsistent token counts in usage information")
                    return False

            return True

        except Exception as e:
            log_error(f"Error during response validation: {e}")
            return False

    @staticmethod
    def _parse_plain_text_response(text: str) -> Optional[Dict[str, Any]]:
        """
        Fallback parser for plain text responses from Azure OpenAI.
        
        Args:
            text (str): The plain text response to parse.
            
        Returns:
            Optional[Dict[str, Any]]: Parsed response data or None if parsing fails.
        """
        log_debug("Attempting plain text response parsing.")
        try:
            lines = text.strip().split('\n')
            result = {
                "docstring": "",
                "summary": "",
                "changelog": "Initial documentation",
                "complexity_score": 0
            }
            current_key = None
            buffer = []

            for line in lines:
                line = line.strip()
                if line.endswith(':') and line[:-1].lower() in ['summary', 'changelog', 'docstring', 'complexity_score']:
                    if current_key and buffer:
                        content = '\n'.join(buffer).strip()
                        if current_key == 'complexity_score':
                            try:
                                result[current_key] = int(content)
                            except ValueError:
                                result[current_key] = 0
                        else:
                            result[current_key] = content
                    current_key = line[:-1].lower()
                    buffer = []
                elif current_key:
                    buffer.append(line)

            if current_key and buffer:
                content = '\n'.join(buffer).strip()
                if current_key == 'complexity_score':
                    try:
                        result[current_key] = int(content)
                    except ValueError:
                        result[current_key] = 0
                else:
                    result[current_key] = content

            return result if result["docstring"] and result["summary"] else None

        except Exception as e:
            log_error(f"Failed to parse plain text response: {e}")
            return None
