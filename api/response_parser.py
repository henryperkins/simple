# response_parser.py (existing implementation)
import json
from typing import Optional, Dict, Any
from jsonschema import validate, ValidationError
from core.logger import log_info, log_error, log_debug

# Existing JSON schema
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
    """Parses and validates responses from Azure OpenAI API."""

    def __init__(self, token_manager: Optional['TokenManager'] = None):
        """Initialize the ResponseParser with an optional TokenManager."""
        self.token_manager = token_manager

    def parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse the Azure OpenAI response."""
        log_debug("Parsing JSON response.")
        try:
            # Track token usage if token manager is available
            if self.token_manager:
                tokens = self.token_manager.estimate_tokens(response)
                self.token_manager.track_request(0, tokens)

            # Handle both string and dict inputs
            if isinstance(response, dict):
                response_json = response
            else:
                response = response.strip()
                if response.startswith('```') and response.endswith('```'):
                    response = response[3:-3].strip()
                if response.startswith('{'):
                    response_json = json.loads(response)
                else:
                    return self._parse_plain_text_response(response)

            # Validate against JSON schema
            validate(instance=response_json, schema=JSON_SCHEMA)
            log_debug("Response validated successfully against JSON schema.")

            return {
                "docstring": response_json["docstring"].strip(),
                "summary": response_json["summary"].strip(),
                "changelog": response_json.get("changelog", "Initial documentation").strip(),
                "complexity_score": response_json.get("complexity_score", 0)
            }

        except (json.JSONDecodeError, ValidationError) as e:
            log_error(f"Response parsing/validation error: {e}")
            log_debug(f"Invalid response content: {response}")
            return None

    def validate_response(self, response: Dict[str, Any]) -> bool:
        """Validate the response from the API."""
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
        """Fallback parser for plain text responses."""
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