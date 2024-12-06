import json
import logging
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArgSchema(BaseModel):
    name: str
    type: Optional[str] = None
    description: str

class DocstringSchema(BaseModel):
    description: str
    args: List[ArgSchema] = []
    returns: Optional[str]
    raises: Optional[List[str]] = None
    examples: Optional[List[str]] = None


def clean_response(response_text: str) -> str:
    response_text = response_text.strip()
    # Remove any lines starting or ending with ```
    lines = response_text.split('\n')
    cleaned_lines = [line for line in lines if not line.strip().startswith('```')]
    # Ensure the cleaned response is valid JSON
    cleaned_response = "\n".join(cleaned_lines).strip()
    return cleaned_response

def parse_and_validate_response(response_text: str) -> DocstringSchema:
    cleaned_response = clean_response(response_text)
    if not cleaned_response:
        raise ValueError("Empty response from AI")

    try:
        response_data = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {e} - Response: {cleaned_response}")

    try:
        docstring_schema = DocstringSchema(**response_data)
    except ValidationError as e:
        raise ValueError(f"Validation Error: {e} - Response Data: {response_data}")
    
    return docstring_schema

# Example usage
if __name__ == "__main__":
    # Mock response for demonstration
    mock_response = """
    ```json
    {
        "description": "Calculates the average of a list of numbers.",
        "args": [{"name": "numbers", "type": "List[int]", "description": "A list of integers."}],
        "returns": "float",
        "raises": "ValueError if the list is empty",
        "examples": ["calculate_average([1, 2, 3, 4, 5])"]
    }
    ```
    """
    try:
        docstring = parse_and_validate_response(mock_response)
        print(docstring)
    except ValueError as e:
        print(e)
