import pytest
import ast
from unittest.mock import patch, AsyncMock, MagicMock
from code_extractor import extract_metadata_from_file, FunctionMetadata, ModuleMetadata
from prompt_generator import generate_function_prompt
from ai_client import get_enriched_docstring
from response_parser import parse_and_validate_response, DocstringSchema
from docstring_integrator import integrate_docstring_into_file
from project_context import build_project_context, summarize_global_context_for_function
import os

# Test data
TEST_FILE_PATH = "test_module.py"
TEST_FUNCTION_NAME = "test_function"

# Sample Python file content for testing
TEST_FILE_CONTENT = """
def test_function(param1: int, param2: str) -> bool:
    \"\"\"A simple test function.\"\"\"
    return True
"""

@pytest.fixture
def setup_test_file(tmp_path):
    """Fixture to create a temporary Python file for testing."""
    test_file = tmp_path / TEST_FILE_PATH
    test_file.write_text(TEST_FILE_CONTENT)
    return test_file

def test_code_extractor(setup_test_file):
    """Test the code_extractor module for correct metadata extraction."""
    result = extract_metadata_from_file(str(setup_test_file))
    assert len(result.functions) == 1
    function_meta = result.functions[0]
    assert function_meta.name == TEST_FUNCTION_NAME
    assert function_meta.args == [{'name': 'param1', 'type': 'int'}, {'name': 'param2', 'type': 'str'}]
    assert function_meta.return_type == 'bool'
    assert function_meta.docstring == "A simple test function."

def test_prompt_generator():
    """Test prompt_generator for correct prompt formatting."""
    function_meta = FunctionMetadata(
        name=TEST_FUNCTION_NAME,
        args=[{'name': 'param1', 'type': 'int'}, {'name': 'param2', 'type': 'str'}],
        return_type='bool',
        decorators=[],
        docstring="A simple test function."
    )
    module_meta = ModuleMetadata(name="test_module", docstring="Test module.")
    prompt = generate_function_prompt(function_meta, module_meta)
    assert "Function Name: test_function" in prompt
    assert "Signature: test_function(param1, param2)" in prompt
    assert "Description: A simple test function." in prompt

@patch('ai_client.client.chat.completions.create', new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_ai_client(mock_create):
    """Test ai_client for handling API responses and errors."""
    # Mock a successful API response
    mock_create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="{}"))])
    response = await get_enriched_docstring("test prompt")
    assert response == "{}"

    # Test error handling by raising an exception
    mock_create.side_effect = Exception("API error")
    response = await get_enriched_docstring("test prompt")
    assert "An error occurred with the Azure OpenAI API." in response

def test_response_parser():
    """Test response_parser for parsing and validation of JSON responses."""
    valid_json = """
    {
        "description": "A simple test function.",
        "args": [{"name": "param1", "type": "int", "description": "First parameter."}],
        "returns": "bool",
        "raises": [],
        "examples": "test_function(1, 'example')"
    }
    """
    docstring_schema = parse_and_validate_response(valid_json)
    assert isinstance(docstring_schema, DocstringSchema)

    invalid_json = "{ invalid json }"
    with pytest.raises(ValueError):
        parse_and_validate_response(invalid_json)

def test_docstring_integrator(setup_test_file):
    """Test docstring_integrator for correct docstring insertion."""
    docstring_data = DocstringSchema(
        description="A simple test function.",
        args=[{"name": "param1", "type": "int", "description": "First parameter."}],
        returns="bool",
        raises=[],
        examples="test_function(1, 'example')"
    )
    integrate_docstring_into_file(str(setup_test_file), docstring_data, TEST_FUNCTION_NAME)

    with open(setup_test_file, 'r') as f:
        updated_content = f.read()

    assert '"""A simple test function.' in updated_content

def test_project_context(tmp_path):
    """Test project_context for building context and dependency graph."""
    # Create a small set of files for testing
    file1 = tmp_path / "module1.py"
    file1.write_text("import module2\n\ndef func1(): pass\n")

    file2 = tmp_path / "module2.py"
    file2.write_text("def func2(): pass\n")

    context, graph = build_project_context(str(tmp_path))
    assert "module1" in context
    assert "module2" in context
    assert ("module1", "module2") in graph.edges()

    function_context = summarize_global_context_for_function("func1", context)
    assert "module1" in function_context['modules']
    assert "func1" in function_context['related_functions']

# Run the tests
if __name__ == "__main__":
    pytest.main()