from code_extractor import FunctionMetadata, ModuleMetadata

def generate_function_prompt(function_meta: FunctionMetadata, module_meta: ModuleMetadata, global_context: dict = None) -> str:
    """
    Generate a prompt for an AI model to produce a Google-style docstring for a given function.

    Args:
        function_meta (FunctionMetadata): Metadata of the function for which to generate the docstring.
        module_meta (ModuleMetadata): Metadata of the module containing the function.
        global_context (dict, optional): Additional context that might be relevant for generating the docstring.

    Returns:
        str: A prompt string formatted for an AI model.
    """
    # Prepare the function signature
    signature = f"{function_meta.name}({', '.join([arg['name'] for arg in function_meta.args])})"

    # Prepare the arguments with their types
    args_details = "\n".join([
        f"  - {arg['name']} ({arg['type']}): Description of {arg['name']}."
        for arg in function_meta.args
    ])

    # Prepare the global context if available
    global_context_info = ""
    if global_context:
        global_context_info = "\n".join([
            f"{key}: {value}" for key, value in global_context.items()
        ])

    # Construct the prompt
    prompt = f"""
You are an AI assistant tasked with generating a Google-style Python docstring for the given function.

Please return the response in a JSON format only. Example:
{{
    "description": "Short description here.",
    "args": [
        {{"name": "arg_name", "type": "arg_type", "description": "Description of the argument."}},
        ...
    ],
    "returns": "Description of the return value.",
    "raises": ["Description of raised exceptions."],
    "examples": ["Example usage here"]
}}

Function Name: {function_meta.name}
Signature: {signature}
Existing Description: {function_meta.docstring or "None"}
Arguments:
{args_details}

Return Type: {function_meta.return_type or "None"}
Decorators: {', '.join(function_meta.decorators) or "None"}
Module: {module_meta.name} - {module_meta.docstring or "No description"}

Global Context:
{global_context_info}

Only return the docstring content, formatted as JSON, without any additional comments or explanations.
"""
    return prompt.strip()

# Example usage:
if __name__ == "__main__":
    # Mock data for demonstration
    function_meta = FunctionMetadata(
        name="calculate_average",
        args=[{'name': 'numbers', 'type': 'List[int]'}],
        return_type='float',
        decorators=[],
        docstring="Calculate the average of a list of numbers."
    )
    module_meta = ModuleMetadata(
        name="math_utils",
        docstring="Utility functions for mathematical operations."
    )
    prompt = generate_function_prompt(function_meta, module_meta)
    print(prompt)