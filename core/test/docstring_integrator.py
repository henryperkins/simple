import ast
import astor  # Ensure you have astor installed: pip install astor
import subprocess
from response_parser import DocstringSchema
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_docstring(docstring_data: DocstringSchema) -> str:
    args_formatted = "\n".join([
        f"    {arg.name} ({arg.type or 'None'}): {arg.description}"
        for arg in docstring_data.args
    ])
    raises_formatted = "\n".join([
        f"    {exc}" for exc in (docstring_data.raises or [])
    ])
    examples_formatted = ""
    if docstring_data.examples:
        examples_formatted = "Examples:\n" + "\n".join([
            f"    {ex}" for ex in docstring_data.examples
        ])

    logger.debug("Formatting docstring data into string.")
    return f'''{docstring_data.description}
    
    Args:
    {args_formatted}
    
    Returns:
        {docstring_data.returns or ''}
    
    Raises:
    {raises_formatted}
    
    {examples_formatted}'''.strip()

def insert_or_update_docstring(ast_node, docstring_str):
    """
    Insert or update the docstring in the given AST node.

    Args:
        ast_node (ast.AST): The AST node of the function or class.
        docstring_str (str): The docstring to insert or update.
    """
    logger.debug(f"Inserting or updating docstring for AST node: {ast_node.name}")
    docstring_node = ast.Expr(value=ast.Constant(value=docstring_str))
    if ast.get_docstring(ast_node):
        ast_node.body[0] = docstring_node
        logger.info(f"Updated existing docstring for {ast_node.name}")
    else:
        ast_node.body.insert(0, docstring_node)
        logger.info(f"Inserted new docstring for {ast_node.name}")

def integrate_docstring_into_file(file_path: str, docstring_data: DocstringSchema, target_function_name: str):
    """
    Integrate an enriched docstring into a Python file by updating the AST.

    Args:
        file_path (str): The path to the Python file.
        docstring_data (DocstringSchema): The enriched docstring data.
        target_function_name (str): The name of the function to update.
    """
    logger.info(f"Integrating docstring into file: {file_path} for function: {target_function_name}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        logger.debug(f"Read source code from {file_path}")
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise

    try:
        tree = ast.parse(source_code)
        logger.debug("Parsed AST from source code.")
    except SyntaxError as e:
        logger.error(f"Syntax error while parsing {file_path}: {e}")
        raise

    # Find the target function and update its docstring
    function_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == target_function_name:
            formatted_docstring = format_docstring(docstring_data)
            insert_or_update_docstring(node, formatted_docstring)
            function_found = True
            break

    if not function_found:
        logger.warning(f"Function {target_function_name} not found in {file_path}")
        return

    # Write the updated code back to the file
    try:
        updated_code = astor.to_source(tree)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_code)
        logger.info(f"Successfully updated the file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to write updated code to {file_path}: {e}")
        raise

    # Run a code formatter if available
    try:
        subprocess.run(['black', file_path], check=True)
        logger.info(f"Formatted {file_path} using black.")
    except FileNotFoundError:
        logger.warning("Black formatter not found. Skipping code formatting.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running black formatter: {e}")
