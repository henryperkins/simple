# code_extraction.py
import ast
import logging

def extract_classes_and_functions_from_ast(tree, content):
    """Extract class and function details from AST with full node coverage.

    Args:
        tree (ast.AST): The AST tree of the Python file.
        content (str): The source code content of the file.

    Returns:
        dict: A dictionary containing lists of classes and functions extracted from the AST.
    """
    classes = []
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(extract_class_details(node, content))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not isinstance(getattr(node, 'parent', None), ast.ClassDef):
                functions.append(extract_function_details(node, content))

    return {"classes": classes, "functions": functions}

def extract_class_details(node, content):
    """Extract details from a class node.

    Args:
        node (ast.ClassDef): The class node.
        content (str): The source code content.

    Returns:
        dict: A dictionary containing class details.
    """
    class_name = node.name
    class_docstring = ast.get_docstring(node) or ""
    class_code = ast.get_source_segment(content, node)

    methods = []
    for body_item in node.body:
        if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(extract_function_details(body_item, content))

    return {
        "name": class_name,
        "docstring": class_docstring,
        "code": class_code,
        "methods": methods,
        "node": node,
    }

def extract_function_details(node, content):
    """Extract details from a function node.

    Args:
        node (ast.FunctionDef or ast.AsyncFunctionDef): The function node.
        content (str): The source code content.

    Returns:
        dict: A dictionary containing function details.
    """
    func_name = node.name
    params = [
        (
            arg.arg,
            ast.unparse(arg.annotation) if hasattr(arg, "annotation") and arg.annotation else "Unknown",
        )
        for arg in node.args.args
    ]
    return_type = ast.unparse(node.returns) if node.returns else "Unknown"
    docstring = ast.get_docstring(node) or ""
    function_code = ast.get_source_segment(content, node)
    complexity_score = calculate_complexity(node)

    return {
        "name": func_name,
        "params": params,
        "return_type": return_type,
        "docstring": docstring,
        "code": function_code,
        "complexity_score": complexity_score,
        "node": node,
    }

def calculate_complexity(node):
    """Calculate the cyclomatic complexity score for a function.

    Args:
        node (ast.AST): The AST node representing a function.

    Returns:
        int: The cyclomatic complexity score.
    """
    complexity = 1  # Start with one for the function entry point
    for subnode in ast.walk(node):
        if isinstance(subnode, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            complexity += 1
        elif isinstance(subnode, ast.BoolOp):
            # Each boolean operation (and/or) adds a branch
            complexity += len(subnode.values) - 1
    return complexity