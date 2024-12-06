import os
import glob
import ast
import networkx as nx
from typing import Dict, Tuple

def build_project_context(project_root: str) -> Tuple[Dict[str, Dict], nx.DiGraph]:
    """
    Analyze Python files in a project directory to build a global context and dependency graph.

    Args:
        project_root (str): The root directory of the project.

    Returns:
        Tuple[Dict[str, Dict], nx.DiGraph]: A context dictionary and a module dependency graph.
    """
    context = {}
    dependency_graph = nx.DiGraph()

    # Find all Python files in the project directory
    python_files = glob.glob(os.path.join(project_root, '**/*.py'), recursive=True)

    for file_path in python_files:
        module_name = os.path.relpath(file_path, project_root).replace(os.path.sep, '.')[:-3]
        context[module_name] = {
            'imports': [],
            'functions': [],
            'classes': []
        }

        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()

        tree = ast.parse(source_code)

        # Traverse the AST to gather information
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    context[module_name]['imports'].append(alias.name)
                    dependency_graph.add_edge(module_name, alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ''
                for alias in node.names:
                    full_import = f"{module}.{alias.name}" if module else alias.name
                    context[module_name]['imports'].append(full_import)
                    dependency_graph.add_edge(module_name, full_import)
            elif isinstance(node, ast.FunctionDef):
                context[module_name]['functions'].append(node.name)
            elif isinstance(node, ast.ClassDef):
                context[module_name]['classes'].append(node.name)

    return context, dependency_graph

def summarize_global_context_for_function(function_name: str, project_context: Dict[str, Dict]) -> Dict:
    """
    Summarize relevant global context for a given function.

    Args:
        function_name (str): The name of the function to summarize context for.
        project_context (Dict[str, Dict]): The global project context.

    Returns:
        Dict: A dictionary of relevant global information for the function.
    """
    relevant_context = {
        'modules': [],
        'related_functions': [],
        'related_classes': []
    }

    for module, details in project_context.items():
        if function_name in details['functions']:
            relevant_context['modules'].append(module)
            relevant_context['related_functions'].extend(details['functions'])
            relevant_context['related_classes'].extend(details['classes'])

    return relevant_context

# Example usage
if __name__ == "__main__":
    project_root = 'path/to/your/project'
    context, graph = build_project_context(project_root)
    print("Project Context:", context)
    print("Dependency Graph:", graph.edges())

    function_context = summarize_global_context_for_function('calculate_average', context)
    print("Function Context:", function_context)