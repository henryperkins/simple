import ast
from typing import Dict, Any, List, Union, Optional
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class DocstringUtils:
    """Utility methods for extracting docstring and metadata from AST nodes."""

    @staticmethod
    def extract_metadata(node: ast.AST) -> Dict[str, Any]:
        """Extract common metadata and docstring info from an AST node."""
        try:
            docstring_info = DocstringUtils.extract_docstring_info(node)
            return {
                'name': getattr(node, 'name', None),
                'lineno': getattr(node, 'lineno', None),
                'docstring_info': docstring_info
            }
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}", exc_info=True)
            return {}

    @staticmethod
    def extract_docstring_info(node: ast.AST) -> Dict[str, Any]:
        """Extract docstring and related metadata from an AST node."""
        try:
            docstring = ast.get_docstring(node) or ""

            # Extract argument information
            args = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = DocstringUtils.extract_function_args(node)

            # Extract return information
            returns = DocstringUtils.extract_return_info(node)

            # Extract raised exceptions
            raises = DocstringUtils.extract_raises(node)

            return {
                'docstring': docstring,
                'args': args,
                'returns': returns,
                'raises': raises,
                'metadata': {
                    'lineno': getattr(node, 'lineno', None),
                    'name': getattr(node, 'name', None),
                    'type': type(node).__name__
                }
            }
        except Exception as e:
            logger.error(f"Error extracting docstring info: {e}", exc_info=True)
            return {}

    @staticmethod
    def extract_function_args(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict[str, Any]]:
        """Extract function argument information."""
        args = []
        for arg in node.args.args:
            arg_info = {
                'name': arg.arg,
                'type': DocstringUtils.get_node_name(arg.annotation) if arg.annotation else 'Any',
                'description': '',  # Will be filled by docstring processor
            }
            # Handle default values
            if node.args.defaults:
                default_index = len(node.args.args) - len(node.args.defaults)
                if node.args.args.index(arg) >= default_index:
                    default_value = DocstringUtils.get_node_name(
                        node.args.defaults[node.args.args.index(arg) - default_index]
                    )
                    arg_info['default_value'] = default_value
                    arg_info['optional'] = True
            args.append(arg_info)
        return args

    @staticmethod
    def extract_return_info(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        """Extract return type information."""
        if node.returns:
            return {
                'type': DocstringUtils.get_node_name(node.returns),
                'description': ''  # Will be filled by docstring processor
            }
        return {'type': 'Any', 'description': ''}

    @staticmethod
    def extract_raises(node: ast.AST) -> List[Dict[str, Any]]:
        """Extract exception information from the AST."""
        raises = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Raise) and child.exc:
                exc_name = DocstringUtils.get_exception_name(child.exc)
                if exc_name:
                    raises.add(exc_name)
        return [{'exception': exc, 'description': ''} for exc in raises]

    @staticmethod
    def get_exception_name(node: ast.AST) -> Optional[str]:
        """Get the name of an exception node."""
        try:
            if isinstance(node, ast.Call):
                return DocstringUtils.get_node_name(node.func)
            elif isinstance(node, (ast.Name, ast.Attribute)):
                return DocstringUtils.get_node_name(node)
            return "Exception"
        except Exception as e:
            logger.error(f"Error getting exception name: {e}", exc_info=True)
            return None

    @staticmethod
    def get_node_name(node: Optional[ast.AST]) -> str:
        """Get a string representation of a node's name."""
        if node is None:
            return "Any"
        try:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return f"{DocstringUtils.get_node_name(node.value)}.{node.attr}"
            elif isinstance(node, ast.Subscript):
                value = DocstringUtils.get_node_name(node.value)
                slice_val = DocstringUtils.get_node_name(node.slice)
                return f"{value}[{slice_val}]"
            elif isinstance(node, ast.Call):
                return f"{DocstringUtils.get_node_name(node.func)}()"
            elif isinstance(node, (ast.Tuple, ast.List)):
                elements = ', '.join(DocstringUtils.get_node_name(e) for e in node.elts)
                return f"({elements})" if isinstance(node, ast.Tuple) else f"[{elements}]"
            elif isinstance(node, ast.Constant):
                return str(node.value)
            elif hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                return f"Unknown<{type(node).__name__}>"
        except Exception as e:
            logger.error(f"Error getting name from node {type(node).__name__}: {e}", exc_info=True)
            return f"Unknown<{type(node).__name__}>"