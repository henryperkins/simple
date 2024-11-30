"""  
Code processor that uses AST parsing instead of code execution.  
"""  
  
import ast  
from typing import Optional, Tuple, Dict, Any, List  
from datetime import datetime  
from core.logger import LoggerSetup  
  
class ASTProcessor:  
    """  
    Processes Python code using AST parsing instead of code execution.  
    """  
  
    def __init__(self):  
        """Initialize the AST processor."""  
        self.logger = LoggerSetup.get_logger(__name__)  # Initialize self.logger  
        self.required_imports = set()  
        self.classes = []  
        self.functions = []  
        self.dependencies = set()  
  
    async def process_code(self, source_code: str, cache_key: Optional[str] = None, extracted_info: Optional[Dict[str, Any]] = None) -> Optional[Tuple[ast.AST, Dict[str, Any]]]:  
        """  
        Process source code to generate documentation using AST parsing.  
        """  
        try:  
            # Parse the source code into an AST  
            tree = ast.parse(source_code)  
            self._add_parents(tree)  
  
            # Analyze the AST to extract metadata  
            metadata = {}  
            self._analyze_tree(tree)  
            metadata['required_imports'] = list(self.required_imports)  
            metadata['classes'] = self.classes  
            metadata['functions'] = self.functions  
            metadata['dependencies'] = list(self.dependencies)  
  
            return tree, metadata  
  
        except Exception as e:  
            self.logger.error(f"Error processing code: {str(e)}")  
            return None  
  
    def _analyze_tree(self, tree: ast.AST) -> None:  
        """  
        Analyze AST to extract required imports and other metadata.  
  
        Args:  
            tree: The AST to analyze  
        """  
        for node in ast.walk(tree):  
            # Find datetime references and other builtins  
            if isinstance(node, ast.Name) and node.id in {'datetime', 'timedelta'}:  
                self.required_imports.add('datetime')  
  
            # Record class definitions  
            elif isinstance(node, ast.ClassDef):  
                class_info = {  
                    'name': node.name,  
                    'bases': [self._get_name(base) for base in node.bases],  
                    'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],  
                    'line_number': node.lineno  
                }  
                self.classes.append(class_info)  
                self.logger.debug(f"Found class: {node.name}")  
  
            # Record function definitions  
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):  
                func_info = {  
                    'name': node.name,  
                    'args': self._get_function_args(node),  
                    'is_async': isinstance(node, ast.AsyncFunctionDef),  
                    'line_number': node.lineno  
                }  
                self.functions.append(func_info)  
                self.logger.debug(f"Found function: {node.name}")  
  
            # Track imports  
            elif isinstance(node, (ast.Import, ast.ImportFrom)):  
                self._process_imports(node)  
  
    def _get_function_args(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:  
        """Get function argument information."""  
        args = []  
        for arg in node.args.args:  
            arg_info = {  
                'name': arg.arg,  
                'annotation': self._get_name(arg.annotation) if arg.annotation else 'Any'  
            }  
            args.append(arg_info)  
        return args  
  
    def _get_name(self, node: Optional[ast.AST]) -> str:  
        """Get string representation of a name node."""  
        if node is None:  
            return 'Any'  
        elif isinstance(node, ast.Name):  
            return node.id  
        elif isinstance(node, ast.Attribute):  
            return f"{self._get_name(node.value)}.{node.attr}"  
        elif isinstance(node, ast.Constant):  
            return str(node.value)  
        else:  
            return ast.unparse(node)  
  
    def _process_imports(self, node: ast.AST) -> None:  
        """Process import nodes to track dependencies."""  
        try:  
            if isinstance(node, ast.Import):  
                for name in node.names:  
                    self.dependencies.add(name.name)  
                    self.logger.debug(f"Found import: {name.name}")  
            elif isinstance(node, ast.ImportFrom):  
                if node.module:  
                    self.dependencies.add(node.module)  
                    self.logger.debug(f"Found import from: {node.module}")  
        except Exception as e:  
            self.logger.error(f"Error processing imports: {str(e)}")  
  
    def _add_parents(self, node: ast.AST) -> None:  
        """Add parent references to AST nodes."""  
        for child in ast.iter_child_nodes(node):  
            child.parent = node  
            self._add_parents(child)  