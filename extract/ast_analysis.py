"""AST Analysis Module - Provides utilities for analyzing Python AST nodes."""  
  
import ast  
from typing import List, Optional, Union, Dict, Any  
from core.logger import LoggerSetup  
from core.metrics import Metrics  
  
logger = LoggerSetup.get_logger(__name__)  
  
class ASTAnalyzer:  
    """  
    Provides utilities for analyzing Python AST nodes, including parsing,  
    extracting classes and functions, and handling docstrings and annotations.  
    """  
  
    def __init__(self) -> None:  
        """Initialize ASTAnalyzer with a metrics calculator."""  
        self.metrics_calculator = Metrics()  
  
    def parse_source_code(self, source_code: str) -> ast.AST:  
        """Parse source code into an Abstract Syntax Tree (AST)."""  
        logger.debug("Parsing source code into AST.")  
        return ast.parse(source_code)  
  
    @staticmethod  
    def extract_functions(tree: ast.AST) -> List[ast.FunctionDef]:  
        """Extract function definitions from an AST."""  
        logger.debug("Extracting function definitions from AST.")  
        return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]  
  
    @staticmethod  
    def extract_classes(tree: ast.AST) -> List[ast.ClassDef]:  
        """Extract class definitions from an AST."""  
        logger.debug("Extracting class definitions from AST.")  
        return [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]  
  
    def extract_docstring(  
        self, node: Union[ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef]  
    ) -> Optional[str]:  
        """Extract the docstring from an AST node."""  
        logger.debug(f"Extracting docstring from node: {type(node).__name__}")  
        return ast.get_docstring(node)  
  
    def get_annotation(self, node: Optional[ast.AST]) -> str:  
        """Get the annotation of an AST node."""  
        if node is None:  
            return "Any"  
        try:  
            return ast.unparse(node)  
        except Exception as e:  
            logger.error(f"Error processing annotation: {e}")  
            return "Any"  
  
    def add_parent_info(self, tree: ast.AST) -> None:  
        """Add parent node information to each node in an AST."""  
        logger.debug("Adding parent information to AST nodes.")  
        for parent in ast.walk(tree):  
            for child in ast.iter_child_nodes(parent):  
                child.parent = parent  
        logger.debug("Parent information added to AST nodes.")  
  
    def calculate_function_metrics(  
        self, function_node: ast.FunctionDef  
    ) -> Dict[str, Union[int, float]]:  
        """Calculate complexity metrics for a function."""  
        try:  
            metrics = self.metrics_calculator.calculate_metrics(function_node)  
            logger.debug(f"Calculated metrics for function '{function_node.name}': {metrics}")  
            return metrics  
        except Exception as e:  
            logger.error(f"Error calculating metrics for function '{function_node.name}': {e}")  
            return {}  
  
    def calculate_class_metrics(  
        self, class_node: ast.ClassDef  
    ) -> Dict[str, Union[int, float]]:  
        """Calculate complexity metrics for a class."""  
        method_metrics = {}  
        for node in class_node.body:  
            if isinstance(node, ast.FunctionDef):  
                metrics = self.calculate_function_metrics(node)  
                method_metrics[node.name] = metrics  
        class_metrics = {  
            "methods_metrics": method_metrics  
        }  
        logger.debug(f"Calculated metrics for class '{class_node.name}': {class_metrics}")  
        return class_metrics  
  
    def extract_imports(self, tree: ast.AST) -> List[str]:  
        """Extract import statements from the AST."""  
        logger.debug("Extracting import statements from AST.")  
        imports = []  
        for node in ast.walk(tree):  
            if isinstance(node, ast.Import):  
                for alias in node.names:  
                    imports.append(alias.name)  
            elif isinstance(node, ast.ImportFrom):  
                module = node.module or ""  
                for alias in node.names:  
                    imports.append(f"{module}.{alias.name}")  
        logger.debug(f"Extracted imports: {imports}")  
        return imports  
  
    def extract_globals(self, tree: ast.AST) -> List[str]:  
        """Extract global variables from the AST."""  
        logger.debug("Extracting global variables from AST.")  
        globals_vars = []  
        for node in ast.walk(tree):  
            if isinstance(node, ast.Assign) and isinstance(node.parent, ast.Module):  
                for target in node.targets:  
                    if isinstance(target, ast.Name):  
                        globals_vars.append(target.id)  
        logger.debug(f"Extracted global variables: {globals_vars}")  
        return globals_vars  