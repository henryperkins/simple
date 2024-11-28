"""  
Base Extraction Module  
  
Provides a base class for extracting information from AST nodes.  
"""  
  
import ast  
from abc import ABC, abstractmethod  
from typing import Generator, Optional, Dict, Any, List  
from core.logger import LoggerSetup  
from core.utils import handle_exceptions  
from extract.ast_analysis import ASTAnalyzer  
  
logger = LoggerSetup.get_logger(__name__)  
  
class BaseExtractor(ABC):  
    """Base class for extracting information from AST nodes."""  
  
    def __init__(self, source_code: str) -> None:  
        """  
        Initialize the extractor with source code and parse it into an AST.  
  
        Args:  
            source_code (str): The source code to parse.  
        """  
        self.analyzer = ASTAnalyzer()  
        self.source_code = source_code  
        self.tree = self.analyzer.parse_source_code(source_code)  
  
    def walk_tree(self) -> Generator[ast.AST, None, None]:  
        """Walk the AST and yield all nodes."""  
        logger.debug("Walking the AST.")  
        return ast.walk(self.tree)  
  
    @handle_exceptions(logger.error)  
    def extract_docstring(self, node: ast.AST) -> Optional[str]:  
        """Extract the docstring from an AST node."""  
        return self.analyzer.extract_docstring(node)  
  
    def _extract_common_details(self, node: ast.AST) -> Dict[str, Any]:  
        """Extract common details from an AST node."""  
        details = {  
            'name': getattr(node, 'name', '<unknown>'),  
            'docstring': self.extract_docstring(node),  
            'lineno': getattr(node, 'lineno', 0)  
        }  
        return details  
  
    def _extract_decorators(self, node: Union[ast.FunctionDef, ast.ClassDef]) -> List[str]:  
        """Extract decorators from a function or class node."""  
        decorators = []  
        for decorator in node.decorator_list:  
            try:  
                decorators.append(ast.unparse(decorator))  
            except Exception as e:  
                logger.error(f"Error unparsing decorator: {e}")  
        return decorators  
  
    def _detect_exceptions(self, node: ast.FunctionDef) -> List[str]:  
        """Detect exceptions that could be raised by the function."""  
        exceptions = set()  
        for child in ast.walk(node):  
            if isinstance(child, ast.Raise):  
                exc = child.exc  
                if isinstance(exc, ast.Name):  
                    exceptions.add(exc.id)  
                elif isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):  
                    exceptions.add(exc.func.id)  
        return list(exceptions)  
  
    @abstractmethod  
    def extract_details(self, node: ast.AST) -> Dict[str, Any]:  
        """Abstract method to extract details from a given AST node."""  
        pass  