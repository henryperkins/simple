"""  
Function Extraction Module  
  
Extracts function definitions and their metadata from Python source code.  
"""  
  
import ast  
from typing import List, Dict, Any  
from core.logger import LoggerSetup  
from extract.base import BaseExtractor  
from core.utils import handle_exceptions  
from extract.ast_analysis import ASTAnalyzer  
  
logger = LoggerSetup.get_logger(__name__)  
  
class FunctionExtractor(BaseExtractor):  
    """  
    Extract function definitions and their metadata from Python source code.  
    """  
  
    def __init__(self, source_code: str):  
        super().__init__(source_code)  
        self.analyzer = ASTAnalyzer()  
  
    @handle_exceptions(logger.error)  
    def extract_functions(self) -> List[Dict[str, Any]]:  
        """Extract all function definitions and their metadata from the source code."""  
        logger.debug("Extracting function definitions.")  
        functions = []  
        for node in self.analyzer.extract_functions(self.tree):  
            logger.debug(f"Processing function: {node.name}")  
            function_info = self.extract_details(node)  
            if function_info:  
                functions.append(function_info)  
                logger.info(f"Extracted function '{node.name}' with metadata.")  
        logger.debug(f"Total functions extracted: {len(functions)}")  
        return functions  
  
    @handle_exceptions(logger.error)  
    def extract_details(self, node: ast.FunctionDef) -> Dict[str, Any]:  
        """Extract details from a function definition node."""  
        logger.debug(f"Extracting details for function: {node.name}")  
        details = self._extract_common_details(node)  
        details.update({  
            "args": [  
                {  
                    "name": arg.arg,  
                    "type": self.analyzer.get_annotation(arg.annotation),  
                    "default": self._get_default_value(arg)  
                }  
                for arg in node.args.args  
            ],  
            "return_type": self.analyzer.get_annotation(node.returns),  
            "decorators": self._extract_decorators(node),  
            "exceptions": self._detect_exceptions(node),  
            "body_summary": self.get_body_summary(node),  
        })  
  
        # Include inter-module context if needed  
        details["external_calls"] = self._detect_external_calls(node)  
  
        return details  
  
    def get_body_summary(self, node: ast.FunctionDef) -> str:  
        """Generate a summary of the function body."""  
        logger.debug(f"Generating body summary for function: {node.name}")  
        body_statements = node.body[:3]  
        summary = "\n".join(ast.unparse(stmt) for stmt in body_statements) + "\n..."  
        return summary  
  
    def _get_default_value(self, arg: ast.arg) -> Optional[str]:  
        """Get the default value of a function argument if it exists."""  
        defaults = self.tree.body[0].args.defaults if isinstance(self.tree.body[0], ast.FunctionDef) else []  
        if defaults:  
            index = self.tree.body[0].args.args.index(arg) - (len(self.tree.body[0].args.args) - len(defaults))  
            if index >= 0:  
                default_value = ast.unparse(defaults[index])  
                return default_value  
        return None  
  
    def _detect_external_calls(self, node: ast.FunctionDef) -> List[str]:  
        """Detect external function or method calls within a function."""  
        logger.debug(f"Detecting external calls in function: {node.name}")  
        external_calls = []  
        for child in ast.walk(node):  
            if isinstance(child, ast.Call):  
                func_name = self._get_call_name(child.func)  
                if func_name:  
                    external_calls.append(func_name)  
        logger.debug(f"External calls detected: {external_calls}")  
        return external_calls  
  
    def _get_call_name(self, node: ast.AST) -> Optional[str]:  
        """Get the full name of a function or method call."""  
        if isinstance(node, ast.Name):  
            return node.id  
        elif isinstance(node, ast.Attribute):  
            value = self._get_call_name(node.value)  
            if value:  
                return f"{value}.{node.attr}"  
            else:  
                return node.attr  
        return None  