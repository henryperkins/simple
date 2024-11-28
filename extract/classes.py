"""  
Class Extraction Module  
  
Extracts class definitions and their metadata from Python source code.  
"""  
  
import ast  
from typing import List, Dict, Any  
from core.logger import LoggerSetup  
from extract.base import BaseExtractor  
from core.utils import handle_exceptions  
from extract.ast_analysis import ASTAnalyzer  
  
logger = LoggerSetup.get_logger(__name__)  
  
class ClassExtractor(BaseExtractor):  
    """  
    Extract class definitions and their metadata from Python source code.  
    """  
  
    def __init__(self, source_code: str):  
        super().__init__(source_code)  
        self.analyzer = ASTAnalyzer()  
  
    @handle_exceptions(logger.error)  
    def extract_classes(self) -> List[Dict[str, Any]]:  
        """Extract all class definitions and their metadata from the source code."""  
        logger.debug("Extracting class definitions.")  
        classes = []  
        for node in self.analyzer.extract_classes(self.tree):  
            logger.debug(f"Processing class: {node.name}")  
            class_info = self.extract_details(node)  
            if class_info:  
                classes.append(class_info)  
                logger.info(f"Extracted class '{node.name}' with metadata.")  
        logger.debug(f"Total classes extracted: {len(classes)}")  
        return classes  
  
    @handle_exceptions(logger.error)  
    def extract_details(self, node: ast.ClassDef) -> Dict[str, Any]:  
        """Extract details from a class definition node."""  
        logger.debug(f"Extracting details for class: {node.name}")  
        details = self._extract_common_details(node)  
        details.update({  
            "bases": [self.analyzer.get_annotation(base) for base in node.bases],  
            "decorators": self._extract_decorators(node),  
            "methods": [self.extract_method_details(n) for n in node.body if isinstance(n, ast.FunctionDef)],  
            "attributes": self.extract_class_attributes(node),  
            "inherits_exception": self._inherits_exception(node),  
        })  
  
        # Include inter-module context if needed  
        details["used_classes"] = self._detect_used_classes(node)  
  
        return details  
  
    def extract_method_details(self, node: ast.FunctionDef) -> Dict[str, Any]:  
        """Extract detailed information from a method node."""  
        logger.debug(f"Extracting method details for method: {node.name}")  
        method_extractor = FunctionExtractor(ast.unparse(node))  
        method_details = method_extractor.extract_details(node)  
        return method_details  
  
    def extract_class_attributes(self, class_node: ast.ClassDef) -> List[Dict[str, Any]]:  
        """Extract attributes from a class node."""  
        logger.debug(f"Extracting attributes for class: {class_node.name}")  
        attributes = []  
        for node in class_node.body:  
            if isinstance(node, ast.Assign):  
                for target in node.targets:  
                    if isinstance(target, ast.Name):  
                        attr_info = {  
                            "name": target.id,  
                            "type": self._infer_type(node.value),  
                            "lineno": node.lineno,  
                        }  
                        attributes.append(attr_info)  
        return attributes  
  
    def _infer_type(self, value_node: ast.AST) -> str:  
        """Infer the type of a value node."""  
        if isinstance(value_node, ast.Constant):  
            return type(value_node.value).__name__  
        elif isinstance(value_node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):  
            return type(value_node).__name__  
        else:  
            return "Unknown"  
  
    def get_body_summary(self, node: ast.ClassDef) -> str:  
        """Generate a summary of the class body."""  
        logger.debug(f"Generating body summary for class: {node.name}")  
        body_statements = node.body[:3]  
        summary = "\n".join(ast.unparse(stmt) for stmt in body_statements) + "\n..."  
        return summary  
  
    def _inherits_exception(self, node: ast.ClassDef) -> bool:  
        """Check if the class inherits from Exception or BaseException."""  
        exception_bases = {'Exception', 'BaseException'}  
        for base in node.bases:  
            base_name = self.analyzer.get_annotation(base)  
            if base_name in exception_bases:  
                return True  
        return False  
  
    def _detect_used_classes(self, node: ast.ClassDef) -> List[str]:  
        """Detect other classes used within this class."""  
        logger.debug(f"Detecting used classes in class: {node.name}")  
        used_classes = []  
        for child in ast.walk(node):  
            if isinstance(child, ast.Call):  
                func_name = self._get_call_name(child.func)  
                if func_name and func_name[0].isupper():  
                    used_classes.append(func_name)  
        logger.debug(f"Used classes detected: {used_classes}")  
        return used_classes  
  
    def _get_call_name(self, node: ast.AST) -> Optional[str]:  
        """Get the full name of a class being instantiated."""  
        if isinstance(node, ast.Name):  
            return node.id  
        elif isinstance(node, ast.Attribute):  
            value = self._get_call_name(node.value)  
            if value:  
                return f"{value}.{node.attr}"  
            else:  
                return node.attr  
        return None  