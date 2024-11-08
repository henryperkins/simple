from typing import Any, Optional, List, Union
from abc import ABC, abstractmethod
from core.logger import LoggerSetup

# Initialize logger for this module
logger = LoggerSetup.get_logger("extract.base")

class BaseExtractor(ABC):
    def __init__(self, node: ast.AST, content: str) -> None:
        if node is None:
            raise ValueError("AST node cannot be None")
        if not content:
            raise ValueError("Content cannot be empty")
        self.node = node
        self.content = content
        logger.debug(f"Initialized {self.__class__.__name__} for node type {type(node).__name__}")

    def get_annotation(self, annotation: Optional[ast.AST]) -> str:
        try:
            return get_annotation(annotation)
        except Exception as e:
            logger.error(f"Error getting annotation: {e}")
            return "Unknown"

    def get_source_segment(self, node: ast.AST) -> str:
        try:
            return ast.get_source_segment(self.content, node)
        except Exception as e:
            logger.error(f"Error getting source segment: {e}")
            return ""

    @abstractmethod
    def extract_details(self) -> dict:
        pass

    def get_docstring(self) -> str:
        try:
            return ast.get_docstring(self.node) or ""
        except Exception as e:
            logger.error(f"Error extracting docstring: {e}")
            return ""

    def get_decorators(self) -> List[str]:
        decorators = []
        try:
            if hasattr(self.node, 'decorator_list'):
                for decorator in self.node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorators.append(decorator.id)
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name):
                            decorators.append(decorator.func.id)
        except Exception as e:
            logger.error(f"Error extracting decorators: {e}")
        return decorators