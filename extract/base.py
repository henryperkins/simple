# extract/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any
import ast
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger("extract.base")

class BaseExtractor(ABC):
    """Base class for AST extractors."""

    def __init__(self, node: ast.AST, content: str) -> None:
        """
        Initialize the BaseExtractor with an AST node and source content.

        Args:
            node (ast.AST): The AST node to extract information from.
            content (str): The source code content.
        
        Raises:
            ValueError: If node is None or content is empty.
        """
        if node is None:
            raise ValueError("AST node cannot be None")
        if not content:
            raise ValueError("Content cannot be empty")
        self.node = node
        self.content = content
        logger.debug(f"Initialized {self.__class__.__name__} for node type {type(node).__name__}")

    @abstractmethod
    def extract_details(self) -> Dict[str, Any]:
        """Extract details from the AST node."""
        pass

    def get_docstring(self) -> str:
        """Extract docstring from node."""
        try:
            return ast.get_docstring(self.node) or ""
        except Exception as e:
            logger.error(f"Error extracting docstring: {e}")
            return ""

    def get_source_segment(self, node: ast.AST) -> str:
        """Get source code segment for a node."""
        try:
            return ast.get_source_segment(self.content, node) or ""
        except Exception as e:
            logger.error(f"Error getting source segment: {e}")
            return ""

    def _get_empty_details(self) -> Dict[str, Any]:
        """Return empty details structure matching schema."""
        return {
            "name": "",
            "docstring": "",
            "params": [],
            "returns": {"type": "None", "has_type_hint": False},
            "complexity_score": 0,
            "line_number": 0,
            "end_line_number": 0,
            "code": "",
            "is_async": False,
            "is_generator": False,
            "is_recursive": False,
            "summary": "",
            "changelog": []
        }
