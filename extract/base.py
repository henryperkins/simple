import ast
from typing import Any, Dict, List, Optional
from core.logger import LoggerSetup
logger = LoggerSetup.get_logger("extract.base")

class BaseExtractor:
    """
    Base class for extractors that provides common functionality for extracting
    details from AST nodes.
    """

    def __init__(self, node: ast.AST, content: str):
        """
        Initialize the BaseExtractor with an AST node and the source code content.

        Args:
            node (ast.AST): The AST node to extract information from.
            content (str): The source code content.
        """
        self.node = node
        self.content = content
        self.logger = LoggerSetup.get_logger(self.__class__.__name__)

    def extract_details(self) -> Dict[str, Any]:
        """
        Extract details from the AST node. This method should be implemented by subclasses.

        Returns:
            Dict[str, Any]: The extracted details.
        """
        raise NotImplementedError("Subclasses must implement extract_details")

    def get_annotation(self, annotation: Optional[ast.AST]) -> str:
        """
        Convert an AST annotation to a string representation.

        Args:
            annotation (Optional[ast.AST]): The annotation node to process.

        Returns:
            str: The string representation of the annotation.
        """
        if annotation is None:
            return "None"

        try:
            if isinstance(annotation, ast.Name):
                return annotation.id
            elif isinstance(annotation, ast.Constant):
                return str(annotation.value)
            # Add other annotation types as needed
            return "Unknown"
        except Exception as e:
            self.logger.error(f"Error processing annotation: {e}")
            return "Unknown"