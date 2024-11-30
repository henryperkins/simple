"""
Code processor that uses AST parsing instead of code execution.
"""

import ast
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Set

# Remove unused import:
# from typing_extensions import TypedDict  

from core.logger import LoggerSetup
from dataclasses import dataclass

@dataclass
class CodeMetadata:
    """Represents metadata extracted from code analysis."""
    required_imports: Set[str]
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    dependencies: Set[str]
    
    @classmethod
    def create_empty(cls) -> 'CodeMetadata':
        """Create an empty metadata instance."""
        return cls(
            required_imports=set(),
            classes=[],
            functions=[],
            dependencies=set()
        )
        
        
class ASTProcessor:
    """Processes Python code using AST parsing instead of code execution."""

    def __init__(self, cache=None):
        """
        Initialize the AST processor.
        
        Args:
            cache: Optional cache implementation for storing processed results
        """
        self.logger = LoggerSetup.get_logger(__name__)
        self.metadata = CodeMetadata.create_empty()
        self.cache = cache
  
    async def process_code(
        self, 
        source_code: str,
        cache_key: Optional[str] = None,
        extracted_info: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[ast.AST, Dict[str, Any]]]:
        """
        Process source code to generate documentation using AST parsing.
        
        Args:
            source_code: The source code to process
            cache_key: Optional cache key for previously processed code
            extracted_info: Optional pre-extracted code information
            
        Returns:
            Tuple of (AST, metadata dictionary) or None if processing fails
        """
        try:
            # Check cache if cache_key is provided
            if cache_key and hasattr(self, 'cache'):
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    self.logger.debug("Cache hit for key: %s", cache_key)
                    return cached_result

            tree = ast.parse(source_code)
            self._add_parents(tree)
            
            # Use extracted_info if provided, otherwise analyze tree
            if extracted_info:
                self.metadata = CodeMetadata(
                    required_imports=set(extracted_info.get('required_imports', [])),
                    classes=extracted_info.get('classes', []),
                    functions=extracted_info.get('functions', []),
                    dependencies=set(extracted_info.get('dependencies', []))
                )
                self.logger.debug("Using pre-extracted info")
            else:
                self._analyze_tree(tree)
                
            result = (tree, {
                'required_imports': list(self.metadata.required_imports),
                'classes': self.metadata.classes,
                'functions': self.metadata.functions,
                'dependencies': list(self.metadata.dependencies)
            })

            # Store in cache if cache_key is provided
            if cache_key and hasattr(self, 'cache'):
                await self.cache.set(cache_key, result)
                self.logger.debug("Cached result for key: %s", cache_key)

            return result

        except Exception as e:
            self.logger.error("Error processing code: %s", str(e))
            return None
        
    def _analyze_tree(self, tree: ast.AST) -> None:
        """
        Analyze AST to extract required imports and other metadata.

        Args:
            tree: The AST to analyze
        """
        try:
            for node in ast.walk(tree):
                # Handle function definitions (both sync and async)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_info = {
                        'name': node.name,
                        'args': self._get_function_args(node),
                        'is_async': isinstance(node, ast.AsyncFunctionDef),
                        'line_number': node.lineno
                    }
                    # Fix: Use self.metadata.functions instead of self.functions
                    self.metadata.functions.append(func_info)
                    self.logger.debug("Found function: %s", node.name)  # Use % formatting

                # Handle classes
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'bases': [self._get_name(base) for base in node.bases],
                        'methods': [m.name for m in node.body 
                                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))],
                        'line_number': node.lineno
                    }
                    # Fix: Use self.metadata.classes
                    self.metadata.classes.append(class_info)
                    self.logger.debug("Found class: %s", node.name)  # Use % formatting

                # Handle imports
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._process_imports(node)

                # Handle potential datetime usage
                elif isinstance(node, ast.Name):
                    if node.id in {'datetime', 'timedelta'}:
                        self.metadata.required_imports.add('datetime')

        except Exception as e:
            self.logger.error("Error analyzing AST: %s", str(e))
    def _get_annotation(self, node: Optional[ast.AST]) -> Optional[str]:
        """
        Extract type annotation from AST node.
        
        Args:
            node: AST node containing type annotation
            
        Returns:
            String representation of the type annotation or None if not present
        """
        if node is None:
            return None
            
        try:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Constant):
                return str(node.value)
            elif isinstance(node, ast.Attribute):
                return f"{self._get_name(node.value)}.{node.attr}"
            elif isinstance(node, ast.Subscript):
                return f"{self._get_name(node.value)}[{self._get_annotation(node.slice)}]"
            elif isinstance(node, ast.Tuple):
                elts = [self._get_annotation(elt) for elt in node.elts]
                return f"Tuple[{', '.join(filter(None, elts))}]"
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                # Handle Union types written with | operator (Python 3.10+)
                left = self._get_annotation(node.left)
                right = self._get_annotation(node.right)
                return f"Union[{left}, {right}]"
            else:
                return ast.unparse(node)  # Fallback for complex annotations
                
        except Exception as e:
            self.logger.warning("Failed to parse type annotation: %s", str(e))
            return None
        
    def _get_function_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict[str, Any]]:
        """Extract function argument information from AST node."""
        args = []
        for arg in node.args.args:
            arg_info = {
                'name': arg.arg,
                'annotation': self._get_annotation(arg.annotation),
                'has_default': arg.lineno in node.args.defaults
            }
            args.append(arg_info)
        return args
  
    def _get_name(self, node: ast.AST) -> str:
        """
        Get name from AST node.
        
        Args:
            node: AST node to extract name from
            
        Returns:
            Extracted name as string
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)
  
    def _process_imports(self, node: ast.AST) -> None:
        """
        Process import nodes to track dependencies.
        
        Args:
            node: AST node representing an import statement
        """
        try:
            if isinstance(node, ast.Import):
                for name in node.names:
                    # Fix: Use self.metadata.dependencies instead of self.dependencies
                    self.metadata.dependencies.add(name.name)
                    self.logger.debug("Found import: %s", name.name)  # Use % formatting
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Fix: Use self.metadata.dependencies instead of self.dependencies
                    self.metadata.dependencies.add(node.module)
                    self.logger.debug("Found import from: %s", node.module)  # Use % formatting
        except Exception as e:
            self.logger.error("Error processing imports: %s", str(e)) 
  
    def _add_parents(self, node: ast.AST) -> None:  
        """Add parent references to AST nodes."""  
        for child in ast.iter_child_nodes(node):  
            child.parent = node  
            self._add_parents(child)  