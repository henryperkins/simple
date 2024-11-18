import ast
from typing import Dict, Any, Optional, List, Union, Tuple
from core.logger import log_info, log_error, log_debug

class ExtractionManager:
    """
    Enhanced extraction manager with support for exception classes and robust error handling.
    """

    def extract_metadata(self, source_code: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract metadata from source code with special handling for exception classes.

        Args:
            source_code (str): The source code to analyze.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Extracted metadata for classes and functions.
        """
        try:
            log_debug("Starting metadata extraction")
            tree = ast.parse(source_code)
            
            classes = []
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    try:
                        # Check if this is an exception class
                        is_exception = self._is_exception_class(node)
                        metadata = self._extract_class_metadata(node, is_exception)
                        if metadata:
                            metadata['node'] = node  # Include the AST node
                            classes.append(metadata)
                        log_debug(f"Extracted {'exception ' if is_exception else ''}class: {node.name} with metadata: {metadata}")
                    except Exception as e:
                        log_error(f"Error extracting class metadata for {getattr(node, 'name', '<unknown>')}: {e}")
                        continue
                        
                elif isinstance(node, ast.FunctionDef):
                    try:
                        metadata = self._extract_function_metadata(node)
                        if metadata:
                            metadata['node'] = node  # Include the AST node
                            functions.append(metadata)
                        log_debug(f"Extracted function: {node.name} with metadata: {metadata}")
                    except Exception as e:
                        log_error(f"Error extracting function metadata for {getattr(node, 'name', '<unknown>')}: {e}")
                        continue

            log_info(f"Extraction complete. Found {len(classes)} classes and {len(functions)} functions")
            return {
                'classes': classes,
                'functions': functions
            }
            
        except SyntaxError as e:
            log_error(f"Syntax error in source code: {e}")
            return {'classes': [], 'functions': []}
        except Exception as e:
            log_error(f"Failed to extract metadata: {str(e)}")
            return {'classes': [], 'functions': []}

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """
        Determine if a class is an exception class.

        Args:
            node (ast.ClassDef): The class node to check.

        Returns:
            bool: True if the class is an exception class.
        """
        # Check class bases for Exception inheritance
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in {'Exception', 'BaseException'}:
                return True
            # Handle cases where the base exception is accessed through a module
            elif isinstance(base, ast.Attribute) and base.attr in {'Exception', 'BaseException'}:
                return True
        return False

    def _extract_class_metadata(self, node: ast.ClassDef, is_exception: bool) -> Dict[str, Any]:
        """
        Extract metadata from a class definition with special handling for exception classes.

        Args:
            node (ast.ClassDef): The class node to extract metadata from.
            is_exception (bool): Whether this is an exception class.

        Returns:
            Dict[str, Any]: Extracted metadata.
        """
        try:
            metadata = {
                'name': node.name,
                'docstring': ast.get_docstring(node) or '',
                'lineno': node.lineno,
                'is_exception': is_exception,
                'type': 'exception_class' if is_exception else 'class'
            }

            # For regular classes, extract additional metadata
            if not is_exception:
                metadata.update({
                    'methods': self._extract_methods(node),  # Use list directly
                    'bases': [self._format_base(base) for base in node.bases],  # Use list directly
                    'decorators': self._extract_decorators(node)  # Use list directly
                })
            # For exception classes, extract minimal metadata
            else:
                metadata.update({
                    'bases': [self._format_base(base) for base in node.bases],  # Use list directly
                    'error_code': self._extract_error_code(node)
                })

            return metadata

        except Exception as e:
            log_error(f"Error in class metadata extraction for {node.name}: {e}")
            # Return minimal metadata for error cases
            return {
                'name': node.name,
                'type': 'exception_class' if is_exception else 'class',
                'error': str(e)
            }

    def _extract_error_code(self, node: ast.ClassDef) -> Optional[str]:
        """
        Extract error code from an exception class if present.

        Args:
            node (ast.ClassDef): The exception class node.

        Returns:
            Optional[str]: The error code if found.
        """
        try:
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == 'code':
                            if isinstance(item.value, ast.Str):
                                return item.value.s
                            elif isinstance(item.value, ast.Constant):
                                return str(item.value.value)
            return None
        except Exception:
            return None

    def _format_base(self, node: ast.AST) -> str:
        """
        Format a base class reference.

        Args:
            node (ast.AST): The AST node representing the base class.

        Returns:
            str: Formatted base class reference.
        """
        try:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return f"{self._format_base(node.value)}.{node.attr}"
            else:
                return ast.unparse(node)
        except Exception:
            return "<unknown_base>"

    def _extract_methods(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """
        Extract method information from a class.

        Args:
            node (ast.ClassDef): The class node.

        Returns:
            List[Dict[str, Any]]: List of method metadata.
        """
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                try:
                    method_data = self._extract_function_metadata(item)
                    methods.append(method_data)
                except Exception as e:
                    log_error(f"Error extracting method {item.name}: {e}")
                    continue
        return methods

    def _extract_function_metadata(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """
        Extract metadata from a function definition.

        Args:
            node (ast.FunctionDef): The function node.

        Returns:
            Dict[str, Any]: Extracted function metadata.
        """
        try:
            return {
                'name': node.name,
                'docstring': ast.get_docstring(node) or '',
                'args': self._extract_arguments(node),
                'return_type': self._extract_return_type(node),
                'decorators': self._extract_decorators(node),
                'lineno': node.lineno
            }
        except Exception as e:
            log_error(f"Error in function metadata extraction for {node.name}: {e}")
            return {
                'name': node.name,
                'type': 'function',
                'error': str(e)
            }

    def _extract_arguments(self, node: ast.FunctionDef) -> List[Tuple[str, str]]:
        """
        Extract function arguments and their types.

        Args:
            node (ast.FunctionDef): The function node.

        Returns:
            List[Tuple[str, str]]: List of (argument_name, type_annotation) pairs.
        """
        args = []
        for arg in node.args.args:
            arg_type = "Any"
            if arg.annotation:
                try:
                    arg_type = ast.unparse(arg.annotation)
                except Exception:
                    pass
            args.append((arg.arg, arg_type))
        return args

    def _extract_return_type(self, node: ast.FunctionDef) -> str:
        """
        Extract function return type.

        Args:
            node (ast.FunctionDef): The function node.

        Returns:
            str: Return type annotation or "Any".
        """
        if node.returns:
            try:
                return ast.unparse(node.returns)
            except Exception:
                pass
        return "Any"

    def _extract_decorators(self, node: Union[ast.ClassDef, ast.FunctionDef]) -> List[str]:
        """
        Extract decorators from a class or function.

        Args:
            node (Union[ast.ClassDef, ast.FunctionDef]): The node to extract decorators from.

        Returns:
            List[str]: List of decorator strings.
        """
        decorators = []
        for decorator in node.decorator_list:
            try:
                decorators.append(ast.unparse(decorator))
            except Exception:
                decorators.append("<unknown_decorator>")
        return decorators
