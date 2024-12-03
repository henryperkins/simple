"""Dependency analysis module.

This module provides functionality to analyze and categorize dependencies in Python source code.
It uses the Abstract Syntax Tree (AST) to parse and analyze the code, identifying imports and
categorizing them as standard library, third-party, or local dependencies.
"""

import ast
import sys
import importlib.util
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional
from core.logger import LoggerSetup
from .types import ExtractionContext
from .utils import ASTUtils

logger = LoggerSetup.get_logger(__name__)

class DependencyAnalyzer:
    """Analyzes and categorizes code dependencies.

    Attributes:
        context (ExtractionContext): The context for extraction, including configuration options.
        module_name (str): The name of the module being analyzed.
        _import_map (Dict[str, str]): A mapping of import aliases to their full module names.
    """

    def __init__(self, context: ExtractionContext):
        """Initialize the DependencyAnalyzer.

        Args:
            context (ExtractionContext): The extraction context containing settings and configurations.
        """
        self.logger = logger
        self.context = context
        self.ast_utils = ASTUtils()
        self.module_name = context.module_name
        self._import_map: Dict[str, str] = {}
        self.logger.debug("Initialized DependencyAnalyzer")

    def analyze_dependencies(self, node: ast.AST, module_name: Optional[str] = None) -> Dict[str, Set[str]]:
        """Analyze module dependencies, including circular dependency detection.

        Args:
            node (ast.AST): The AST node representing the module.
            module_name (Optional[str]): The name of the module being analyzed.

        Returns:
            Dict[str, Set[str]]: A dictionary categorizing dependencies as stdlib, third-party, or local.
        """
        self.logger.info("Starting dependency analysis")
        deps: Dict[str, Set[str]] = defaultdict(set)
        self.module_name = module_name or self.module_name

        try:
            for subnode in ast.walk(node):
                if isinstance(subnode, (ast.Import, ast.ImportFrom)):
                    self._process_import(subnode, deps)

            circular_deps = self._detect_circular_dependencies(deps)
            if circular_deps:
                self.logger.warning(f"Circular dependencies detected: {circular_deps}")

            self.logger.info(f"Dependency analysis completed: {len(deps)} dependencies found")
            return dict(deps)

        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {str(e)}", exc_info=True)
            return {'stdlib': set(), 'third_party': set(), 'local': set()}

    def extract_imports(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Extract and categorize imports from the AST.

        Args:
            node (ast.AST): The AST node representing the module.

        Returns:
            Dict[str, Set[str]]: A dictionary categorizing imports as stdlib, third-party, or local.
        """
        self.logger.info("Extracting imports")
        imports = {
            'stdlib': set(),
            'local': set(),
            'third_party': set()
        }

        for n in ast.walk(node):
            if isinstance(n, ast.Import):
                for name in n.names:
                    self._categorize_import(name.name, imports)
            elif isinstance(n, ast.ImportFrom):
                if n.names[0].name == '*':
                    self.logger.error(f"Star import encountered: from {n.module} import *, skipping.")
                elif n.module:
                    self._categorize_import(n.module, imports)

        self.logger.debug(f"Extracted imports: {imports}")
        return imports

    def _process_import(self, node: ast.AST, deps: Dict[str, Set[str]]) -> None:
        """Process import statements and categorize dependencies.

        Args:
            node (ast.AST): The AST node representing an import statement.
            deps (Dict[str, Set[str]]): A dictionary to store dependencies.
        """
        self.logger.debug(f"Processing import: {ast.dump(node)}")
        try:
            if isinstance(node, ast.Import):
                for name in node.names:
                    self._categorize_import(name.name, deps)
                    self._import_map[name.asname or name.name] = name.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                self._categorize_import(node.module, deps)
                for alias in node.names:
                    if alias.name != '*':
                        full_name = f"{node.module}.{alias.name}"
                        self._import_map[alias.asname or alias.name] = full_name
        except Exception as e:
            self.logger.error(f"Error processing import: {e}", exc_info=True)

    def _categorize_import(self, module_name: str, deps: Dict[str, Set[str]]) -> None:
        """Categorize an import as stdlib, third-party, or local.

        Args:
            module_name (str): The name of the module being imported.
            deps (Dict[str, Set[str]]): A dictionary to store categorized dependencies.
        """
        self.logger.debug(f"Categorizing import: {module_name}")
        try:
            if module_name in sys.stdlib_module_names or module_name.split('.')[0] in sys.stdlib_module_names:
                deps['stdlib'].add(module_name)
                return

            if self.module_name:
                current_module_parts = self.module_name.split('.')
                if any(module_name.startswith(part) for part in current_module_parts):
                    for i in range(1, len(current_module_parts) + 1):
                        test_module_name = ".".join(current_module_parts[:-i] + [module_name])
                        try:
                            if importlib.util.find_spec(test_module_name):
                                deps['local'].add(module_name)
                                return
                        except ModuleNotFoundError:
                            continue

            try:
                if importlib.util.find_spec(module_name):
                    deps['third_party'].add(module_name)
                else:
                    deps['local'].add(module_name)
            except ModuleNotFoundError:
                deps['third_party'].add(module_name)

        except Exception as e:
            self.logger.warning(f"Non-critical error categorizing import {module_name}: {e}", exc_info=True)
            deps['third_party'].add(module_name)

    def _detect_circular_dependencies(self, dependencies: Dict[str, Set[str]]) -> List[Tuple[str, str]]:
        """Detect circular dependencies.

        Args:
            dependencies (Dict[str, Set[str]]): A dictionary of module dependencies.

        Returns:
            List[Tuple[str, str]]: A list of tuples representing circular dependencies.
        """
        self.logger.debug("Detecting circular dependencies")
        circular_dependencies = []
        try:
            for module, deps in dependencies.items():
                for dep in deps:
                    if self.module_name and dep == self.module_name:
                        circular_dependencies.append((module, dep))
                    elif dep in dependencies and module in dependencies[dep]:
                        circular_dependencies.append((module, dep))
            self.logger.debug(f"Circular dependencies: {circular_dependencies}")
        except Exception as e:
            self.logger.error(f"Error detecting circular dependencies: {e}", exc_info=True)
        return circular_dependencies

    def analyze_function_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Analyze dependencies specific to a function.

        Args:
            node (ast.AST): The AST node representing a function.

        Returns:
            Dict[str, Set[str]]: A dictionary of function-specific dependencies.
        """
        self.logger.info(f"Analyzing function dependencies for node: {ast.dump(node)}")
        dependencies = {
            'imports': self._extract_function_imports(node),
            'calls': self._extract_function_calls(node),
            'attributes': self._extract_attribute_access(node)
        }
        self.logger.debug(f"Function dependencies: {dependencies}")
        return dependencies

    def _extract_function_imports(self, node: ast.AST) -> Set[str]:
        """Extract imports used within a function.

        Args:
            node (ast.AST): The AST node representing a function.

        Returns:
            Set[str]: A set of import names used within the function.
        """
        self.logger.debug("Extracting function imports")
        imports = set()
        for subnode in ast.walk(node):
            if isinstance(subnode, (ast.Import, ast.ImportFrom)):
                if isinstance(subnode, ast.Import):
                    for name in subnode.names:
                        imports.add(name.name)
                elif subnode.module:
                    imports.add(subnode.module)
        return imports

    def _extract_function_calls(self, node: ast.AST) -> Set[str]:
        """Extract function calls within a function.

        Args:
            node (ast.AST): The AST node representing a function.

        Returns:
            Set[str]: A set of function call names.
        """
        self.logger.debug("Extracting function calls")
        calls = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                try:
                    func_name = self.ast_utils.get_name(child.func)
                    calls.add(func_name)
                except Exception as e:
                    self.logger.debug(f"Could not unparse function call: {e}", exc_info=True)
        return calls

    def _extract_attribute_access(self, node: ast.AST) -> Set[str]:
        """Extract attribute accesses within a function.

        Args:
            node (ast.AST): The AST node representing a function.

        Returns:
            Set[str]: A set of attribute access names.
        """
        self.logger.debug("Extracting attribute accesses")
        attributes = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                try:
                    attr_name = self.ast_utils.get_name(child)
                    attributes.add(attr_name)
                except Exception as e:
                    self.logger.debug(f"Failed to unparse attribute access: {e}", exc_info=True)
        return attributes
