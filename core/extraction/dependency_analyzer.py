"""
Dependency analysis module for Python source code.

This module provides functionality to analyze and categorize dependencies
within Python source code using the Abstract Syntax Tree (AST). It identifies
imports, function calls, and attributes used in the code, detects circular
dependencies, and can generate a visual dependency graph.

Functions:
    extract_dependencies_from_node: Extracts dependencies from an AST node.

Classes:
    DependencyAnalyzer: Analyzes and categorizes code dependencies.

Example usage:
    analyzer = DependencyAnalyzer(context)
    dependencies = analyzer.analyze_dependencies(ast_tree)
"""

import ast
import importlib.util
import os
import sys
from typing import Dict, Set, Optional, List, Tuple, Union, Any

from core.logger import LoggerSetup
from core.types import ExtractionContext
from core.utils import get_node_name

logger = LoggerSetup.get_logger(__name__)

def extract_dependencies_from_node(node: ast.AST) -> Dict[str, Set[str]]:
    """Extract dependencies from an AST node.

    Args:
        node (ast.AST): The AST node to extract dependencies from.

    Returns:
        Dict[str, Set[str]]: A dictionary with keys 'imports', 'calls', and 'attributes',
        each containing a set of strings representing the dependencies.
    """
    dependencies = {"imports": set(), "calls": set(), "attributes": set()}
    for child in ast.walk(node):
        try:
            if isinstance(child, ast.Import):
                for name in child.names:
                    dependencies["imports"].add(name.name)
            elif isinstance(child, ast.ImportFrom) and child.module:
                dependencies["imports"].add(child.module)
            elif isinstance(child, ast.Call):
                dependencies["calls"].add(get_node_name(child.func))
            elif isinstance(child, ast.Attribute):
                dependencies["attributes"].add(get_node_name(child))
        except Exception as e:
            logger.warning(f"Unsupported AST node encountered: {e}")
    return dependencies

class DependencyAnalyzer:
    """Analyzes and categorizes code dependencies.

    Attributes:
        context (ExtractionContext): The context for extraction operations.
        module_name (Optional[str]): The name of the module being analyzed.
        _function_errors (List[str]): List of errors encountered during function metadata extraction.
    """

    def __init__(self, context: ExtractionContext):
        """Initialize the DependencyAnalyzer.

        Args:
            context (ExtractionContext): The context for extraction operations.
        """
        self.logger = logger
        self.context = context
        self.module_name = context.module_name
        self._function_errors = []
        self.logger.debug("Initialized DependencyAnalyzer")

    def analyze_dependencies(self, node: ast.AST, module_name: Optional[str] = None) -> Dict[str, Set[str]]:
        """Analyze module dependencies, including circular dependency detection.

        Args:
            node (ast.AST): The AST node to analyze.
            module_name (Optional[str]): The name of the module being analyzed.

        Returns:
            Dict[str, Set[str]]: A dictionary categorizing dependencies into 'stdlib', 'third_party', and 'local'.
        """
        self.logger.info("Starting dependency analysis")
        self.module_name = module_name or self.module_name

        try:
            raw_deps = extract_dependencies_from_node(node)
            deps = self._categorize_dependencies(raw_deps)
            circular_deps = self._detect_circular_dependencies(deps)
            if circular_deps:
                self.logger.warning(f"Circular dependencies detected: {circular_deps}")

            self.logger.info(f"Dependency analysis completed: {len(deps)} categories found")
            return deps

        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {str(e)}", exc_info=True)
            return {"stdlib": set(), "third_party": set(), "local": set()}

    def extract_function_metadata(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        """Extract function metadata including raises information.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function node to extract metadata from.

        Returns:
            Dict[str, Any]: A dictionary containing function metadata such as name, docstring, raises, args, and returns.
        """
        try:
            metadata = {
                "name": node.name,
                "docstring": ast.get_docstring(node) or "",
                "raises": self._extract_raises(node),
                "args": self._extract_args(node),
                "returns": self._extract_returns(node)
            }
            return metadata
        except Exception as e:
            self._function_errors.append(f"Error extracting metadata for {getattr(node, 'name', 'unknown')}: {e}")
            return {}

    def _extract_raises(self, node: ast.AST) -> List[Dict[str, str]]:
        """Extract raise statements from function body.

        Args:
            node (ast.AST): The function node to extract raise statements from.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing exception names and descriptions.
        """
        raises = []
        for child in ast.walk(node):
            if isinstance(child, ast.Raise) and child.exc:
                exc_name = get_node_name(child.exc)
                if exc_name:
                    raises.append({
                        "exception": exc_name,
                        "description": "Exception raised in function execution"
                    })
        return raises

    def _extract_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict[str, str]]:
        """Extract arguments from function definition.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function node to extract arguments from.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing argument names, types, and descriptions.
        """
        args = []
        for arg in node.args.args:
            args.append({
                "name": arg.arg,
                "type": "Unknown",  # Type inference can be added if needed
                "description": "No description available"
            })
        return args

    def _extract_returns(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, str]:
        """Extract return type from function definition.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function node to extract return type from.

        Returns:
            Dict[str, str]: A dictionary containing the return type and description.
        """
        return {
            "type": "Unknown",  # Type inference can be added if needed
            "description": "No description available"
        }

    def _categorize_dependencies(self, raw_deps: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Categorize raw dependencies into stdlib, third-party, or local.

        Args:
            raw_deps (Dict[str, Set[str]]): The raw dependencies extracted from the AST.

        Returns:
            Dict[str, Set[str]]: A dictionary categorizing dependencies into 'stdlib', 'third_party', and 'local'.
        """
        categorized_deps = {"stdlib": set(), "third_party": set(), "local": set()}

        for module_name in raw_deps.get("imports", []):
            self._categorize_import(module_name, categorized_deps)

        return categorized_deps

    def _categorize_import(self, module_name: str, deps: Dict[str, Set[str]]) -> None:
        """Categorize an import as stdlib, third-party, or local.

        Args:
            module_name (str): The name of the module to categorize.
            deps (Dict[str, Set[str]]): The dictionary to store categorized dependencies.
        """
        self.logger.debug(f"Categorizing import: {module_name}")

        try:
            if module_name in sys.builtin_module_names or module_name in self._get_stdlib_modules():
                deps["stdlib"].add(module_name)
                return

            if self.module_name:
                module_parts = self.module_name.split(".")
                if module_name.startswith(".") or module_name.startswith(self.module_name):
                    deps["local"].add(module_name)
                    return
                for i in range(len(module_parts)):
                    potential_module = ".".join(module_parts[:i] + [module_name])
                    if self._module_exists(potential_module):
                        deps["local"].add(module_name)
                        return

            deps["third_party"].add(module_name)

        except Exception as e:
            self.logger.warning(f"Non-critical error categorizing import {module_name}: {e}", exc_info=True)
            deps["third_party"].add(module_name)

    def _module_exists(self, module_name: str) -> bool:
        """Check if a module exists.

        Args:
            module_name (str): The name of the module to check.

        Returns:
            bool: True if the module exists, False otherwise.
        """
        spec = importlib.util.find_spec(module_name)
        return spec is not None

    def _get_stdlib_modules(self) -> Set[str]:
        """Get a set of standard library module names.

        Returns:
            Set[str]: A set of standard library module names.
        """
        if hasattr(sys, "stdlib_module_names"):  # Python 3.10+
            return set(sys.stdlib_module_names)
        
        try:
            import sysconfig
            paths = sysconfig.get_paths()
            stdlib_dir = paths.get("stdlib")
            if not stdlib_dir:
                return set()
                
            modules = set()
            for root, _, files in os.walk(stdlib_dir):
                for file in files:
                    if file.endswith(".py"):
                        module = os.path.splitext(file)[0]
                        modules.add(module)
            return modules
        except Exception as e:
            logger.error("Error getting stdlib modules: %s", e)
            return set()

    def _detect_circular_dependencies(self, dependencies: Dict[str, Set[str]]) -> List[Tuple[str, str]]:
        """Detect circular dependencies.

        Args:
            dependencies (Dict[str, Set[str]]): The dependencies to check for circular references.

        Returns:
            List[Tuple[str, str]]: A list of tuples representing circular dependencies.
        """
        self.logger.debug("Detecting circular dependencies")
        circular_dependencies = []
        try:
            for dep in dependencies.get("local", set()):
                if dep == self.module_name:
                    circular_dependencies.append((self.module_name, dep))
        except Exception as e:
            self.logger.error(f"Error detecting circular dependencies: {e}", exc_info=True)

        if circular_dependencies:
            self.logger.debug(f"Circular dependencies: {circular_dependencies}")
        return circular_dependencies

    def generate_dependency_graph(self, dependencies: Dict[str, Set[str]], output_file: str) -> None:
        """Generates a visual dependency graph.

        Args:
            dependencies (Dict[str, Set[str]]): The dependencies to visualize.
            output_file (str): The file path to save the generated graph.
        """
        self.logger.debug("Generating dependency graph.")
        try:
            from graphviz import Digraph
        except ImportError:
            self.logger.warning("Graphviz not installed. Skipping dependency graph generation.")
            return

        try:
            dot = Digraph(comment="Module Dependencies")

            for category, modules in dependencies.items():
                sub = Digraph(name=f"cluster_{category}")
                sub.attr(label=category)
                for module in modules:
                    sub.node(module)
                dot.subgraph(sub)

            dot.render(output_file, view=False, cleanup=True)
            self.logger.info(f"Dependency graph saved to {output_file}")

        except Exception as e:
            self.logger.error(f"Error generating dependency graph: {e}")
            self.logger.error("Make sure Graphviz is installed and in your system PATH")

    def extract_dependencies(self, tree: ast.AST) -> List[str]:
        """Extract dependencies from AST.

        Args:
            tree (ast.AST): The AST to analyze.

        Returns:
            List[str]: A list of dependency strings.
        """
        deps = self.analyze_dependencies(tree)
        return list({dep for deps in deps.values() for dep in deps})