import ast
import importlib.util
import sys
from collections import defaultdict
from typing import Dict, Set, Optional, List, Tuple
from core.logger import LoggerSetup
from core.types import ExtractionContext

logger = LoggerSetup.get_logger(__name__)

def extract_dependencies_from_node(node: ast.AST) -> Dict[str, Set[str]]:
    """Extract dependencies from an AST node.

    Args:
        node (ast.AST): The AST node to analyze.

    Returns:
        Dict[str, Set[str]]: A dictionary categorizing dependencies.
    """
    dependencies = {
        'imports': set(),
        'calls': set(),
        'attributes': set()
    }
    for child in ast.walk(node):
        if isinstance(child, ast.Import):
            for name in child.names:
                dependencies['imports'].add(name.name)
        elif isinstance(child, ast.ImportFrom) and child.module:
            dependencies['imports'].add(child.module)
        elif isinstance(child, ast.Call):
            dependencies['calls'].add(get_node_name(child.func))
        elif isinstance(child, ast.Attribute):
            dependencies['attributes'].add(get_node_name(child))
    return dependencies

def get_node_name(node: Optional[ast.AST]) -> str:
    """Get a string representation of a node's name."""
    if node is None:
        return "Any"
    try:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{get_node_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            value = get_node_name(node.value)
            slice_val = get_node_name(node.slice)
            return f"{value}[{slice_val}]"
        elif isinstance(node, ast.Call):
            return f"{get_node_name(node.func)}()"
        elif isinstance(node, (ast.Tuple, ast.List)):
            elements = ', '.join(get_node_name(e) for e in node.elts)
            return f"({elements})" if isinstance(node, ast.Tuple) else f"[{elements}]"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif hasattr(ast, 'unparse'):
            return ast.unparse(node)
        else:
            return f"Unknown<{type(node).__name__}>"
    except Exception as e:
        logger.error(f"Error getting name from node {type(node).__name__}: {e}", exc_info=True)
        return f"Unknown<{type(node).__name__}>"

class DependencyAnalyzer:
    """Analyzes and categorizes code dependencies."""

    def __init__(self, context: ExtractionContext):
        """Initialize the DependencyAnalyzer.

        Args:
            context (ExtractionContext): The extraction context containing settings and configurations.
        """
        self.logger = logger
        self.context = context
        self.module_name = context.module_name
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
        self.module_name = module_name or self.module_name

        try:
            # Use the shared utility function for dependency extraction
            raw_deps = extract_dependencies_from_node(node)
            # Categorize dependencies as stdlib, third-party, or local
            deps = self._categorize_dependencies(raw_deps)

            circular_deps = self._detect_circular_dependencies(deps)
            if circular_deps:
                self.logger.warning(f"Circular dependencies detected: {circular_deps}")

            self.logger.info(f"Dependency analysis completed: {len(deps)} categories found")
            return deps

        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {str(e)}", exc_info=True)
            return {'stdlib': set(), 'third_party': set(), 'local': set()}

    def _categorize_dependencies(self, raw_deps: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Categorize raw dependencies into stdlib, third-party, or local.

        Args:
            raw_deps (Dict[str, Set[str]]): The raw dependencies extracted from the AST node.

        Returns:
            Dict[str, Set[str]]: Categorized dependencies.
        """
        categorized_deps = {
            'stdlib': set(),
            'third_party': set(),
            'local': set()
        }

        for module_name in raw_deps.get('imports', []):
            self._categorize_import(module_name, categorized_deps)

        return categorized_deps

    def _categorize_import(self, module_name: str, deps: Dict[str, Set[str]]) -> None:
        """Categorize an import as stdlib, third-party, or local.

        Args:
            module_name (str): The name of the module being imported.
            deps (Dict[str, Set[str]]): A dictionary to store categorized dependencies.
        """
        self.logger.debug(f"Categorizing import: {module_name}")

        try:
            # Check if the module is in the standard library
            if module_name in sys.builtin_module_names or module_name in self._get_stdlib_modules():
                deps['stdlib'].add(module_name)
                return

            # Check if it's a local module
            if self.module_name:
                module_parts = self.module_name.split('.')
                if module_name.startswith('.') or module_name.startswith(self.module_name):
                    deps['local'].add(module_name)
                    return
                for i in range(len(module_parts)):
                    potential_module = '.'.join(module_parts[:i] + [module_name])
                    if self._module_exists(potential_module):
                        deps['local'].add(module_name)
                        return

            # If it's not standard library or local, consider it third-party
            deps['third_party'].add(module_name)

        except Exception as e:
            self.logger.warning(f"Non-critical error categorizing import {module_name}: {e}", exc_info=True)
            deps['third_party'].add(module_name)

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
            Set[str]: A set containing the names of standard library modules.
        """
        if hasattr(sys, 'stdlib_module_names'):  # Python 3.10+
            return set(sys.stdlib_module_names)
        else:
            # For earlier versions, we can approximate
            import sysconfig
            paths = sysconfig.get_paths()
            stdlib_dir = paths.get('stdlib')
            modules = set()
            if stdlib_dir:
                import os
                for root, dirs, files in os.walk(stdlib_dir):
                    for file in files:
                        if file.endswith('.py'):
                            module = os.path.splitext(file)[0]
                            modules.add(module)
            return modules

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
            # Simplistic detection: check if the module depends on itself
            for dep in dependencies.get('local', set()):
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
            dependencies (Dict[str, Set[str]]): A dictionary of module dependencies.
            output_file (str): The file path to save the graph.
        """
        self.logger.debug("Generating dependency graph.")
        try:
            from graphviz import Digraph
        except ImportError:
            self.logger.warning("Graphviz not installed. Skipping dependency graph generation.")
            return

        try:
            dot = Digraph(comment='Module Dependencies')

            # Add nodes and edges
            for category, modules in dependencies.items():
                with dot.subgraph(name=f"cluster_{category}") as sub:
                    sub.attr(label=category)
                    for module in modules:
                        sub.node(module)

            # For simplicity, not adding edges between modules in this example

            # Render graph
            dot.render(output_file, view=False, cleanup=True)
            self.logger.info(f"Dependency graph saved to {output_file}")

        except Exception as e:
            self.logger.error(f"Error generating dependency graph: {e}")
            self.logger.error("Make sure Graphviz is installed and in your system PATH")