"""Dependency analysis module for Python source code."""

import ast
import importlib.util
import os
import sys
from typing import Dict, Set, Optional, List, Tuple

from core.logger import LoggerSetup
from core.types import ExtractionContext
from core.utils import get_node_name  # Ensure this import is present

logger = LoggerSetup.get_logger(__name__)

def extract_dependencies_from_node(node: ast.AST) -> Dict[str, Set[str]]:
    """Extract dependencies from an AST node."""
    dependencies = {"imports": set(), "calls": set(), "attributes": set()}
    for child in ast.walk(node):
        if isinstance(child, ast.Import):
            for name in child.names:
                dependencies["imports"].add(name.name)
        elif isinstance(child, ast.ImportFrom) and child.module:
            dependencies["imports"].add(child.module)
        elif isinstance(child, ast.Call):
            dependencies["calls"].add(get_node_name(child.func))
        elif isinstance(child, ast.Attribute):
            dependencies["attributes"].add(get_node_name(child))
    return dependencies

class DependencyAnalyzer:
    """Analyzes and categorizes code dependencies."""

    def __init__(self, context: ExtractionContext):
        """Initialize the DependencyAnalyzer."""
        self.logger = logger
        self.context = context
        self.module_name = context.module_name
        self.logger.debug("Initialized DependencyAnalyzer")

    def analyze_dependencies(
        self, node: ast.AST, module_name: Optional[str] = None
    ) -> Dict[str, Set[str]]:
        """Analyze module dependencies, including circular dependency detection."""
        self.logger.info("Starting dependency analysis")
        self.module_name = module_name or self.module_name

        try:
            raw_deps = extract_dependencies_from_node(node)
            deps = self._categorize_dependencies(raw_deps)
            circular_deps = self._detect_circular_dependencies(deps)
            if circular_deps:
                self.logger.warning(f"Circular dependencies detected: {circular_deps}")

            self.logger.info(
                f"Dependency analysis completed: {len(deps)} categories found"
            )
            return deps

        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {str(e)}", exc_info=True)
            return {"stdlib": set(), "third_party": set(), "local": set()}

    def _categorize_dependencies(
        self, raw_deps: Dict[str, Set[str]]
    ) -> Dict[str, Set[str]]:
        """Categorize raw dependencies into stdlib, third-party, or local."""
        categorized_deps = {"stdlib": set(), "third_party": set(), "local": set()}

        for module_name in raw_deps.get("imports", []):
            self._categorize_import(module_name, categorized_deps)

        return categorized_deps

    def _categorize_import(self, module_name: str, deps: Dict[str, Set[str]]) -> None:
        """Categorize an import as stdlib, third-party, or local."""
        self.logger.debug(f"Categorizing import: {module_name}")

        try:
            if (
                module_name in sys.builtin_module_names
                or module_name in self._get_stdlib_modules()
            ):
                deps["stdlib"].add(module_name)
                return

            if self.module_name:
                module_parts = self.module_name.split(".")
                if module_name.startswith(".") or module_name.startswith(
                    self.module_name
                ):
                    deps["local"].add(module_name)
                    return
                for i in range(len(module_parts)):
                    potential_module = ".".join(module_parts[:i] + [module_name])
                    if self._module_exists(potential_module):
                        deps["local"].add(module_name)
                        return

            deps["third_party"].add(module_name)

        except Exception as e:
            self.logger.warning(
                f"Non-critical error categorizing import {module_name}: {e}",
                exc_info=True,
            )
            deps["third_party"].add(module_name)

    def _module_exists(self, module_name: str) -> bool:
        """Check if a module exists."""
        spec = importlib.util.find_spec(module_name)
        return spec is not None

    def _get_stdlib_modules(self) -> Set[str]:
        """Get a set of standard library module names."""
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

    def _detect_circular_dependencies(
        self, dependencies: Dict[str, Set[str]]
    ) -> List[Tuple[str, str]]:
        """Detect circular dependencies."""
        self.logger.debug("Detecting circular dependencies")
        circular_dependencies = []
        try:
            for dep in dependencies.get("local", set()):
                if dep == self.module_name:
                    circular_dependencies.append((self.module_name, dep))
        except Exception as e:
            self.logger.error(
                f"Error detecting circular dependencies: {e}", exc_info=True
            )

        if circular_dependencies:
            self.logger.debug(f"Circular dependencies: {circular_dependencies}")
        return circular_dependencies

    def generate_dependency_graph(
        self, dependencies: Dict[str, Set[str]], output_file: str
    ) -> None:
        """Generates a visual dependency graph."""
        self.logger.debug("Generating dependency graph.")
        try:
            from graphviz import Digraph
        except ImportError:
            self.logger.warning(
                "Graphviz not installed. Skipping dependency graph generation."
            )
            return

        try:
            dot = Digraph(comment="Module Dependencies")

            for category, modules in dependencies.items():
                with dot.subgraph(name=f"cluster_{category}") as sub:
                    sub.attr(label=category)
                    for module in modules:
                        sub.node(module)

            dot.render(output_file, view=False, cleanup=True)
            self.logger.info(f"Dependency graph saved to {output_file}")

        except Exception as e:
            self.logger.error(f"Error generating dependency graph: {e}")
            self.logger.error("Make sure Graphviz is installed and in your system PATH")