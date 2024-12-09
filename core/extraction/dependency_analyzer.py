"""
Dependency analysis module for Python source code.

This module provides functionality to analyze and categorize dependencies
within Python source code using the Abstract Syntax Tree (AST).
"""

import ast
import sys
import importlib.util
import sysconfig
from typing import Dict, Set, Optional, List, Tuple, Any
from pathlib import Path

from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.types import ExtractionContext
from utils import (
    NodeNameVisitor,
    handle_extraction_error,
    check_module_exists,
    get_module_path,
    get_node_name
)


class DependencyAnalyzer:
    """Analyzes and categorizes code dependencies."""

    def __init__(
        self,
        context: ExtractionContext,
        correlation_id: Optional[str] = None
    ) -> None:
        """Initialize the dependency analyzer."""
        self.logger = CorrelationLoggerAdapter(LoggerSetup.get_logger(__name__), correlation_id)
        self.context = context
        self.module_name = context.module_name
        self._function_errors: List[str] = []
        self._stdlib_modules: Optional[Set[str]] = None

    def analyze_dependencies(
        self,
        node: ast.AST,
        module_name: Optional[str] = None
    ) -> Dict[str, Set[str]]:
        """Analyze dependencies in an AST node."""
        try:
            if module_name:
                self.module_name = module_name

            # Extract raw dependencies
            raw_deps = self.extract_dependencies(node)

            # Categorize dependencies
            categorized_deps = self._categorize_dependencies(raw_deps)

            # Detect circular dependencies
            circular_deps = self._detect_circular_dependencies(categorized_deps)
            if circular_deps:
                self.logger.warning(
                    f"Circular dependencies detected: {circular_deps}",
                    extra={'dependencies': circular_deps}
                )

            # Calculate maintainability impact
            impact = self._calculate_maintainability_impact(categorized_deps)
            categorized_deps["maintainability_impact"] = impact

            return categorized_deps

        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {e}", exc_info=True)
            return {"stdlib": set(), "third_party": set(), "local": set()}

    def extract_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Extract dependencies from an AST node."""
        dependencies = {
            "imports": set(),
            "calls": set(),
            "attributes": set(),
        }
        for child in ast.walk(node):
            try:
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    if isinstance(child, ast.Import):
                        for name in child.names:
                            dependencies["imports"].add(name.name)
                    elif child.module:
                        dependencies["imports"].add(child.module)
                elif isinstance(child, ast.Call):
                    visitor = NodeNameVisitor()
                    visitor.visit(child.func)
                    dependencies["calls"].add(visitor.name)
                elif isinstance(child, ast.Attribute):
                    visitor = NodeNameVisitor()
                    visitor.visit(child)
                    dependencies["attributes"].add(visitor.name)
            except Exception as e:
                self.logger.debug(f"Error extracting dependency: {e}")
        return dependencies

    def _categorize_dependencies(
        self,
        raw_deps: Dict[str, Set[str]]
    ) -> Dict[str, Set[str]]:
        """Categorize dependencies into stdlib, third-party, and local."""
        categorized = {
            "stdlib": set(),
            "third_party": set(),
            "local": set()
        }

        for module_name in raw_deps.get("imports", set()):
            if self._is_stdlib_module(module_name):
                categorized["stdlib"].add(module_name)
            elif self._is_local_module(module_name):
                categorized["local"].add(module_name)
            else:
                categorized["third_party"].add(module_name)

        return categorized

    def _is_stdlib_module(self, module_name: str) -> bool:
        """Check if a module is part of the standard library."""
        if self._stdlib_modules is None:
            self._stdlib_modules = self._get_stdlib_modules()

        return (
            module_name in sys.builtin_module_names or
            module_name in self._stdlib_modules
        )

    def _is_local_module(self, module_name: str) -> bool:
        """Check if a module is local to the project."""
        if not self.module_name:
            return False

        # Check if it's a relative import
        if module_name.startswith('.'):
            return True

        # Check if it's a submodule of the current package
        if module_name.startswith(self.module_name.split('.')[0]):
            return True

        # Check if the module exists in the project directory
        if self.context.base_path:
            module_path = self.context.base_path / f"{module_name.replace('.', '/')}.py"
            return module_path.exists()

        return False

    def _get_stdlib_modules(self) -> Set[str]:
        """Get a set of standard library module names."""
        stdlib_modules = set()

        # For Python 3.10+, use built-in stdlib_module_names
        if hasattr(sys, "stdlib_module_names"):
            return set(sys.stdlib_module_names)

        try:
            # Get standard library path
            paths = sysconfig.get_paths()
            stdlib_dir = paths.get("stdlib")

            if not stdlib_dir:
                self.logger.warning("Could not find stdlib directory")
                return stdlib_modules

            # Walk through stdlib directory
            stdlib_path = Path(stdlib_dir)
            for path in stdlib_path.rglob("*.py"):
                module_name = path.stem
                if module_name != "__init__":
                    stdlib_modules.add(module_name)

            return stdlib_modules

        except Exception as e:
            self.logger.error(f"Error getting stdlib modules: {e}", exc_info=True)
            return set()

    def _detect_circular_dependencies(
        self,
        dependencies: Dict[str, Set[str]]
    ) -> List[Tuple[str, str]]:
        """Detect circular dependencies in the module."""
        circular_deps: List[Tuple[str, str]] = []
        visited: Set[str] = set()
        path: Set[str] = set()

        def visit(module: str) -> None:
            if module in path:
                if self.module_name:
                    circular_deps.append((module, self.module_name))
                return
            if module in visited:
                return

            visited.add(module)
            path.add(module)

            # Check dependencies of the current module
            for dep_type in ["local", "third_party"]:
                for dep in dependencies.get(dep_type, set()):
                    visit(dep)

            path.remove(module)

        try:
            if self.module_name:
                visit(self.module_name)
        except Exception as e:
            self.logger.error(f"Error detecting circular dependencies: {e}", exc_info=True)

        return circular_deps

    def _calculate_maintainability_impact(
        self,
        dependencies: Dict[str, Set[str]]
    ) -> float:
        """Calculate the impact of dependencies on maintainability."""
        try:
            # Count dependencies by type
            stdlib_count = len(dependencies.get("stdlib", set()))
            third_party_count = len(dependencies.get("third_party", set()))
            local_count = len(dependencies.get("local", set()))

            total_deps = stdlib_count + third_party_count + local_count
            if total_deps == 0:
                return 100.0

            # Calculate impact score
            # - Third-party dependencies have highest impact (weight: 2.0)
            # - Local dependencies have medium impact (weight: 1.5)
            # - Stdlib dependencies have lowest impact (weight: 1.0)
            impact_score = 100.0 - (
                (third_party_count * 2.0) +
                (local_count * 1.5) +
                (stdlib_count * 1.0)
            )

            # Normalize score between 0 and 100
            return max(0.0, min(impact_score, 100.0))

        except Exception as e:
            self.logger.error(f"Error calculating maintainability impact: {e}", exc_info=True)
            return 0.0

    def generate_dependency_graph(self) -> Optional[str]:
        """Generate a visual representation of dependencies."""
        try:
            import graphviz

            # Create a new directed graph
            dot = graphviz.Digraph(comment='Module Dependencies')
            dot.attr(rankdir='LR')

            # Add nodes and edges based on dependencies
            if self.context.tree:
                deps = self.analyze_dependencies(self.context.tree)

                # Add current module
                if self.module_name:
                    dot.node(self.module_name, self.module_name, shape='box')

                # Add dependencies with different colors by type
                colors = {
                    "stdlib": "lightblue",
                    "third_party": "lightgreen",
                    "local": "lightyellow"
                }

                for dep_type, deps_set in deps.items():
                    if dep_type != "maintainability_impact":
                        for dep in deps_set:
                            dot.node(dep, dep, fillcolor=colors.get(dep_type, "white"),
                                       style="filled")
                            if self.module_name:
                                dot.edge(self.module_name, dep)

            # Return the graph in DOT format
            return dot.source

        except ImportError:
            self.logger.warning("graphviz package not installed, cannot generate graph")
            return None
        except Exception as e:
            self.logger.error(f"Error generating dependency graph: {e}", exc_info=True)
            return None

    def get_dependency_metrics(self) -> Dict[str, Any]:
        """Get metrics about the module's dependencies."""
        try:
            if not self.context.tree:
                return {}

            deps = self.analyze_dependencies(self.context.tree)

            return {
                "total_dependencies": sum(len(deps[k]) for k in ["stdlib", "third_party", "local"]),
                "stdlib_count": len(deps.get("stdlib", set())),
                "third_party_count": len(deps.get("third_party", set())),
                "local_count": len(deps.get("local", set())),
                "maintainability_impact": deps.get("maintainability_impact", 0.0),
                "has_circular_dependencies": bool(self._detect_circular_dependencies(deps)),
            }

        except Exception as e:
            self.logger.error(f"Error getting dependency metrics: {e}", exc_info=True)
            return {}

    async def analyze_project_dependencies(self, project_root: Path) -> Dict[str, Any]:
        """Analyze dependencies across an entire project."""
        try:
            project_deps = {
                "modules": {},
                "global_metrics": {
                    "total_modules": 0,
                    "total_dependencies": 0,
                    "avg_maintainability": 0.0,
                    "circular_dependencies": []
                }
            }

            # Analyze each Python file in the project
            for py_file in project_root.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source = f.read()

                    tree = ast.parse(source)
                    module_name = py_file.stem

                    # Analyze dependencies for this module
                    deps = self.analyze_dependencies(tree, module_name)
                    metrics = self.get_dependency_metrics()

                    project_deps["modules"][module_name] = {
                        "dependencies": deps,
                        "metrics": metrics
                    }

                    # Update global metrics
                    project_deps["global_metrics"]["total_modules"] += 1
                    project_deps["global_metrics"]["total_dependencies"] += metrics["total_dependencies"]

                except Exception as e:
                    self.logger.error(f"Error analyzing {py_file}: {e}")

            # Calculate average maintainability
            if project_deps["global_metrics"]["total_modules"] > 0:
                total_maintainability = sum(
                    m["metrics"].get("maintainability_impact", 0)
                    for m in project_deps["modules"].values()
                )
                project_deps["global_metrics"]["avg_maintainability"] = (
                    total_maintainability / project_deps["global_metrics"]["total_modules"]
                )

            return project_deps

        except Exception as e:
            self.logger.error(f"Error analyzing project dependencies: {e}", exc_info=True)
            return {}