"""
Dependency analysis module for Python source code.
"""

import ast
import sys
import pkgutil
import sysconfig
from typing import Dict, Set, Optional, List, Tuple, Any
from pathlib import Path

from core.logger import CorrelationLoggerAdapter, LoggerSetup, get_correlation_id
from core.types import ExtractionContext
from utils import handle_extraction_error, get_node_name
from core.exceptions import ExtractionError


class DependencyAnalyzer:
    """Analyzes and categorizes code dependencies."""

    def __init__(
        self, context: ExtractionContext, correlation_id: Optional[str] = None
    ) -> None:
        """Initialize the dependency analyzer."""
        self.logger = LoggerSetup.get_logger(__name__, correlation_id)
        self.context = context
        self.module_name = context.module_name
        self.errors: List[str] = []
        self._stdlib_modules: Optional[Set[str]] = None

    def analyze_dependencies(
        self, node: ast.AST, module_name: Optional[str] = None
    ) -> Dict[str, Set[str]]:
        """Analyze dependencies in an AST node."""
        try:
            if module_name:
                self.module_name = module_name

            raw_deps = self._extract_dependencies(node)
            categorized_deps = self._categorize_dependencies(raw_deps)
            self.logger.debug(
                f"Categorized dependencies: {categorized_deps}",
                extra={"correlation_id": get_correlation_id()},
            )

            circular_deps = self.detect_circular_dependencies(categorized_deps)
            if circular_deps:
                self.logger.warning(f"Circular dependencies detected: {circular_deps}")

            impact = self._calculate_maintainability_impact(categorized_deps)
            categorized_deps["maintainability_impact"] = impact

            self.logger.debug(
                f"Formatted dependencies for AI service: {categorized_deps}",
                extra={"correlation_id": get_correlation_id()},
            )
            if not isinstance(categorized_deps, dict):
                self.logger.error(
                    "Categorized dependencies are not a dictionary. Formatting it as an empty dictionary.",
                    extra={"correlation_id": get_correlation_id()},
                )
                categorized_deps = {"stdlib": set(), "third_party": set(), "local": set()}
            return categorized_deps

        except Exception as e:
            handle_extraction_error(self.logger, self.errors, "dependency_analysis", e)
            self.logger.error(
                "Failed to analyze dependencies. Returning empty structure.",
                extra={"correlation_id": get_correlation_id()},
            )
            return {
                "stdlib": set(),
                "third_party": set(),
                "local": set(),
            }

    def _extract_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Extract raw dependencies from AST node."""
        dependencies: Dict[str, Set[str]] = {
            "imports": set(),
            "calls": set(),
            "attributes": set(),
        }

        for child in ast.walk(node):
            try:
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    for alias in getattr(
                        child, "names", []
                    ):  # Use getattr to prevent AttributeError
                        dependencies["imports"].add(alias.name)
                    module = getattr(
                        child, "module", None
                    )  # Prevent AttributeError if child.module doesn't exist
                    if module:
                        dependencies["imports"].add(module)
                elif isinstance(child, ast.Call):
                    dependencies["calls"].add(get_node_name(child.func))
                elif isinstance(child, ast.Attribute):
                    dependencies["attributes"].add(get_node_name(child))

            except Exception as e:  # Handle individual errors during extraction
                handle_extraction_error(
                    self.logger, self.errors, "dependency_item_extraction", e
                )
                if self.context.strict_mode:
                    raise

        return dependencies

    def _categorize_dependencies(
        self, raw_deps: Dict[str, Set[str]]
    ) -> Dict[str, Set[str]]:
        """Categorize dependencies."""
        categorized: Dict[str, Set[str]] = {
            "stdlib": set(),
            "third_party": set(),
            "local": set(),
        }

        for module in raw_deps["imports"]:
            if self._is_stdlib_module(module):
                categorized["stdlib"].add(module)
            elif self._is_local_module(module):
                categorized["local"].add(module)
            else:
                categorized["third_party"].add(module)

        return categorized

    def _is_stdlib_module(self, module_name: str) -> bool:
        """Check if module is in standard library."""
        if self._stdlib_modules is None:  # compute this once
            self._stdlib_modules = {
                name for _, name, ispkg in pkgutil.iter_modules(sys.modules) if ispkg
            }

        return (
            module_name in sys.builtin_module_names
            or module_name in self._stdlib_modules
        )

    def _is_local_module(self, module_name: str) -> bool:
        """Check if module is local."""
        # Handle cases where module_name is not set
        if not self.module_name:
            return False

        if module_name.startswith("."):  # Relative import
            return True

        if module_name.startswith(
            self.module_name.split(".")[0]
        ):  # Submodule of current package
            return True

        if self.context.base_path:
            module_path = self.context.base_path / f"{module_name.replace('.', '/')}.py"
            return module_path.exists()

        return False

    def detect_circular_dependencies(
        self, dependencies: Dict[str, Set[str]]
    ) -> List[Tuple[str, str]]:
        if not isinstance(dependencies, dict):
            raise TypeError(
                f"Expected 'dependencies' to be a dictionary, got {type(dependencies).__name__}"
            )
        """Detect circular dependencies."""

        circular_deps: List[Tuple[str, str]] = []
        visited: Set[str] = set()
        path: Set[str] = set()

        def visit(module: str) -> None:
            """Inner function to perform depth-first search."""
            if module in path:
                circular_deps.append(
                    (module, self.module_name)
                )  # Found circular dependency
                return

            if module in visited:
                return

            visited.add(module)
            path.add(module)

            local_deps = dependencies.get("local", set())
            if local_deps:
                for dep in local_deps:
                    if (
                        dep != module and (module, dep) not in circular_deps
                    ):  # Skip self and already found
                        visit(dep)

            path.remove(module)

        if self.module_name:
            visit(self.module_name)

        return circular_deps

    def _calculate_maintainability_impact(
        self, dependencies: Dict[str, Set[str]]
    ) -> float:
        """Calculate maintainability impact of dependencies."""
        try:
            stdlib_count = len(dependencies.get("stdlib", set()))
            third_party_count = len(dependencies.get("third_party", set()))
            local_count = len(dependencies.get("local", set()))

            total_deps = stdlib_count + third_party_count + local_count
            if total_deps == 0:
                return 100.0  # No dependencies, maximum maintainability

            # Use configuration for weights
            stdlib_weight = self.context.config.get("maintainability_weights", {}).get(
                "stdlib", 1.0
            )
            third_party_weight = self.context.config.get(
                "maintainability_weights", {}
            ).get("third_party", 2.0)
            local_weight = self.context.config.get("maintainability_weights", {}).get(
                "local", 1.5
            )

            impact = (
                100.0
                - (
                    (third_party_count * third_party_weight)
                    + (local_count * local_weight)
                    + (stdlib_count * stdlib_weight)
                )
                / total_deps
                * 100
            )
            return max(0.0, min(100.0, impact))  # Normalize between 0 and 100

        except Exception as e:
            handle_extraction_error(
                self.logger, self.errors, "maintainability_calculation", e
            )
            return 0.0

    def generate_dependency_graph(self) -> Optional[str]:
        """Generate a visual representation of dependencies."""
        try:
            import graphviz

            # Create a new directed graph
            dot = graphviz.Digraph(comment="Module Dependencies")
            dot.attr(rankdir="LR")

            # Add nodes and edges based on dependencies
            if self.context.tree:
                deps = self.analyze_dependencies(self.context.tree)

                # Add current module
                if self.module_name:
                    dot.node(self.module_name, self.module_name, shape="box")

                # Add dependencies with different colors by type
                colors = {
                    "stdlib": "lightblue",
                    "third_party": "lightgreen",
                    "local": "lightyellow",
                }

                for dep_type, deps_set in deps.items():
                    if dep_type != "maintainability_impact":
                        for dep in deps_set:
                            dot.node(
                                dep,
                                dep,
                                fillcolor=colors.get(dep_type, "white"),
                                style="filled",
                            )
                            if self.module_name:
                                dot.edge(self.module_name, dep)

            # Return the graph in DOT format
            return dot.source

        except ImportError:
            self.logger.warning("graphviz package not installed, cannot generate graph")
            return None
        except Exception as e:
            handle_extraction_error(self.logger, self.errors, "graph_generation", e)
            return None

    def get_dependency_metrics(self) -> Dict[str, Any]:
        """Get metrics about the module's dependencies."""
        try:
            if not self.context.tree:
                return {}

            deps = self.analyze_dependencies(self.context.tree)

            return {
                "total_dependencies": sum(
                    len(deps[k]) for k in ["stdlib", "third_party", "local"]
                ),
                "stdlib_count": len(deps.get("stdlib", set())),
                "third_party_count": len(deps.get("third_party", set())),
                "local_count": len(deps.get("local", set())),
                "maintainability_impact": deps.get("maintainability_impact", 0.0),
                "has_circular_dependencies": bool(
                    self._detect_circular_dependencies(deps)
                ),
            }

        except Exception as e:
            handle_extraction_error(self.logger, self.errors, "dependency_metrics", e)
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
                    "circular_dependencies": [],
                },
            }

            # Analyze each Python file in the project
            for py_file in project_root.rglob("*.py"):
                try:
                    with open(py_file, "r", encoding="utf-8") as f:
                        source = f.read()

                    tree = ast.parse(source)
                    module_name = py_file.stem

                    # Analyze dependencies for this module
                    deps = self.analyze_dependencies(tree, module_name)
                    metrics = self.get_dependency_metrics()

                    project_deps["modules"][module_name] = {
                        "dependencies": deps,
                        "metrics": metrics,
                    }

                    # Update global metrics
                    project_deps["global_metrics"]["total_modules"] += 1
                    project_deps["global_metrics"]["total_dependencies"] += metrics[
                        "total_dependencies"
                    ]

                except Exception as e:
                    handle_extraction_error(
                        self.logger,
                        self.errors,
                        "project_dependency_analysis",
                        e,
                        file_path=str(py_file),
                    )

            # Calculate average maintainability
            if project_deps["global_metrics"]["total_modules"] > 0:
                total_maintainability = sum(
                    m["metrics"].get("maintainability_impact", 0)
                    for m in project_deps["modules"].values()
                )
                project_deps["global_metrics"]["avg_maintainability"] = (
                    total_maintainability
                    / project_deps["global_metrics"]["total_modules"]
                )

            # Detect circular dependencies across the project
            all_local_deps = set()
            for module, data in project_deps["modules"].items():
                all_local_deps.update(data["dependencies"].get("local", set()))

            circular_deps = []
            for module in all_local_deps:
                if module in project_deps["modules"]:
                    visited = set()
                    path = set()

                    def visit(mod: str) -> None:
                        if mod in path:
                            circular_deps.append((mod, module))
                            return
                        if mod in visited:
                            return

                        visited.add(mod)
                        path.add(mod)

                        local_deps = (
                            project_deps["modules"]
                            .get(mod, {})
                            .get("dependencies", {})
                            .get("local", set())
                        )
                        for dep in local_deps:
                            if dep != mod and (mod, dep) not in circular_deps:
                                visit(dep)

                        path.remove(mod)

                    visit(module)

            project_deps["global_metrics"]["circular_dependencies"] = circular_deps

            return project_deps

        except Exception as e:
            handle_extraction_error(
                self.logger, self.errors, "project_dependency_analysis", e
            )
            return {}
