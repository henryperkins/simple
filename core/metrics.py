import ast
import math
import os
import sys
import importlib.util
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Union
from graphviz import Digraph  # For dependency graphs
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class MetricsError(Exception):
    """Base exception for metrics calculation errors."""
    pass

class Metrics:
    """
    A class to calculate various code complexity metrics for Python code.
    """

    MAINTAINABILITY_THRESHOLDS: Dict[str, int] = {
        'good': 80,
        'moderate': 60,
        'poor': 40
    }

    def __init__(self) -> None:
        """Initializes the Metrics class."""
        self.module_name: Union[str, None] = None

    def calculate_cyclomatic_complexity(self, function_node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
        """
        Calculates cyclomatic complexity for a function.

        Args:
            function_node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing a function.

        Returns:
            int: The cyclomatic complexity as an integer.
        """
        logger.debug(f"Calculating cyclomatic complexity for: {function_node.name}")
        complexity = 1
        for node in ast.walk(function_node):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.AsyncFor,
                                  ast.Try, ast.ExceptHandler, ast.With,
                                  ast.AsyncWith, ast.BoolOp)):
                complexity += 1
        logger.debug(f"Cyclomatic complexity for {function_node.name}: {complexity}")
        return complexity

    def calculate_cognitive_complexity(self, function_node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
        """
        Calculates cognitive complexity for a function.

        Args:
            function_node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing a function.

        Returns:
            int: The cognitive complexity as an integer.
        """
        logger.debug(f"Calculating cognitive complexity for: {function_node.name}")
        complexity = 0

        def _increment_complexity(node: ast.AST, nesting: int) -> None:
            nonlocal complexity
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try,
                                  ast.ExceptHandler, ast.With)):
                complexity += 1
                nesting += 1
            elif isinstance(node, (ast.BoolOp, ast.Break, ast.Continue,
                                    ast.Raise, ast.Return, ast.Yield,
                                    ast.YieldFrom)):
                complexity += nesting + 1

            for child in ast.iter_child_nodes(node):
                _increment_complexity(child, nesting)

        _increment_complexity(function_node, 0)
        logger.debug(f"Cognitive complexity for {function_node.name}: {complexity}")
        return complexity

    def calculate_halstead_metrics(self, node: ast.AST) -> Dict[str, float]:
        """
        Calculates Halstead metrics.

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            Dict[str, float]: A dictionary containing various Halstead metrics.
        """
        logger.debug("Calculating Halstead metrics.")
        operators: Set[str] = set()
        operands: Set[str] = set()
        operator_counts: Dict[str, int] = defaultdict(int)
        operand_counts: Dict[str, int] = defaultdict(int)

        for n in ast.walk(node):
            if isinstance(n, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
                              ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
                              ast.FloorDiv, ast.And, ast.Or, ast.Not, ast.Invert,
                              ast.UAdd, ast.USub, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
                              ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
                              ast.Call, ast.Attribute, ast.Subscript, ast.Assign,
                              ast.AugAssign, ast.AnnAssign, ast.Yield, ast.YieldFrom)):
                operators.add(type(n).__name__)
                operator_counts[type(n).__name__] += 1
            elif isinstance(n, (ast.Constant, ast.Name, ast.List, ast.Tuple, ast.Set, ast.Dict)):
                operand_name = self._get_operand_name(n)
                if operand_name:
                    operands.add(operand_name)
                    operand_counts[operand_name] += 1

        n1 = len(operators)
        n2 = len(operands)
        N1 = sum(operator_counts.values())
        N2 = sum(operand_counts.values())

        program_length = N1 + N2
        program_vocabulary = n1 + n2

        if program_vocabulary == 0 or n2 == 0:  # Handle potential division by zero
            logger.warning("Program vocabulary or operands are zero, returning default Halstead metrics.")
            return {
                'program_length': program_length,
                'program_vocabulary': program_vocabulary,
                'program_volume': 0.0,
                'program_difficulty': 0.0,
                'program_effort': 0.0,
                'time_required_to_program': 0.0,
                'number_delivered_bugs': 0.0
            }

        program_volume = program_length * math.log2(program_vocabulary)
        program_difficulty = (n1 / 2) * (N2 / n2)
        program_effort = program_difficulty * program_volume
        time_required_to_program = program_effort / 18  # seconds
        number_delivered_bugs = program_volume / 3000

        metrics = {
            'program_length': program_length,
            'program_vocabulary': program_vocabulary,
            'program_volume': program_volume,
            'program_difficulty': program_difficulty,
            'program_effort': program_effort,
            'time_required_to_program': time_required_to_program,
            'number_delivered_bugs': number_delivered_bugs
        }
        logger.debug(f"Halstead metrics: {metrics}")
        return metrics

    def calculate_complexity(self, node: ast.AST) -> int:
        """
        Calculates complexity for any AST node.

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            int: The complexity as an integer.
        """
        logger.debug("Calculating complexity.")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            complexity = self.calculate_cyclomatic_complexity(node)
        elif isinstance(node, ast.ClassDef):
            complexity = sum(self.calculate_complexity(m) for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)))
        elif isinstance(node, ast.Module):
            complexity = sum(self.calculate_complexity(n) for n in ast.iter_child_nodes(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
        else:
            complexity = 0
        logger.debug(f"Complexity: {complexity}")
        return complexity

    def calculate_maintainability_index(self, node: ast.AST) -> float:
        """
        Calculates maintainability index.

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            float: The maintainability index as a float.
        """
        logger.debug("Calculating maintainability index.")
        halstead = self.calculate_halstead_metrics(node)
        complexity = self.calculate_complexity(node)
        sloc = self._count_source_lines(node)

        volume = halstead['program_volume']

        if volume == 0 or sloc == 0:  # Handle potential errors
            mi = 100.0  # Maximum value if no volume or lines of code
        else:
            mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(sloc)
            mi = max(0, mi)  # Ensure MI is not negative
            mi = min(100, mi * 100 / 171)  # Normalize to 0-100

        logger.debug(f"Maintainability index: {mi}")
        return round(mi, 2)

    def _count_source_lines(self, node: ast.AST) -> int:
        """
        Counts source lines of code (excluding comments and blank lines).

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            int: The number of source lines as an integer.
        """
        logger.debug("Counting source lines of code.")
        if hasattr(ast, 'unparse'):
            source = ast.unparse(node)
        else:
            source = self._get_source_code(node)
        lines = [line.strip() for line in source.splitlines()]
        sloc = len([line for line in lines if line and not line.startswith('#')])
        logger.debug(f"Source lines of code: {sloc}")
        return sloc

    def _get_source_code(self, node: ast.AST) -> str:
        """
        Extracts source code for Python < 3.9.

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            str: The source code as a string.
        """
        return ast.dump(node)

    def _get_operand_name(self, node: ast.AST) -> str:
        """
        Gets the name of an operand node.

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            str: The operand name as a string.
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return ''

    def analyze_dependencies(self, node: ast.AST, module_name: str = None) -> Dict[str, Set[str]]:
        """
        Analyzes module dependencies, including circular dependency detection.

        Args:
            node (ast.AST): The AST node to analyze.
            module_name (str, optional): The name of the module being analyzed.

        Returns:
            Dict[str, Set[str]]: A dictionary of module dependencies.

        Raises:
            MetricsError: If an error occurs during import processing.
        """
        logger.debug("Analyzing module dependencies.")
        deps: Dict[str, Set[str]] = defaultdict(set)
        self.module_name = module_name  # Store for circular dependency check

        for subnode in ast.walk(node):
            if isinstance(subnode, (ast.Import, ast.ImportFrom)):
                self._process_import(subnode, deps)

        circular_deps = self._detect_circular_dependencies(deps)
        if circular_deps:
            logger.warning(f"Circular dependencies detected: {circular_deps}")

        logger.debug(f"Module dependencies: {deps}")
        return dict(deps)

    def _detect_circular_dependencies(self, dependencies: Dict[str, Set[str]]) -> List[Tuple[str, str]]:
        """
        Detects circular dependencies.

        Args:
            dependencies (Dict[str, Set[str]]): A dictionary of module dependencies.

        Returns:
            List[Tuple[str, str]]: A list of tuples representing circular dependencies.
        """
        logger.debug("Detecting circular dependencies.")
        circular_dependencies = []
        for module, deps in dependencies.items():
            for dep in deps:
                if self.module_name and dep == self.module_name:  # Check against current module
                    circular_dependencies.append((module, dep))
                elif dep in dependencies and module in dependencies[dep]:
                    circular_dependencies.append((module, dep))
        logger.debug(f"Circular dependencies: {circular_dependencies}")
        return circular_dependencies

    def generate_dependency_graph(self, dependencies: Dict[str, Set[str]], output_file: str) -> None:
        """
        Generates a visual dependency graph.

        Args:
            dependencies (Dict[str, Set[str]]): A dictionary of module dependencies.
            output_file (str): The file path to save the graph.
        """
        logger.debug("Generating dependency graph.")
        try:
            from graphviz import Digraph
        except ImportError:
            logger.warning("Graphviz not installed. Skipping dependency graph generation.")
            return

        try:
            dot = Digraph(comment='Module Dependencies')
            
            # Add nodes and edges
            for module, deps in dependencies.items():
                dot.node(module, module)
                for dep in deps:
                    dot.edge(module, dep)

            # Render graph
            dot.render(output_file, view=False, cleanup=True)
            logger.info(f"Dependency graph saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating dependency graph: {e}")
            logger.error("Make sure Graphviz is installed and in your system PATH")

    def _process_import(self, node: ast.AST, deps: Dict[str, Set[str]]) -> None:
        """
        Processes import statements and categorizes dependencies.

        Args:
            node (ast.AST): The AST node representing an import statement.
            deps (Dict[str, Set[str]]): A dictionary to store dependencies.

        Raises:
            MetricsError: If an error occurs during import processing.
        """
        try:
            if isinstance(node, ast.Import):
                for name in node.names:
                    self._categorize_import(name.name, deps)
            elif isinstance(node, ast.ImportFrom) and node.module:
                self._categorize_import(node.module, deps)
        except Exception as e:
            logger.error(f"Error processing import: {e}")
            raise MetricsError(f"Error processing import: {e}")

    def _categorize_import(self, module_name: str, deps: Dict[str, Set[str]]) -> None:
        """
        Categorizes an import as stdlib, third-party, or local.

        Args:
            module_name (str): The name of the module being imported.
            deps (Dict[str, Set[str]]): A dictionary to store categorized dependencies.

        Raises:
            MetricsError: If an error occurs during categorization.
        """
        try:
            if importlib.util.find_spec(module_name) is not None:
                deps['stdlib'].add(module_name)
            elif '.' in module_name:
                deps['local'].add(module_name)
            else:
                deps['third_party'].add(module_name)
        except Exception as e:
            logger.error(f"Error categorizing import {module_name}: {e}")
            raise MetricsError(f"Error categorizing import {module_name}: {e}")