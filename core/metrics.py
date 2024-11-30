"""  
Code Metrics Analysis Module  
  
This module provides comprehensive code quality and complexity metrics for Python source code,  
including cyclomatic complexity, cognitive complexity, Halstead metrics, and code quality analysis.  
  
Version: 1.1.0  
Author: Development Team  
"""  
  
import ast  
import math  
import sys  
from collections import defaultdict  
from typing import Dict, Set, Any, Union
from core.logger import LoggerSetup, log_debug, log_info, log_error  
  
class MetricsError(Exception):  
    """Base exception for metrics calculation errors."""  
    pass  
  
class Metrics:  
    """  
    Provides methods to calculate different complexity metrics for Python functions.  
  
    This class includes methods for calculating cyclomatic complexity, cognitive complexity,  
    Halstead metrics, and maintainability index. It also provides functionality to analyze  
    module dependencies.  
    """  
  
    MAINTAINABILITY_THRESHOLDS = {  
        'good': 80,  
        'moderate': 60,  
        'poor': 40  
    }  
  
    def __init__(self):  
        """Initialize the Metrics class with a logger."""  
        self.logger = LoggerSetup.get_logger(__name__)  
  
    def calculate_cyclomatic_complexity(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
        """Calculate cyclomatic complexity for a function."""
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.logger.error("Provided node is not a function definition: %s", ast.dump(node))
            return 0

        complexity = 1  # Start with 1 for the function itself

        decision_points = (
            ast.If, ast.For, ast.AsyncFor, ast.While, ast.And, ast.Or,
            ast.ExceptHandler, ast.With, ast.AsyncWith, ast.Try,
            ast.BoolOp, ast.Lambda
        )

        for child in ast.walk(node):
            if isinstance(child, decision_points):
                complexity += 1

        return complexity
  
    def calculate_cognitive_complexity(self, function_node: ast.FunctionDef) -> int:  
        """  
        Calculate the cognitive complexity of a function.  
  
        Parameters:  
            function_node (ast.FunctionDef): The AST node representing the function.  
  
        Returns:  
            int: The cognitive complexity of the function.  
        """  
        self.logger.debug(f"Calculating cognitive complexity for function: {getattr(function_node, 'name', 'unknown')}")  
        if not isinstance(function_node, ast.FunctionDef):  
            self.logger.error(f"Provided node is not a function definition: {ast.dump(function_node)}")  
            return 0  
  
        cognitive_complexity = 0  
        nesting_depth = 0  
  
        def traverse(node, nesting_level):  
            nonlocal cognitive_complexity  
            for child in ast.iter_child_nodes(node):  
                if self._is_nesting_construct(child):  
                    nesting_level += 1  
                    cognitive_complexity += 1  # Structural increment  
                    self.logger.debug(f"Nesting level {nesting_level} increased at node: {ast.dump(child)}")  
                    traverse(child, nesting_level)  
                    nesting_level -= 1  
                elif self._is_complexity_increment(child):  
                    cognitive_complexity += nesting_level + 1  # Complexity increment with nesting consideration  
                    self.logger.debug(f"Incremented cognitive complexity by {nesting_level + 1} at node: {ast.dump(child)}")  
                    traverse(child, nesting_level)  
                else:  
                    traverse(child, nesting_level)  
  
        traverse(function_node, 0)  
        self.logger.info(f"Calculated cognitive complexity for function '{function_node.name}' is {cognitive_complexity}")  
        return cognitive_complexity  
  
    def calculate_complexity(self, node: ast.AST) -> int:
        """Calculate complexity for any AST node."""
        try:
            # Handle async and regular functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return self.calculate_cyclomatic_complexity(node)
            elif isinstance(node, ast.ClassDef):
                # Calculate class complexity as sum of methods complexity
                return sum(
                    self.calculate_cyclomatic_complexity(method)
                    for method in node.body 
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
            elif isinstance(node, ast.Module):
                # Calculate module complexity
                return sum(
                    self.calculate_complexity(child)
                    for child in ast.iter_child_nodes(node)
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                )
            return 0
        except Exception as e:
            self.logger.error("Error calculating complexity: %s", str(e))
            return 0
  
    def calculate_maintainability_index(self, node: ast.AST) -> float:  
        """  
        Calculate maintainability index based on various metrics.  
  
        Args:  
            node (ast.AST): AST node to analyze  
  
        Returns:  
            float: Maintainability index score (0-100)  
        """  
        self.logger.debug("Calculating maintainability index.")  
        try:  
            halstead = self.calculate_halstead_metrics(node)  
            complexity = self.calculate_complexity(node)  
            sloc = self._count_source_lines(node)  
  
            # Calculate Maintainability Index  
            volume = halstead['program_volume']  
            if volume == 0 or sloc == 0:  
                mi = 0.0  
            else:  
                mi = max(0, (171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(sloc)) * 100 / 171)  
                mi = min(100, mi)  # Normalize to 0-100  
  
            self.logger.info(f"Calculated maintainability index is {mi}")  
            return round(mi, 2)  
  
        except Exception as e:  
            self.logger.error(f"Error calculating maintainability index: {e}")  
            return 0.0  
  
    def calculate_halstead_metrics(self, node: ast.AST) -> Dict[str, float]:  
        """  
        Calculate Halstead metrics for the given AST node.  
  
        Args:  
            node (ast.AST): The AST node to analyze.  
  
        Returns:  
            Dict[str, float]: A dictionary containing Halstead metrics.  
        """  
        self.logger.debug("Calculating Halstead metrics.")  
        operators = set()  
        operands = set()  
        operator_count = 0  
        operand_count = 0  
  
        operator_types = (  
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,  
            ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,  
            ast.FloorDiv, ast.And, ast.Or, ast.Not, ast.Invert,  
            ast.UAdd, ast.USub, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,  
            ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,  
            ast.Call, ast.Attribute, ast.Subscript, ast.Index, ast.Slice,  
            ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Yield, ast.YieldFrom  
        )  
  
        operand_types = (  
            ast.Constant, ast.Name, ast.List, ast.Tuple, ast.Set, ast.Dict  
        )  
  
        for n in ast.walk(node):  
            if isinstance(n, operator_types):  
                operators.add(type(n).__name__)  
                operator_count += 1  
            elif isinstance(n, operand_types):  
                operands.add(self._get_operand_name(n))  
                operand_count += 1  
  
        n1 = len(operators)  
        n2 = len(operands)  
        N1 = operator_count  
        N2 = operand_count  
  
        program_length = N1 + N2  
        program_vocabulary = n1 + n2  
        program_volume = program_length * math.log2(program_vocabulary) if program_vocabulary > 0 else 0  
  
        self.logger.info(f"Calculated Halstead metrics: Length={program_length}, Vocabulary={program_vocabulary}, Volume={program_volume}")  
        return {  
            'program_length': program_length,  
            'program_vocabulary': program_vocabulary,  
            'program_volume': program_volume  
        }  
  
    def _get_operand_name(self, node: ast.AST) -> str:  
        """Get a string representation of an operand."""  
        if isinstance(node, ast.Constant):  
            return str(node.value)  
        elif isinstance(node, ast.Name):  
            return node.id  
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):  
            return type(node).__name__  
        else:  
            return ''  
  
    def _count_source_lines(self, node: ast.AST) -> int:  
        """  
        Count source lines of code (excluding comments and blank lines).  
  
        Args:  
            node (ast.AST): AST node to analyze  
  
        Returns:  
            int: Number of source code lines  
        """  
        self.logger.debug("Counting source lines of code.")  
        try:  
            source = ast.unparse(node)  
            lines = [line.strip() for line in source.splitlines()]  
            count = len([line for line in lines if line and not line.startswith('#')])  
            self.logger.info(f"Counted {count} source lines of code.")  
            return count  
        except Exception as e:  
            self.logger.error(f"Error counting source lines: {e}")  
            return 0  
  
    def _is_nesting_construct(self, node: ast.AST) -> bool:  
        """  
        Determine if a node represents a nesting construct for cognitive complexity.  
  
        Parameters:  
            node (ast.AST): The AST node to check.  
  
        Returns:  
            bool: True if the node is a nesting construct, False otherwise.  
        """  
        nesting_construct = isinstance(node, (  
            ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler, ast.With,  
            ast.Lambda, ast.AsyncFunctionDef, ast.FunctionDef  
        ))  
        self.logger.debug(f"Node {ast.dump(node)} is {'a' if nesting_construct else 'not a'} nesting construct.")  
        return nesting_construct  
  
    def _is_complexity_increment(self, node: ast.AST) -> bool:  
        """  
        Determine if a node should increment cognitive complexity.  
  
        Parameters:  
            node (ast.AST): The current AST node.  
  
        Returns:  
            bool: True if the node should increment complexity, False otherwise.  
        """  
        increment = isinstance(node, (  
            ast.BoolOp, ast.Compare, ast.Break, ast.Continue, ast.Raise, ast.Return, ast.Yield, ast.YieldFrom  
        ))  
        self.logger.debug(f"Node {ast.dump(node)} {'increments' if increment else 'does not increment'} complexity.")  
        return increment  
  
    def analyze_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:  
        """  
        Analyze module dependencies and imports.  
  
        Args:  
            node (ast.AST): AST node to analyze  
  
        Returns:  
            Dict[str, Set[str]]: Dictionary of module dependencies  
        """  
        self.logger.debug("Analyzing module dependencies.")  
        deps = {  
            'stdlib': set(),  
            'third_party': set(),  
            'local': set()  
        }  
  
        try:  
            for subnode in ast.walk(node):  
                if isinstance(subnode, (ast.Import, ast.ImportFrom)):  
                    self._process_import(subnode, deps)  
            self.logger.info(f"Analyzed dependencies: {deps}")  
            return deps  
        except Exception as e:  
            self.logger.error(f"Error analyzing dependencies: {e}")  
            return deps  
  
    def _process_import(self, node: ast.AST, deps: Dict[str, Set[str]]) -> None:  
        """Process import statement and categorize dependency."""  
        self.logger.debug(f"Processing import: {ast.dump(node)}")  
        try:  
            if isinstance(node, ast.Import):  
                for name in node.names:  
                    self._categorize_import(name.name, deps)  
            elif isinstance(node, ast.ImportFrom) and node.module:  
                self._categorize_import(node.module, deps)  
        except Exception as e:  
            self.logger.error(f"Error processing import: {e}")  
  
    def _categorize_import(self, module_name: str, deps: Dict[str, Set[str]]) -> None:  
        """Categorize import as stdlib, third-party, or local."""  
        self.logger.debug(f"Categorizing import: {module_name}")  
        try:  
            if module_name in sys.builtin_module_names:  
                deps['stdlib'].add(module_name)  
            elif '.' in module_name:  
                deps['local'].add(module_name)  
            else:  
                deps['third_party'].add(module_name)  
        except Exception as e:  
            self.logger.error(f"Error categorizing import {module_name}: {e}")  
  
# If needed, testing functions can be included below  