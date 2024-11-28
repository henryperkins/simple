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
from typing import Dict, Set  
from core.logger import LoggerSetup, log_debug, log_info, log_error  
  
logger = LoggerSetup.get_logger(__name__)  
  
  
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
  
    @staticmethod  
    def calculate_cyclomatic_complexity(function_node: ast.FunctionDef) -> int:  
        """  
        Calculate the cyclomatic complexity of a function.  
  
        Parameters:  
            function_node (ast.FunctionDef): The AST node representing the function.  
  
        Returns:  
            int: The cyclomatic complexity of the function.  
        """  
        log_debug(f"Calculating cyclomatic complexity for function: {getattr(function_node, 'name', 'unknown')}")  
        if not isinstance(function_node, ast.FunctionDef):  
            log_error(f"Provided node is not a function definition: {ast.dump(function_node)}")  
            return 0  
  
        complexity = 1  # Start with 1 for the function itself  
  
        decision_points = (  
            ast.If, ast.For, ast.While, ast.And, ast.Or, ast.ExceptHandler,  
            ast.With, ast.Try, ast.BoolOp, ast.Lambda, ast.ListComp, ast.DictComp,  
            ast.SetComp, ast.GeneratorExp, ast.IfExp, ast.Match  # For Python 3.10+  
        )  
  
        for node in ast.walk(function_node):  
            if isinstance(node, decision_points):  
                if isinstance(node, ast.BoolOp):  
                    complexity += len(node.values) - 1  
                    log_debug(f"Incremented complexity for BoolOp with {len(node.values) - 1} decision points: {ast.dump(node)}")  
                elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):  
                    complexity += 1  
                    log_debug(f"Incremented complexity for comprehension: {ast.dump(node)}")  
                elif isinstance(node, ast.Match):  
                    complexity += len(node.cases)  
                    log_debug(f"Incremented complexity for Match with {len(node.cases)} cases: {ast.dump(node)}")  
                elif isinstance(node, ast.IfExp):  
                    complexity += 1  
                    log_debug(f"Incremented complexity for IfExp: {ast.dump(node)}")  
                else:  
                    complexity += 1  
                    log_debug(f"Incremented complexity at node: {ast.dump(node)}")  
  
        log_info(f"Calculated cyclomatic complexity for function '{function_node.name}' is {complexity}")  
        return complexity  
  
    @staticmethod  
    def calculate_cognitive_complexity(function_node: ast.FunctionDef) -> int:  
        """  
        Calculate the cognitive complexity of a function.  
  
        Parameters:  
            function_node (ast.FunctionDef): The AST node representing the function.  
  
        Returns:  
            int: The cognitive complexity of the function.  
        """  
        log_debug(f"Calculating cognitive complexity for function: {getattr(function_node, 'name', 'unknown')}")  
        if not isinstance(function_node, ast.FunctionDef):  
            log_error(f"Provided node is not a function definition: {ast.dump(function_node)}")  
            return 0  
  
        cognitive_complexity = 0  
        nesting_depth = 0  
  
        def traverse(node, nesting_level):  
            nonlocal cognitive_complexity  
            for child in ast.iter_child_nodes(node):  
                if Metrics._is_nesting_construct(child):  
                    nesting_level += 1  
                    cognitive_complexity += 1  # Structural increment  
                    log_debug(f"Nesting level {nesting_level} increased at node: {ast.dump(child)}")  
                    traverse(child, nesting_level)  
                    nesting_level -= 1  
                elif Metrics._is_complexity_increment(child):  
                    cognitive_complexity += nesting_level + 1  # Complexity increment with nesting consideration  
                    log_debug(f"Incremented cognitive complexity by {nesting_level + 1} at node: {ast.dump(child)}")  
                    traverse(child, nesting_level)  
                else:  
                    traverse(child, nesting_level)  
  
        traverse(function_node, 0)  
        log_info(f"Calculated cognitive complexity for function '{function_node.name}' is {cognitive_complexity}")  
        return cognitive_complexity  
  
    def calculate_complexity(self, node: ast.AST) -> int:  
        """  
        Calculate the overall complexity of the given AST node.  
  
        Args:  
            node (ast.AST): The AST node to analyze.  
  
        Returns:  
            int: The overall complexity score.  
        """  
        log_debug("Calculating overall complexity.")  
        if not isinstance(node, ast.FunctionDef):  
            log_error(f"Provided node is not a function definition: {ast.dump(node)}")  
            return 0  
        cyclomatic_complexity = self.calculate_cyclomatic_complexity(node)  
        cognitive_complexity = self.calculate_cognitive_complexity(node)  
        overall_complexity = cyclomatic_complexity + cognitive_complexity  
        log_info(f"Calculated overall complexity for function '{node.name}' is {overall_complexity}")  
        return overall_complexity  
  
    def calculate_maintainability_index(self, node: ast.AST) -> float:  
        """  
        Calculate maintainability index based on various metrics.  
  
        Args:  
            node (ast.AST): AST node to analyze  
  
        Returns:  
            float: Maintainability index score (0-100)  
        """  
        log_debug("Calculating maintainability index.")  
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
  
            log_info(f"Calculated maintainability index is {mi}")  
            return round(mi, 2)  
  
        except Exception as e:  
            log_error(f"Error calculating maintainability index: {e}")  
            return 0.0  
  
    def calculate_halstead_metrics(self, node: ast.AST) -> Dict[str, float]:  
        """  
        Calculate Halstead metrics for the given AST node.  
  
        Args:  
            node (ast.AST): The AST node to analyze.  
  
        Returns:  
            Dict[str, float]: A dictionary containing Halstead metrics.  
        """  
        log_debug("Calculating Halstead metrics.")  
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
  
        log_info(f"Calculated Halstead metrics: Length={program_length}, Vocabulary={program_vocabulary}, Volume={program_volume}")  
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
        log_debug("Counting source lines of code.")  
        try:  
            source = ast.unparse(node)  
            lines = [line.strip() for line in source.splitlines()]  
            count = len([line for line in lines if line and not line.startswith('#')])  
            log_info(f"Counted {count} source lines of code.")  
            return count  
        except Exception as e:  
            log_error(f"Error counting source lines: {e}")  
            return 0  
  
    @staticmethod  
    def _is_nesting_construct(node: ast.AST) -> bool:  
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
        log_debug(f"Node {ast.dump(node)} is {'a' if nesting_construct else 'not a'} nesting construct.")  
        return nesting_construct  
  
    @staticmethod  
    def _is_complexity_increment(node: ast.AST) -> bool:  
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
        log_debug(f"Node {ast.dump(node)} {'increments' if increment else 'does not increment'} complexity.")  
        return increment  
  
    def analyze_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:  
        """  
        Analyze module dependencies and imports.  
  
        Args:  
            node (ast.AST): AST node to analyze  
  
        Returns:  
            Dict[str, Set[str]]: Dictionary of module dependencies  
        """  
        log_debug("Analyzing module dependencies.")  
        deps = {  
            'stdlib': set(),  
            'third_party': set(),  
            'local': set()  
        }  
  
        try:  
            for subnode in ast.walk(node):  
                if isinstance(subnode, (ast.Import, ast.ImportFrom)):  
                    self._process_import(subnode, deps)  
            log_info(f"Analyzed dependencies: {deps}")  
            return deps  
        except Exception as e:  
            log_error(f"Error analyzing dependencies: {e}")  
            return deps  
  
    def _process_import(self, node: ast.AST, deps: Dict[str, Set[str]]) -> None:  
        """Process import statement and categorize dependency."""  
        log_debug(f"Processing import: {ast.dump(node)}")  
        try:  
            if isinstance(node, ast.Import):  
                for name in node.names:  
                    self._categorize_import(name.name, deps)  
            elif isinstance(node, ast.ImportFrom) and node.module:  
                self._categorize_import(node.module, deps)  
        except Exception as e:  
            log_error(f"Error processing import: {e}")  
  
    def _categorize_import(self, module_name: str, deps: Dict[str, Set[str]]) -> None:  
        """Categorize import as stdlib, third-party, or local."""  
        log_debug(f"Categorizing import: {module_name}")  
        try:  
            if module_name in sys.builtin_module_names:  
                deps['stdlib'].add(module_name)  
            elif '.' in module_name:  
                deps['local'].add(module_name)  
            else:  
                deps['third_party'].add(module_name)  
        except Exception as e:  
            log_error(f"Error categorizing import {module_name}: {e}")  
  
  
def test_metrics():  
    """  
    Test function for the Metrics class.  
  
    This function tests the calculation of cyclomatic and cognitive complexity  
    for a sample function defined in source_code.  
    """  
    log_info("Starting test_metrics.")  
    source_code = """  
def example_function(x):  
    if x > 0:  
        for i in range(x):  
            if i % 2 == 0:  
                print(i)  
            else:  
                continue  
    else:  
        return -1  
    return 0  
"""  
    tree = ast.parse(source_code)  
    function_node = tree.body[0]  
  
    if isinstance(function_node, ast.FunctionDef):  
        # Test cyclomatic complexity  
        cyclomatic_complexity = Metrics.calculate_cyclomatic_complexity(function_node)  
        assert cyclomatic_complexity == 4, f"Expected 4, got {cyclomatic_complexity}"  
        log_info(f"Cyclomatic complexity test passed: {cyclomatic_complexity}")  
  
        # Test cognitive complexity  
        cognitive_complexity = Metrics.calculate_cognitive_complexity(function_node)  
        assert cognitive_complexity == 9, f"Expected 9, got {cognitive_complexity}"  
        log_info(f"Cognitive complexity test passed: {cognitive_complexity}")  
  
        log_info("All tests passed.")  
    else:  
        log_error("The node is not a function definition.")  
  
  
# Ensure tests run only when the script is executed directly  
if __name__ == "__main__":  
    test_metrics()  