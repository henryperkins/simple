import ast
from collections import defaultdict
from typing import Any, Dict, List
from core.logger import LoggerSetup

# Initialize a logger specifically for this module
logger = LoggerSetup.get_logger("metrics")

class CodeMetrics:
    def __init__(self):
        self.total_functions = 0
        self.total_classes = 0
        self.total_lines = 0
        self.docstring_coverage = 0.0
        self.type_hint_coverage = 0.0
        self.avg_complexity = 0.0
        self.max_complexity = 0
        self.cognitive_complexity = 0
        self.halstead_metrics = defaultdict(float)
        self.type_hints_stats = defaultdict(int)
        self.quality_issues = []
        logger.debug("Initialized CodeMetrics instance.")

    def calculate_complexity(self, node: ast.AST) -> int:
        """
        Calculate the cyclomatic complexity score for a function or method.
        Args:
            node (ast.AST): The AST node representing a function or method.
        Returns:
            int: The cyclomatic complexity score.
        """
        name = getattr(node, 'name', 'unknown')
        complexity = 1  # Start with one for the function entry point
        try:
            for subnode in ast.walk(node):
                if isinstance(subnode, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp)):
                    complexity += 1
            logger.debug(f"Calculated complexity for node {name}: {complexity}")
            self.max_complexity = max(self.max_complexity, complexity)
        except Exception as e:
            logger.error(f"Error calculating complexity for node {name}: {e}")
        return complexity

    def calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """
        Calculate the cognitive complexity score for a function or method.
        This metric measures how difficult the code is to understand, considering:
        - Nesting depth (loops, conditionals)
        - Logical operators
        - Recursion
        - Multiple exit points
        
        Args:
            node (ast.AST): The AST node representing a function or method.
        Returns:
            int: The cognitive complexity score.
        """
        name = getattr(node, 'name', 'unknown')
        try:
            cognitive_score = 0
            nesting_level = 0

            class CognitiveComplexityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.score = 0
                    self.depth = 0

                def visit_If(self, node):
                    self.score += (1 + self.depth)  # Base cost + nesting
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1

                def visit_For(self, node):
                    self.score += (1 + self.depth)
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1

                def visit_While(self, node):
                    self.score += (1 + self.depth)
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1

                def visit_BoolOp(self, node):
                    # Add complexity for boolean operations (and, or)
                    self.score += len(node.values) - 1
                    self.generic_visit(node)

                def visit_Try(self, node):
                    self.score += 1  # Base cost for try block
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1
                    # Add cost for each except handler
                    self.score += len(node.handlers)

                def visit_Return(self, node):
                    # Add complexity for multiple return statements
                    if self.depth > 0:
                        self.score += 1
                    self.generic_visit(node)

            visitor = CognitiveComplexityVisitor()
            visitor.visit(node)
            cognitive_score = visitor.score

            logger.debug(f"Calculated cognitive complexity for node {name}: {cognitive_score}")
            return cognitive_score

        except Exception as e:
            logger.error(f"Error calculating cognitive complexity for node {name}: {e}")
            return 0

    def calculate_halstead_metrics(self, node: ast.AST) -> Dict[str, float]:
        """
        Calculate Halstead metrics for a function or method.
        Halstead metrics include:
        - Program Length (N): Total number of operators and operands
        - Program Vocabulary (n): Number of unique operators and operands
        - Program Volume (V): N * log2(n)
        - Difficulty (D): Related to error proneness
        - Effort (E): Mental effort required to implement
        
        Args:
            node (ast.AST): The AST node representing a function or method.
        Returns:
            Dict[str, float]: The Halstead metrics.
        """
        name = getattr(node, 'name', 'unknown')
        try:
            class HalsteadVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.operators = set()
                    self.operands = set()
                    self.total_operators = 0
                    self.total_operands = 0

                def visit_BinOp(self, node):
                    self.operators.add(type(node.op).__name__)
                    self.total_operators += 1
                    self.generic_visit(node)

                def visit_UnaryOp(self, node):
                    self.operators.add(type(node.op).__name__)
                    self.total_operators += 1
                    self.generic_visit(node)

                def visit_BoolOp(self, node):
                    self.operators.add(type(node.op).__name__)
                    self.total_operators += 1
                    self.generic_visit(node)

                def visit_Compare(self, node):
                    for op in node.ops:
                        self.operators.add(type(op).__name__)
                        self.total_operators += 1
                    self.generic_visit(node)

                def visit_Name(self, node):
                    self.operands.add(node.id)
                    self.total_operands += 1
                    self.generic_visit(node)

                def visit_Constant(self, node):
                    self.operands.add(str(node.value))
                    self.total_operands += 1
                    self.generic_visit(node)

            visitor = HalsteadVisitor()
            visitor.visit(node)

            # Calculate basic Halstead metrics
            n1 = len(visitor.operators)  # Number of unique operators
            n2 = len(visitor.operands)   # Number of unique operands
            N1 = visitor.total_operators # Total operators
            N2 = visitor.total_operands  # Total operands

            # Prevent division by zero and log(0)
            if n1 + n2 == 0:
                return {
                    "program_length": 0,
                    "vocabulary_size": 0,
                    "program_volume": 0,
                    "difficulty": 0,
                    "effort": 0
                }

            import math
            program_length = N1 + N2
            vocabulary_size = n1 + n2
            volume = program_length * math.log2(vocabulary_size) if vocabulary_size > 0 else 0
            difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
            effort = difficulty * volume

            metrics = {
                "program_length": program_length,
                "vocabulary_size": vocabulary_size,
                "program_volume": volume,
                "difficulty": difficulty,
                "effort": effort
            }

            logger.debug(f"Calculated Halstead metrics for node {name}: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating Halstead metrics for node {name}: {e}")
            return {
                "program_length": 0,
                "vocabulary_size": 0,
                "program_volume": 0,
                "difficulty": 0,
                "effort": 0
            }

    def analyze_function_quality(self, function_info: Dict[str, Any]) -> None:
        """
        Analyze function quality and add recommendations.
        Args:
            function_info (Dict[str, Any]): The function details.
        """
        name = function_info.get('name', 'unknown')
        score = function_info.get('complexity_score', 0)
        logger.debug(f"Analyzing quality for function: {name}")
        if score > 10:
            msg = (
                f"Function '{name}' has high complexity ({score}). "
                "Consider breaking it down."
            )
            self.quality_issues.append(msg)
            logger.info(msg)
        if not function_info.get("docstring"):
            msg = f"Function '{name}' lacks a docstring."
            self.quality_issues.append(msg)
            logger.info(msg)
        params_without_types = [
            p["name"] for p in function_info.get("params", [])
            if not p.get("has_type_hint")
        ]
        if params_without_types:
            params_str = ", ".join(params_without_types)
            msg = (
                f"Function '{name}' has parameters without type hints: "
                f"{params_str}"
            )
            self.quality_issues.append(msg)
            logger.info(msg)

    def analyze_class_quality(self, class_info: Dict[str, Any]) -> None:
        """
        Analyze class quality and add recommendations.
        Args:
            class_info (Dict[str, Any]): The class details.
        """
        name = class_info.get('name', 'unknown')
        logger.debug(f"Analyzing quality for class: {name}")
        if not class_info.get("docstring"):
            msg = f"Class '{name}' lacks a docstring."
            self.quality_issues.append(msg)
            logger.info(msg)
        method_count = len(class_info.get("methods", []))
        if method_count > 10:
            msg = (
                f"Class '{name}' has many methods ({method_count}). "
                "Consider splitting it."
            )
            self.quality_issues.append(msg)
            logger.info(msg)

    def update_type_hint_stats(self, function_info: Dict[str, Any]) -> None:
        """
        Update type hint statistics based on function information.
        Args:
            function_info (Dict[str, Any]): The function details.
        """
        total_hints_possible = len(function_info.get("params", [])) + 1  # Including return type
        hints_present = sum(
            1 for p in function_info.get("params", []) if p.get("has_type_hint")
        )
        if function_info.get("return_type", {}).get("has_type_hint", False):
            hints_present += 1
        self.type_hints_stats["total_possible"] += total_hints_possible
        self.type_hints_stats["total_present"] += hints_present
        logger.debug(f"Updated type hint stats: {self.type_hints_stats}")

    def calculate_final_metrics(self, all_items: List[Dict[str, Any]]) -> None:
        """
        Calculate final metrics after processing all items.
        Args:
            all_items (List[Dict[str, Any]]): List of all functions and methods analyzed.
        """
        total_items = len(all_items)
        logger.debug(f"Calculating final metrics for {total_items} items.")
        if total_items > 0:
            items_with_doc = sum(1 for item in all_items if item.get("docstring"))
            self.docstring_coverage = (items_with_doc / total_items) * 100
            total_complexity = sum(
                item.get("complexity_score", 0)
                for item in all_items
            )
            self.avg_complexity = total_complexity / total_items if total_items else 0
        if self.type_hints_stats["total_possible"] > 0:
            self.type_hint_coverage = (
                self.type_hints_stats["total_present"] /
                self.type_hints_stats["total_possible"]
            ) * 100
        logger.info(
            f"Final metrics calculated: Docstring coverage: {self.docstring_coverage:.2f}%, "
            f"Type hint coverage: {self.type_hint_coverage:.2f}%, "
            f"Average complexity: {self.avg_complexity:.2f}, "
            f"Max complexity: {self.max_complexity}"
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of code metrics.
        Returns:
            Dict[str, Any]: The summary of code metrics.
        """
        summary = {
            "total_classes": self.total_classes,
            "total_functions": self.total_functions,
            "total_lines": self.total_lines,
            "docstring_coverage_percentage": round(self.docstring_coverage, 2),
            "type_hint_coverage_percentage": round(self.type_hint_coverage, 2),
            "average_complexity": round(self.avg_complexity, 2),
            "max_complexity": self.max_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "halstead_metrics": dict(self.halstead_metrics),
            "quality_recommendations": self.quality_issues
        }
        logger.debug(f"Generated summary: {summary}")
        return summary