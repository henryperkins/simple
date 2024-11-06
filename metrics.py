import ast
from collections import defaultdict
from typing import Any, Dict, List
from logging_utils import setup_logger

# Initialize a logger specifically for this module
logger = setup_logger("metrics")

class CodeMetrics:
    """Class to store and calculate various code metrics."""

    def __init__(self):
        self.total_functions = 0
        self.total_classes = 0
        self.total_lines = 0
        self.docstring_coverage = 0.0
        self.type_hint_coverage = 0.0
        self.avg_complexity = 0.0
        self.max_complexity = 0
        self.type_hints_stats = defaultdict(int)
        self.quality_issues = []
        logger.debug("Initialized CodeMetrics instance.")

    def calculate_complexity(self, node: ast.AST) -> int:
        """Calculate the cyclomatic complexity score for a function or method."""
        name = getattr(node, 'name', 'unknown')
        logger.debug(f"Calculating complexity for node: {name}")
        complexity = 1  # Start with one for the function entry point
        try:
            for subnode in ast.walk(node):
                complex_nodes = (
                    ast.If, ast.For, ast.While, ast.Try, ast.With,
                    ast.ListComp, ast.DictComp, ast.SetComp,
                    ast.GeneratorExp, ast.BoolOp, ast.And, ast.Or
                )
                if isinstance(subnode, complex_nodes):
                    complexity += 1
            logger.debug(f"Node {name} complexity: {complexity}")
            self.max_complexity = max(self.max_complexity, complexity)
        except Exception as e:
            logger.warning(f"Error calculating complexity for node {name}: {e}")
        return complexity

    def analyze_function_quality(self, function_info: Dict[str, Any]) -> None:
        """Analyze function quality and add recommendations."""
        name = function_info['name']
        score = function_info['complexity_score']
        logger.debug(f"Analyzing quality for function: {name}")

        if score > 10:
            msg = (
                f"Function '{name}' has high complexity ({score}). "
                "Consider breaking it down."
            )
            self.quality_issues.append(msg)
            logger.info(msg)

        if not function_info["docstring"]:
            msg = f"Function '{name}' lacks a docstring."
            self.quality_issues.append(msg)
            logger.info(msg)

        params_without_types = [
            p["name"] for p in function_info["params"]
            if not p["has_type_hint"]
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
        """Analyze class quality and add recommendations."""
        name = class_info['name']
        logger.debug(f"Analyzing quality for class: {name}")

        if not class_info["docstring"]:
            msg = f"Class '{name}' lacks a docstring."
            self.quality_issues.append(msg)
            logger.info(msg)

        method_count = len(class_info["methods"])
        if method_count > 10:
            msg = (
                f"Class '{name}' has many methods ({method_count}). "
                "Consider splitting it."
            )
            self.quality_issues.append(msg)
            logger.info(msg)

    def update_type_hint_stats(self, function_info: Dict[str, Any]) -> None:
        """Update type hint statistics based on function information."""
        total_hints_possible = len(function_info["params"]) + 1
        hints_present = sum(
            1 for p in function_info["params"] if p["has_type_hint"]
        )
        if function_info.get("return_type", {}).get("has_type_hint", False):
            hints_present += 1

        self.type_hints_stats["total_possible"] += total_hints_possible
        self.type_hints_stats["total_present"] += hints_present
        logger.debug(f"Updated type hint stats: {self.type_hints_stats}")

    def calculate_final_metrics(self, all_items: List[Dict[str, Any]]) -> None:
        """Calculate final metrics after processing all items."""
        total_items = len(all_items)
        logger.debug(f"Calculating final metrics for {total_items} items.")
        if total_items > 0:
            items_with_doc = sum(1 for item in all_items if item["docstring"])
            self.docstring_coverage = (items_with_doc / total_items) * 100

            total_complexity = sum(
                item["complexity_score"]
                for item in all_items
                if item["complexity_score"]
            )
            self.avg_complexity = total_complexity / total_items

        if self.type_hints_stats["total_possible"] > 0:
            self.type_hint_coverage = (
                self.type_hints_stats["total_present"] /
                self.type_hints_stats["total_possible"]
            ) * 100

        logger.info(f"Final metrics calculated: Docstring coverage: {self.docstring_coverage:.2f}%, "
                    f"Type hint coverage: {self.type_hint_coverage:.2f}%, "
                    f"Average complexity: {self.avg_complexity:.2f}, "
                    f"Max complexity: {self.max_complexity}")

    def get_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive summary of code metrics."""
        summary = {
            "total_classes": self.total_classes,
            "total_functions": self.total_functions,
            "total_lines": self.total_lines,
            "docstring_coverage_percentage": round(self.docstring_coverage, 2),
            "type_hint_coverage_percentage": round(self.type_hint_coverage, 2),
            "average_complexity": round(self.avg_complexity, 2),
            "max_complexity": self.max_complexity,
            "quality_recommendations": self.quality_issues
        }
        logger.debug(f"Generated summary: {summary}")
        return summary