# extract/code.py

import ast
from datetime import datetime
from typing import Dict, Any
from core.logger import LoggerSetup
from extract.base import BaseExtractor
from extract.classes import ClassExtractor
from extract.functions import FunctionExtractor
from ..utils import validate_schema
from ..metrics import CodeMetrics

logger = LoggerSetup.get_logger("extract.code")

def extract_classes_and_functions_from_ast(tree: ast.AST, content: str) -> Dict[str, Any]:
    """
    Extract all classes and functions from an AST.

    Args:
        tree (ast.AST): The AST to analyze
        content (str): The source code content

    Returns:
        Dict[str, Any]: Extracted information including classes, functions, and metrics
    """
    try:
        metrics = CodeMetrics()
        result = {
            "summary": "",
            "changelog": [],
            "classes": [],
            "functions": [],
            "file_content": [{"content": content}]
        }

        result["changelog"].append({
            "change": "Started code analysis",
            "timestamp": datetime.now().isoformat()
        })

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                try:
                    extractor = ClassExtractor(node, content)
                    class_info = extractor.extract_details()
                    result["classes"].append(class_info)
                    metrics.total_classes += 1

                    result["changelog"].append({
                        "change": f"Analyzed class: {node.name}",
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.error(f"Error extracting class {getattr(node, 'name', 'unknown')}: {e}")
                    result["changelog"].append({
                        "change": f"Error analyzing class {getattr(node, 'name', 'unknown')}: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    })

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not isinstance(getattr(node, 'parent', None), ast.ClassDef):
                    try:
                        extractor = FunctionExtractor(node, content)
                        func_info = extractor.extract_details()
                        result["functions"].append(func_info)
                        metrics.total_functions += 1

                        result["changelog"].append({
                            "change": f"Analyzed function: {node.name}",
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        logger.error(f"Error extracting function {getattr(node, 'name', 'unknown')}: {e}")
                        result["changelog"].append({
                            "change": f"Error analyzing function {getattr(node, 'name', 'unknown')}: {str(e)}",
                            "timestamp": datetime.now().isoformat()
                        })

        metrics.total_lines = len(content.splitlines())

        summary_parts = [
            f"Found {len(result['classes'])} classes and {len(result['functions'])} functions",
            f"Total lines of code: {metrics.total_lines}",
        ]

        if result["functions"]:
            avg_complexity = sum(f.get("complexity_score", 0) for f in result["functions"]) / len(result["functions"])
            max_complexity = max((f.get("complexity_score", 0) for f in result["functions"]), default=0)
            summary_parts.extend([
                f"Average function complexity: {avg_complexity:.2f}",
                f"Maximum function complexity: {max_complexity}"
            ])

        functions_with_docs = sum(1 for f in result["functions"] if f.get("docstring"))
        classes_with_docs = sum(1 for c in result["classes"] if c.get("docstring"))
        total_items = len(result["functions"]) + len(result["classes"])
        if total_items > 0:
            doc_coverage = ((functions_with_docs + classes_with_docs) / total_items) * 100
            summary_parts.append(f"Documentation coverage: {doc_coverage:.1f}%")

        result["summary"] = " | ".join(summary_parts)

        result["changelog"].append({
            "change": "Completed code analysis",
            "timestamp": datetime.now().isoformat()
        })

        validate_schema(result)
        logger.info("Successfully extracted and validated code information")
        return result

    except Exception as e:
        logger.error(f"Error extracting classes and functions: {e}")
        error_result = {
            "summary": "Error during extraction",
            "changelog": [{
                "change": f"Error during extraction: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }],
            "classes": [],
            "functions": [],
            "file_content": [{"content": content}]
        }
        return error_result
