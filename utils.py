import os
import ast
import json
import hashlib
from typing import Any, Dict, Optional, List, Union
from datetime import datetime
import jsonschema
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger("utils")

_schema_cache: Dict[str, Any] = {}

def generate_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()

def load_json_file(filepath: str) -> Dict[str, Any]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filepath}: {e}")
        raise

def save_json_file(filepath: str, data: Dict[str, Any]) -> None:
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        logger.error(f"Failed to save file {filepath}: {e}")
        raise

def create_timestamp() -> str:
    return datetime.now().isoformat()

def ensure_directory(directory: str) -> None:
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        raise

def validate_file_path(filepath: str, extension: Optional[str] = None) -> bool:
    if not os.path.exists(filepath):
        return False
    if extension and not filepath.endswith(extension):
        return False
    return True

def create_error_result(error_type: str, error_message: str) -> Dict[str, Any]:
    return {
        "summary": f"Error: {error_type}",
        "changelog": [{
            "change": f"{error_type}: {error_message}",
            "timestamp": create_timestamp()
        }],
        "classes": [],
        "functions": [],
        "file_content": [{"content": ""}]
    }

def add_parent_info(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent

def get_file_stats(filepath: str) -> Dict[str, Any]:
    try:
        stats = os.stat(filepath)
        return {
            "size": stats.st_size,
            "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stats.st_atime).isoformat()
        }
    except OSError as e:
        logger.error(f"Failed to get file stats for {filepath}: {e}")
        return {}

def filter_files(directory: str, pattern: str = "*.py", exclude_patterns: Optional[List[str]] = None) -> List[str]:
    import fnmatch
    exclude_patterns = exclude_patterns or []
    matching_files = []
    try:
        for root, _, files in os.walk(directory):
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    filepath = os.path.join(root, filename)
                    if not any(fnmatch.fnmatch(filepath, exp) for exp in exclude_patterns):
                        matching_files.append(filepath)
        return matching_files
    except Exception as e:
        logger.error(f"Error filtering files in {directory}: {e}")
        return []

def normalize_path(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))

def get_relative_path(path: str, base_path: str) -> str:
    return os.path.relpath(path, base_path)

def is_python_file(filepath: str) -> bool:
    if not os.path.isfile(filepath):
        return False
    if not filepath.endswith('.py'):
        return False
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True
    except (SyntaxError, UnicodeDecodeError):
        return False
    except Exception as e:
        logger.error(f"Error checking Python file {filepath}: {e}")
        return False

def convert_changelog(changelog: Union[List, str, None]) -> str:
    if changelog is None:
        return "No changes recorded"
    if isinstance(changelog, str):
        return changelog if changelog.strip() else "No changes recorded"
    if isinstance(changelog, list):
        if not changelog:
            return "No changes recorded"
        entries = []
        for entry in changelog:
            if isinstance(entry, dict):
                timestamp = entry.get("timestamp", datetime.now().isoformat())
                change = entry.get("change", "No description")
                entries.append(f"[{timestamp}] {change}")
            else:
                entries.append(str(entry))
        return " | ".join(entries)
    return "No changes recorded"

def format_function_response(function_data: Dict[str, Any]) -> Dict[str, Any]:
    result = function_data.copy()
    result["changelog"] = convert_changelog(result.get("changelog"))
    result.setdefault("summary", "No summary available")
    result.setdefault("docstring", "")
    result.setdefault("params", [])
    result.setdefault("returns", {"type": "None", "description": ""})
    return result

def validate_function_data(data: Dict[str, Any]) -> None:
    try:
        if "changelog" in data:
            data["changelog"] = convert_changelog(data["changelog"])
        schema = _load_schema()
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise

def get_annotation(node: Optional[ast.AST]) -> str:
    try:
        if node is None:
            return "Any"
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Attribute):
            parts = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        elif isinstance(node, ast.Subscript):
            return f"{get_annotation(node.value)}[{get_annotation(node.slice)}]"
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.BitOr):
                left = get_annotation(node.left)
                right = get_annotation(node.right)
                return f"Union[{left}, {right}]"
        else:
            return "Any"
    except Exception as e:
        logger.error(f"Error processing type annotation: {e}")
        return "Any"

def format_response(sections: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "summary": sections.get("summary", "No summary available"),
        "docstring": sections.get("docstring", "No documentation available"),
        "params": sections.get("params", []),
        "returns": sections.get("returns", {
            "type": "None",
            "description": ""
        }),
        "examples": sections.get("examples", []),
        "changelog": convert_changelog(sections.get("changelog", []))
    }
    return result

def _load_schema() -> Dict[str, Any]:
    if 'schema' not in _schema_cache:
        schema_path = os.path.join('/workspaces/simple', 'function_schema.json')
        try:
            with open(schema_path, 'r', encoding='utf-8') as schema_file:
                _schema_cache['schema'] = json.load(schema_file)
                logger.debug("Loaded schema from file")
        except FileNotFoundError:
            logger.error(f"Schema file not found at {schema_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in schema file: {e}")
            raise
    return _schema_cache['schema']

def validate_schema(data: Dict[str, Any]) -> None:
    try:
        schema = _load_schema()
        jsonschema.validate(instance=data, schema=schema)
        logger.debug("Schema validation successful")
    except jsonschema.ValidationError as e:
        logger.error(f"Schema validation failed: {e.message}")
        logger.error(f"Failed at path: {' -> '.join(str(p) for p in e.path)}")
        logger.error(f"Instance: {e.instance}")
        raise
    except jsonschema.SchemaError as e:
        logger.error(f"Invalid schema: {e.message}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during schema validation: {e}")
        raise

def format_validation_error(error: jsonschema.ValidationError) -> str:
    path = ' -> '.join(str(p) for p in error.path) if error.path else 'root'
    return (
        f"Validation error at {path}:\n"
        f"Message: {error.message}\n"
        f"Failed value: {error.instance}\n"
        f"Schema path: {' -> '.join(str(p) for p in error.schema_path)}"
    )

class TextProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        embeddings = self.model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        pass

class MetricsCalculator:
    @staticmethod
    def calculate_precision(retrieved_docs: List[Dict], relevant_docs: List[str]) -> float:
        if not retrieved_docs:
            return 0.0
        relevant_count = sum(
            1 for doc in retrieved_docs
            if any(rel in doc['content'] for rel in relevant_docs)
        )
        return relevant_count / len(retrieved_docs)
    
    @staticmethod
    def calculate_recall(retrieved_docs: List[Dict], relevant_docs: List[str]) -> float:
        if not relevant_docs:
            return 0.0
        retrieved_count = sum(
            1 for rel in relevant_docs
            if any(rel in doc['content'] for doc in retrieved_docs)
        )
        return retrieved_count / len(relevant_docs)
    
    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
