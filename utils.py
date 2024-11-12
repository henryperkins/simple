# utils.py

import os
import ast
import json
import hashlib
import time
from typing import Any, Dict, Optional, List, Union, TypedDict, Literal
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from core.logger import LoggerSetup
from jsonschema import validate, ValidationError

# Initialize logger
logger = LoggerSetup.get_logger("utils")

# Cache for schemas
_schema_cache: Dict[str, Any] = {}

# Define TypedDicts for schema validation
class ParameterProperty(TypedDict):
    type: str
    description: str

class Parameters(TypedDict):
    type: Literal["object"]
    properties: Dict[str, ParameterProperty]
    required: List[str]

class FunctionSchema(TypedDict):
    name: str
    description: str
    parameters: Parameters

def generate_hash(content: str) -> str:
    """Generate a SHA-256 hash of the given content."""
    return hashlib.sha256(content.encode()).hexdigest()

def load_json_file(filepath: str, max_retries: int = 3) -> Dict[str, Any]:
    """Load a JSON file with retry logic."""
    for attempt in range(max_retries):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading JSON file {filepath}: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
    return {}

def save_json_file(filepath: str, data: Dict[str, Any], max_retries: int = 3) -> None:
    """Save data to a JSON file with retry logic."""
    for attempt in range(max_retries):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return
        except OSError as e:
            logger.error(f"Failed to save file {filepath}: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logger.error(f"Unexpected error saving JSON file {filepath}: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

def create_timestamp() -> str:
    """Create a timestamp in ISO format."""
    return datetime.now().isoformat()

def ensure_directory(directory: str) -> None:
    """Ensure that a directory exists."""
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        raise

def validate_file_path(filepath: str, extension: Optional[str] = None) -> bool:
    """Validate if a file path exists and optionally check its extension."""
    if not os.path.exists(filepath):
        return False
    if extension and not filepath.endswith(extension):
        return False
    return True

def create_error_result(error_type: str, error_message: str) -> Dict[str, Any]:
    """Create a standardized error result."""
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
    """Add parent information to AST nodes."""
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            setattr(child, 'parent', parent)

def get_file_stats(filepath: str) -> Dict[str, Any]:
    """Get file statistics."""
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
    """Filter files in a directory based on a pattern and exclusion list."""
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
    """Normalize and return the absolute path."""
    return os.path.normpath(os.path.abspath(path))

def get_relative_path(path: str, base_path: str) -> str:
    """Get the relative path from a base path."""
    return os.path.relpath(path, base_path)

def is_python_file(filepath: str) -> bool:
    """Check if a file is a valid Python file."""
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
    """Convert a changelog to a string format."""
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
    """Format function data into a standardized response."""
    result = function_data.copy()
    result["changelog"] = convert_changelog(result.get("changelog"))
    result.setdefault("summary", "No summary available")
    result.setdefault("docstring", "")
    result.setdefault("params", [])
    result.setdefault("returns", {"type": "None", "description": ""})
    result.setdefault("functions", [])
    return result

def validate_function_data(data: Dict[str, Any]) -> None:
    """Validate function data against a schema."""
    try:
        if "changelog" in data:
            data["changelog"] = convert_changelog(data["changelog"])
        schema = _load_schema()
        validate(instance=data, schema=schema)
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise

def get_annotation(node: Optional[ast.AST]) -> str:
    """Get the annotation of an AST node."""
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
    """Format parsed sections into a standardized response."""
    logger.debug(f"Formatting response with sections: {sections}")
    return {
        "summary": sections.get("summary", "No summary available"),
        "docstring": sections.get("docstring", "No documentation available"),
        "params": sections.get("params", []),
        "returns": sections.get("returns", {"type": "None", "description": ""}),
        "examples": sections.get("examples", []),
        "classes": sections.get("classes", []),
        "functions": sections.get("functions", [])
    }

def _load_schema() -> Dict[str, Any]:
    """Load the JSON schema for validation."""
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

def validate_schema(parsed_response: Dict[str, Any]) -> None:
    """Validate the parsed response against a predefined schema."""
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "docstring": {"type": "string"},
            "params": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["name", "type", "description"]
                }
            },
            "returns": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["type", "description"]
            },
            "examples": {"type": "array", "items": {"type": "string"}},
            "classes": {"type": "array", "items": {"type": "string"}},
            "functions": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["summary", "docstring", "params", "returns", "classes", "functions"]
    }
    
    try:
        validate(instance=parsed_response, schema=schema)
    except ValidationError as e:
        raise ValueError(f"Schema validation error: {e.message}")

def format_validation_error(error: ValidationError) -> str:
    """Format a validation error message."""
    path = ' -> '.join(str(p) for p in error.path) if error.path else 'root'
    return (
        f"Validation error at {path}:\n"
        f"Message: {error.message}\n"
        f"Failed value: {error.instance}\n"
        f"Schema path: {' -> '.join(str(p) for p in error.schema_path)}"
    )

class TextProcessor:
    """Handles text processing tasks such as similarity calculation."""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        embeddings = self.model.encode([text1, text2])
        similarity = cosine_similarity(np.array([embeddings[0]]), np.array([embeddings[1]]))[0][0]
        return float(similarity)
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract keywords from text."""
        return []  # Placeholder for keyword extraction logic

class MetricsCalculator:
    """Calculates precision, recall, and F1 score for document retrieval tasks."""
    
    @staticmethod
    def calculate_precision(retrieved_docs: List[Dict], relevant_docs: List[str]) -> float:
        """Calculate precision for retrieved documents."""
        if not retrieved_docs:
            return 0.0
        relevant_count = sum(
            1 for doc in retrieved_docs
            if any(rel in doc['content'] for rel in relevant_docs)
        )
        return relevant_count / len(retrieved_docs)
    
    @staticmethod
    def calculate_recall(retrieved_docs: List[Dict], relevant_docs: List[str]) -> float:
        """Calculate recall for retrieved documents."""
        if not relevant_docs:
            return 0.0
        retrieved_count = sum(
            1 for rel in relevant_docs
            if any(rel in doc['content'] for doc in retrieved_docs)
        )
        return retrieved_count / len(relevant_docs)
    
    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        """Calculate F1 score based on precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)