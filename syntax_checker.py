import ast
import os
from pathlib import Path

def check_syntax_in_directory(directory: str) -> None:
    """Check all Python files in a directory for syntax errors."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                check_syntax(file_path)

def check_syntax(file_path: Path) -> None:
    """Check a single Python file for syntax errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        print(f"Syntax check passed for {file_path}")
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")

if __name__ == "__main__":
    # Replace 'core' with the directory you want to check
    check_syntax_in_directory('core')
