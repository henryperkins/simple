"""Schema loader utility."""

import json
from pathlib import Path


from typing import Dict, Any

def load_schema(schema_name: str) -> Dict[str, Any]:
    """Load a JSON schema from the schemas directory."""
    schema_path = Path(__file__).parent.parent / "schemas" / f"{schema_name}.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)
