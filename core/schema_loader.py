"""Schema loader utility."""

import json
from pathlib import Path


def load_schema(schema_name: str) -> dict:
    """Load a JSON schema from the schemas directory."""
    schema_path = Path(__file__).parent.parent / "schemas" / f"{schema_name}.json"
    with open(schema_path, "r") as f:
        return json.load(f)
