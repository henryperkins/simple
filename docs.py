import json
import os
import aiofiles
from typing import Dict, Any, List
import jsonschema
from core.logger import LoggerSetup
from datetime import datetime
from utils import validate_schema

# Initialize logger for this module
logger = LoggerSetup.get_logger("docs")

async def write_analysis_to_markdown(results: Dict[str, Any], output_path: str) -> None:
    """
    Write the analysis results to a single comprehensive markdown file.

    This function generates a structured markdown document containing:
    - File summaries
    - Changelogs
    - Class documentation
    - Function documentation
    - Source code sections

    Args:
        results (Dict[str, Any]): The analysis results containing classes and functions
        output_path (str): The directory where the markdown file will be saved

    Raises:
        OSError: If there are issues creating directories or writing files
        Exception: For other unexpected errors during documentation generation
    """
    logger.info(f"Starting documentation generation in {output_path}")
    
    try:
        os.makedirs(output_path, exist_ok=True)
        logger.debug(f"Created output directory: {output_path}")
        
        output_file = os.path.join(output_path, "complete_documentation.md")
        logger.debug(f"Writing documentation to: {output_file}")
        
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as md_file:
            # Process each file's analysis
            for filepath, analysis in results.items():
                logger.debug(f"Processing file: {filepath}")
                
                # Add required fields if missing
                if "summary" not in analysis:
                    analysis["summary"] = "No summary available"
                if "changelog" not in analysis:
                    analysis["changelog"] = []
                if "classes" not in analysis:
                    analysis["classes"] = []  # Ensure classes field is present
                if "functions" not in analysis:
                    analysis["functions"] = []
                if "file_content" not in analysis:
                    analysis["file_content"] = [{"content": ""}]
                
                # Add timestamp to changelog entries if missing
                for entry in analysis["changelog"]:
                    if isinstance(entry, dict) and "timestamp" not in entry:
                        entry["timestamp"] = datetime.now().isoformat()
                
                # Ensure changelog is a list
                if not isinstance(analysis["changelog"], list):
                    logger.error(f"Changelog is not a list for {filepath}: {analysis['changelog']}")
                    analysis["changelog"] = []
                
                # Log changelog for debugging
                logger.debug(f"Changelog for {filepath}: {analysis['changelog']}")
                
                # Validate the analysis data
                try:
                    validate_schema(analysis)
                    logger.debug(f"Schema validation successful for {filepath}")
                except jsonschema.ValidationError as ve:
                    logger.error(f"Schema validation failed for {filepath}: {ve}")
                    raise
                
                # Write the documentation sections
                await write_module_header(md_file, filepath)
                await write_overview(md_file, analysis)
                await write_classes_section(md_file, analysis.get("classes", []))
                await write_functions_section(md_file, analysis.get("functions", []))
                await write_recent_changes_section(md_file, analysis.get("changelog", []))
                await write_source_code_section(md_file, analysis.get("file_content", []))
            
            logger.info("Documentation generation completed successfully")
            
    except OSError as e:
        logger.error(f"File system error during documentation generation: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during documentation generation: {e}")
        raise

async def write_module_header(md_file, filepath: str) -> None:
    """Write the module header."""
    module_name = os.path.basename(filepath).replace('.py', '')
    await md_file.write(f"# Module: {module_name}\n\n")

async def write_overview(md_file, analysis: Dict[str, Any]) -> None:
    """Write the overview section with general statistics."""
    await md_file.write("## Overview\n")
    await md_file.write(f"**File:** `{analysis.get('file_path', 'unknown')}`\n")
    await md_file.write(f"**Description:** {analysis.get('summary', 'No description available')}\n\n")

async def write_classes_section(md_file, classes: List[Dict[str, Any]]) -> None:
    """Write the classes section."""
    await md_file.write("## Classes\n\n")
    await md_file.write("| Class | Inherits From | Complexity Score* |\n")
    await md_file.write("|-------|---------------|------------------|\n")
    for class_info in classes:
        base_classes = ', '.join(class_info.get('base_classes', []))
        complexity_score = class_info.get('summary', '-')
        await md_file.write(f"| `{class_info['name']}` | `{base_classes}` | {complexity_score} |\n")
    await md_file.write("\n### Class Methods\n\n")
    await md_file.write("| Class | Method | Parameters | Returns | Complexity Score* |\n")
    await md_file.write("|-------|--------|------------|---------|------------------|\n")
    for class_info in classes:
        for method in class_info.get('methods', []):
            params = ', '.join([f"{p['name']}: {p['type']}" for p in method.get('params', [])])
            returns = method.get('returns', {}).get('type', 'None')
            complexity_score = method.get('complexity_score', '-')
            await md_file.write(f"| `{class_info['name']}` | `{method['name']}` | `({params})` | `{returns}` | {complexity_score} |\n")
    await md_file.write("\n")

async def write_functions_section(md_file, functions: List[Dict[str, Any]]) -> None:
    """Write the functions section."""
    await md_file.write("## Functions\n\n")
    await md_file.write("| Function | Parameters | Returns | Complexity Score* |\n")
    await md_file.write("|----------|------------|---------|------------------|\n")
    for func_info in functions:
        params = ', '.join([f"{p['name']}: {p['type']}" for p in func_info.get('params', [])])
        returns = func_info.get('returns', {}).get('type', 'None')
        complexity_score = func_info.get('complexity_score', '-')
        await md_file.write(f"| `{func_info['name']}` | `({params})` | `{returns}` | {complexity_score} |\n")
    await md_file.write("\n")

async def write_recent_changes_section(md_file, changelog: List[Dict[str, Any]]) -> None:
    """Write the recent changes section."""
    await md_file.write("## Recent Changes\n")
    for entry in changelog:
        timestamp = entry.get('timestamp', datetime.now().isoformat())
        change = entry.get('change', 'No description provided')
        await md_file.write(f"- [{timestamp}] {change}\n")
    await md_file.write("\n")

async def write_source_code_section(md_file, file_content: List[Dict[str, Any]]) -> None:
    """Write the source code section."""
    if not file_content or not file_content[0].get('content'):
        return
        
    await md_file.write("## Source Code\n\n")
    await md_file.write("```python\n")
    await md_file.write(file_content[0]['content'])
    await md_file.write("\n```\n")
