import os
import aiofiles
from typing import Dict, Any, List
import jsonschema

# Load the function schema
with open('function_schema.json') as schema_file:
    function_schema = json.load(schema_file)

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
    try:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, "complete_documentation.md")
        
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as md_file:
            await write_header(md_file)
            await write_overview(md_file, results)
            
            for filepath, analysis in results.items():
                # Validate the analysis data against the schema
                jsonschema.validate(instance=analysis, schema=function_schema)
                await write_file_section(md_file, filepath, analysis)
                
        logger.info(f"Documentation written to {output_file}")
        
    except jsonschema.ValidationError as e:
        logger.error(f"Schema validation error: {e}")
        raise
    except OSError as e:
        logger.error(f"File system error writing documentation: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error writing documentation: {e}")
        raise

async def write_header(md_file) -> None:
    """Write the documentation header."""
    await md_file.write("# Code Analysis Documentation\n\n")
    await md_file.write("## Overview\n\n")

async def write_overview(md_file, results: Dict[str, Any]) -> None:
    """Write the overview section with general statistics."""
    total_files = len(results)
    total_classes = sum(len(analysis.get("classes", [])) for analysis in results.values())
    total_functions = sum(len(analysis.get("functions", [])) for analysis in results.values())
    
    await md_file.write(f"- Total Files Analyzed: {total_files}\n")
    await md_file.write(f"- Total Classes: {total_classes}\n")
    await md_file.write(f"- Total Functions: {total_functions}\n\n")

async def write_file_section(md_file, filepath: str, analysis: Dict[str, Any]) -> None:
    """Write a complete section for a single file."""
    await md_file.write(f"## File: {filepath}\n\n")
    
    if analysis.get("summary"):
        await write_summary_section(md_file, analysis["summary"])
    
    if analysis.get("changelog"):
        await write_changelog_section(md_file, analysis["changelog"])
    
    if analysis.get("classes"):
        await write_classes_section(md_file, analysis["classes"])
    
    if analysis.get("functions"):
        await write_functions_section(md_file, analysis["functions"])
    
    await write_source_code_section(md_file, analysis)
    await md_file.write("\n---\n\n")

async def write_summary_section(md_file, summary: str) -> None:
    """Write the summary section."""
    await md_file.write("### Summary\n\n")
    await md_file.write(f"{summary}\n\n")

async def write_changelog_section(md_file, changelog: List[Dict[str, Any]]) -> None:
    """
    Write the changelog section.

    Args:
        md_file: The file object to write to
        changelog (List[Dict[str, Any]]): List of changelog entries
    """
    await md_file.write("### Changelog\n\n")
    for change in changelog:
        await md_file.write(f"- {change['timestamp']}: {change['change']}\n")
    await md_file.write("\n")

async def write_classes_section(md_file, classes: List[Dict[str, Any]]) -> None:
    """
    Write the classes section.

    Args:
        md_file: The file object to write to
        classes (List[Dict[str, Any]]): List of class information dictionaries
    """
    await md_file.write("### Classes\n\n")
    for class_info in classes:
        await write_class_details(md_file, class_info)

async def write_class_details(md_file, class_info: Dict[str, Any]) -> None:
    """Write details for a single class."""
    await md_file.write(f"#### Class: {class_info['name']}\n\n")
    await md_file.write(f"{class_info['docstring']}\n\n")
    await write_class_methods(md_file, class_info)
    await write_class_attributes(md_file, class_info)
    await write_class_instance_variables(md_file, class_info)
    await write_class_base_classes(md_file, class_info)

async def write_class_methods(md_file, class_info: Dict[str, Any]) -> None:
    """Write methods for a single class."""
    await md_file.write("##### Methods\n\n")
    for method in class_info['methods']:
        await md_file.write(f"- **{method['name']}**: {method['docstring']}\n")

async def write_class_attributes(md_file, class_info: Dict[str, Any]) -> None:
    """Write attributes for a single class."""
    await md_file.write("##### Attributes\n\n")
    for attribute in class_info['attributes']:
        await md_file.write(f"- **{attribute['name']}**: {attribute['type']}\n")

async def write_class_instance_variables(md_file, class_info: Dict[str, Any]) -> None:
    """Write instance variables for a single class."""
    await md_file.write("##### Instance Variables\n\n")
    for instance_var in class_info['instance_variables']:
        await md_file.write(f"- **{instance_var['name']}** (line {instance_var['line_number']})\n")

async def write_class_base_classes(md_file, class_info: Dict[str, Any]) -> None:
    """Write base classes for a single class."""
    await md_file.write("##### Base Classes\n\n")
    for base_class in class_info['base_classes']:
        await md_file.write(f"- **{base_class}**\n")

async def write_functions_section(md_file, functions: List[Dict[str, Any]]) -> None:
    """
    Write the functions section.

    Args:
        md_file: The file object to write to
        functions (List[Dict[str, Any]]): List of function information dictionaries
    """
    await md_file.write("### Functions\n\n")
    for func_info in functions:
        await write_function_details(md_file, func_info)

async def write_function_details(md_file, func_info: Dict[str, Any]) -> None:
    """Write details for a single function."""
    await md_file.write(f"#### Function: {func_info['name']}\n\n")
    await md_file.write(f"{func_info['docstring']}\n\n")
    await write_function_params(md_file, func_info)
    await write_function_return_type(md_file, func_info)
    await write_function_complexity_metrics(md_file, func_info)

async def write_function_params(md_file, func_info: Dict[str, Any]) -> None:
    """Write parameters for a single function."""
    await md_file.write("##### Parameters\n\n")
    for param in func_info['params']:
        await md_file.write(f"- **{param['name']}**: {param['type']} (type hint: {param['has_type_hint']})\n")

async def write_function_return_type(md_file, func_info: Dict[str, Any]) -> None:
    """Write return type for a single function."""
    await md_file.write("##### Return Type\n\n")
    await md_file.write(f"{func_info['returns']['type']} (type hint: {func_info['returns']['has_type_hint']})\n")

async def write_function_complexity_metrics(md_file, func_info: Dict[str, Any]) -> None:
    """Write complexity metrics for a single function."""
    await md_file.write("##### Complexity Metrics\n\n")
    await md_file.write(f"Cyclomatic Complexity: {func_info['complexity_score']}\n")

async def write_source_code_section(md_file, analysis: Dict[str, Any]) -> None:
    """Write the source code section."""
    await md_file.write("### Source Code\n\n")
    await md_file.write(f"```python\n{analysis['code']}\n```\n")