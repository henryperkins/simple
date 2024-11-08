import json
import os
import aiofiles
from typing import Dict, Any, List
import jsonschema
from core.logger import LoggerSetup
from datetime import datetime

# Initialize logger for this module
logger = LoggerSetup.get_logger("docs")

# Load the function schema
try:
    with open('function_schema.json') as schema_file:
        function_schema = json.load(schema_file)
        logger.debug("Successfully loaded function schema")
except Exception as e:
    logger.error(f"Failed to load function schema: {e}")
    raise

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
            # Validate the input data against schema
            try:
                # Ensure each file's data matches the schema
                for filepath, analysis in results.items():
                    # Add timestamp to changelog entries if missing
                    if "changelog" in analysis and isinstance(analysis["changelog"], list):
                        for entry in analysis["changelog"]:
                            if isinstance(entry, dict) and "timestamp" not in entry:
                                entry["timestamp"] = datetime.now().isoformat()
                    
                    jsonschema.validate(instance=analysis, schema=function_schema)
                logger.debug("Input data validation successful")
            except jsonschema.ValidationError as ve:
                logger.error(f"Schema validation failed: {ve}")
                raise
            
            await write_header(md_file)
            logger.debug("Wrote documentation header")
            
            await write_overview(md_file, results)
            logger.debug("Wrote overview section")
            
            for filepath, analysis in results.items():
                logger.debug(f"Processing file: {filepath}")
                await write_file_section(md_file, filepath, analysis)
            
            logger.info("Documentation generation completed successfully")
            
    except OSError as e:
        logger.error(f"File system error during documentation generation: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during documentation generation: {e}")
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
    
    if analysis.get("file_content"):
        await write_source_code_section(md_file, analysis["file_content"])
    
    await md_file.write("\n---\n\n")

async def write_summary_section(md_file, summary: str) -> None:
    """Write the summary section."""
    await md_file.write("### Summary\n\n")
    await md_file.write(f"{summary}\n\n")

async def write_changelog_section(md_file, changelog: List[Dict[str, Any]]) -> None:
    """Write the changelog section."""
    await md_file.write("### Changelog\n\n")
    for entry in changelog:
        timestamp = entry.get('timestamp', datetime.now().isoformat())
        change = entry.get('change', 'No description provided')
        await md_file.write(f"- {timestamp}: {change}\n")
    await md_file.write("\n")

async def write_classes_section(md_file, classes: List[Dict[str, Any]]) -> None:
    """Write the classes section."""
    await md_file.write("### Classes\n\n")
    for class_info in classes:
        await write_class_details(md_file, class_info)

async def write_class_details(md_file, class_info: Dict[str, Any]) -> None:
    """Write details for a single class."""
    await md_file.write(f"#### Class: {class_info['name']}\n\n")
    
    if class_info.get('docstring'):
        await md_file.write(f"{class_info['docstring']}\n\n")
    
    if class_info.get('code'):
        await md_file.write("```python\n")
        await md_file.write(class_info['code'])
        await md_file.write("\n```\n\n")
    
    await write_class_methods(md_file, class_info)
    await write_class_attributes(md_file, class_info)
    await write_class_instance_variables(md_file, class_info)
    await write_class_base_classes(md_file, class_info)

async def write_class_methods(md_file, class_info: Dict[str, Any]) -> None:
    """Write methods for a single class."""
    if not class_info.get('methods'):
        return
        
    await md_file.write("##### Methods\n\n")
    for method in class_info['methods']:
        await md_file.write(f"###### {method['name']}\n\n")
        if method.get('docstring'):
            await md_file.write(f"{method['docstring']}\n\n")
        if method.get('code'):
            await md_file.write("```python\n")
            await md_file.write(method['code'])
            await md_file.write("\n```\n\n")
        # Write method parameters
        if method.get('params'):
            await md_file.write("Parameters:\n")
            for param in method['params']:
                type_hint = " (typed)" if param.get('has_type_hint') else ""
                await md_file.write(f"- {param['name']}: {param['type']}{type_hint}\n")
            await md_file.write("\n")

async def write_class_attributes(md_file, class_info: Dict[str, Any]) -> None:
    """Write attributes for a single class."""
    if not class_info.get('attributes'):
        return
        
    await md_file.write("##### Attributes\n\n")
    for attribute in class_info['attributes']:
        await md_file.write(f"- **{attribute['name']}**: {attribute['type']}\n")
    await md_file.write("\n")

async def write_class_instance_variables(md_file, class_info: Dict[str, Any]) -> None:
    """Write instance variables for a single class."""
    if not class_info.get('instance_variables'):
        return
        
    await md_file.write("##### Instance Variables\n\n")
    for instance_var in class_info['instance_variables']:
        await md_file.write(f"- **{instance_var['name']}** (line {instance_var['line_number']})\n")
    await md_file.write("\n")

async def write_class_base_classes(md_file, class_info: Dict[str, Any]) -> None:
    """Write base classes for a single class."""
    if not class_info.get('base_classes'):
        return
        
    await md_file.write("##### Base Classes\n\n")
    for base_class in class_info['base_classes']:
        await md_file.write(f"- {base_class}\n")
    await md_file.write("\n")

async def write_functions_section(md_file, functions: List[Dict[str, Any]]) -> None:
    """Write the functions section."""
    await md_file.write("### Functions\n\n")
    for func_info in functions:
        await write_function_details(md_file, func_info)

async def write_function_details(md_file, func_info: Dict[str, Any]) -> None:
    """Write details for a single function."""
    await md_file.write(f"#### Function: {func_info['name']}\n\n")
    
    if func_info.get('docstring'):
        await md_file.write(f"{func_info['docstring']}\n\n")
    
    if func_info.get('code'):
        await md_file.write("```python\n")
        await md_file.write(func_info['code'])
        await md_file.write("\n```\n\n")
    
    await write_function_params(md_file, func_info)
    await write_function_return_type(md_file, func_info)
    await write_function_complexity_metrics(md_file, func_info)

async def write_function_params(md_file, func_info: Dict[str, Any]) -> None:
    """Write parameters for a single function."""
    if not func_info.get('params'):
        return
        
    await md_file.write("##### Parameters\n\n")
    for param in func_info['params']:
        type_hint = " (typed)" if param.get('has_type_hint') else ""
        await md_file.write(f"- **{param['name']}**: {param['type']}{type_hint}\n")
    await md_file.write("\n")

async def write_function_return_type(md_file, func_info: Dict[str, Any]) -> None:
    """Write return type for a single function."""
    if not func_info.get('returns'):
        return
        
    await md_file.write("##### Return Type\n\n")
    returns = func_info['returns']
    type_hint = " (typed)" if returns.get('has_type_hint') else ""
    await md_file.write(f"{returns['type']}{type_hint}\n\n")

async def write_function_complexity_metrics(md_file, func_info: Dict[str, Any]) -> None:
    """Write complexity metrics for a single function."""
    await md_file.write("##### Complexity Metrics\n\n")
    metrics = [
        ("Cyclomatic Complexity", func_info.get('complexity_score', 'N/A')),
        ("Cognitive Complexity", func_info.get('cognitive_complexity', 'N/A')),
        ("Is Async", "Yes" if func_info.get('is_async') else "No"),
        ("Is Generator", "Yes" if func_info.get('is_generator') else "No"),
        ("Is Recursive", "Yes" if func_info.get('is_recursive') else "No")
    ]
    
    for metric_name, metric_value in metrics:
        await md_file.write(f"- {metric_name}: {metric_value}\n")
    await md_file.write("\n")

async def write_source_code_section(md_file, file_content: List[Dict[str, Any]]) -> None:
    """Write the source code section."""
    if not file_content or not file_content[0].get('content'):
        return
        
    await md_file.write("### Source Code\n\n")
    await md_file.write("```python\n")
    await md_file.write(file_content[0]['content'])
    await md_file.write("\n```\n")