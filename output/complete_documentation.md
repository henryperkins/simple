# Code Analysis Documentation

## Overview

- Total Files Analyzed: 15
- Total Classes: 7
- Total Functions: 43

## File: cloned_repo/docs.py

### Summary

Found 0 classes and 18 functions | Total lines of code: 276 | Average function complexity: 3.17 | Maximum function complexity: 10 | Documentation coverage: 100.0%

### Changelog

- 2024-11-09T05:43:06.922356: Started code analysis
- 2024-11-09T05:43:06.924339: Analyzed function: write_analysis_to_markdown
- 2024-11-09T05:43:06.924825: Analyzed function: write_header
- 2024-11-09T05:43:06.925804: Analyzed function: write_overview
- 2024-11-09T05:43:06.926908: Analyzed function: write_file_section
- 2024-11-09T05:43:06.927446: Analyzed function: write_summary_section
- 2024-11-09T05:43:06.928281: Analyzed function: write_changelog_section
- 2024-11-09T05:43:06.928887: Analyzed function: write_classes_section
- 2024-11-09T05:43:06.929914: Analyzed function: write_class_details
- 2024-11-09T05:43:06.931318: Analyzed function: write_class_methods
- 2024-11-09T05:43:06.932147: Analyzed function: write_class_attributes
- 2024-11-09T05:43:06.932979: Analyzed function: write_class_instance_variables
- 2024-11-09T05:43:06.933754: Analyzed function: write_class_base_classes
- 2024-11-09T05:43:06.934427: Analyzed function: write_functions_section
- 2024-11-09T05:43:06.935493: Analyzed function: write_function_details
- 2024-11-09T05:43:06.936453: Analyzed function: write_function_params
- 2024-11-09T05:43:06.937316: Analyzed function: write_function_return_type
- 2024-11-09T05:43:06.938465: Analyzed function: write_function_complexity_metrics
- 2024-11-09T05:43:06.939371: Analyzed function: write_source_code_section
- 2024-11-09T05:43:06.940804: Completed code analysis

### Functions

#### Function: write_analysis_to_markdown

Error: Documentation generation failed.

```python
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
```

##### Parameters

- **results**: Dict[Any] (typed)
- **output_path**: str (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 10
- Cognitive Complexity: 27
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: write_header

Error: Documentation generation failed.

```python
async def write_header(md_file) -> None:
    """Write the documentation header."""
    await md_file.write("# Code Analysis Documentation\n\n")
    await md_file.write("## Overview\n\n")
```

##### Parameters

- **md_file**: Any

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 1
- Cognitive Complexity: 0
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: write_overview

Error: Documentation generation failed.

```python
async def write_overview(md_file, results: Dict[str, Any]) -> None:
    """Write the overview section with general statistics."""
    total_files = len(results)
    total_classes = sum(len(analysis.get("classes", [])) for analysis in results.values())
    total_functions = sum(len(analysis.get("functions", [])) for analysis in results.values())
    
    await md_file.write(f"- Total Files Analyzed: {total_files}\n")
    await md_file.write(f"- Total Classes: {total_classes}\n")
    await md_file.write(f"- Total Functions: {total_functions}\n\n")
```

##### Parameters

- **md_file**: Any
- **results**: Dict[Any] (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 1
- Cognitive Complexity: 0
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: write_file_section

Error: Documentation generation failed.

```python
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
```

##### Parameters

- **md_file**: Any
- **filepath**: str (typed)
- **analysis**: Dict[Any] (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 6
- Cognitive Complexity: 5
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: write_summary_section

Error: Documentation generation failed.

```python
async def write_summary_section(md_file, summary: str) -> None:
    """Write the summary section."""
    await md_file.write("### Summary\n\n")
    await md_file.write(f"{summary}\n\n")
```

##### Parameters

- **md_file**: Any
- **summary**: str (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 1
- Cognitive Complexity: 0
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: write_changelog_section

Error: Documentation generation failed.

```python
async def write_changelog_section(md_file, changelog: List[Dict[str, Any]]) -> None:
    """Write the changelog section."""
    await md_file.write("### Changelog\n\n")
    for entry in changelog:
        timestamp = entry.get('timestamp', datetime.now().isoformat())
        change = entry.get('change', 'No description provided')
        await md_file.write(f"- {timestamp}: {change}\n")
    await md_file.write("\n")
```

##### Parameters

- **md_file**: Any
- **changelog**: List[Dict[Any]] (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 2
- Cognitive Complexity: 1
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: write_classes_section

Error: Documentation generation failed.

```python
async def write_classes_section(md_file, classes: List[Dict[str, Any]]) -> None:
    """Write the classes section."""
    await md_file.write("### Classes\n\n")
    for class_info in classes:
        await write_class_details(md_file, class_info)
```

##### Parameters

- **md_file**: Any
- **classes**: List[Dict[Any]] (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 2
- Cognitive Complexity: 1
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: write_class_details

Error: Documentation generation failed.

```python
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
```

##### Parameters

- **md_file**: Any
- **class_info**: Dict[Any] (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 3
- Cognitive Complexity: 2
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: write_class_methods

Error: Documentation generation failed.

```python
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
```

##### Parameters

- **md_file**: Any
- **class_info**: Dict[Any] (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 7
- Cognitive Complexity: 12
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: write_class_attributes

Error: Documentation generation failed.

```python
async def write_class_attributes(md_file, class_info: Dict[str, Any]) -> None:
    """Write attributes for a single class."""
    if not class_info.get('attributes'):
        return
        
    await md_file.write("##### Attributes\n\n")
    for attribute in class_info['attributes']:
        await md_file.write(f"- **{attribute['name']}**: {attribute['type']}\n")
    await md_file.write("\n")
```

##### Parameters

- **md_file**: Any
- **class_info**: Dict[Any] (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 3
- Cognitive Complexity: 3
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: write_class_instance_variables

Error: Documentation generation failed.

```python
async def write_class_instance_variables(md_file, class_info: Dict[str, Any]) -> None:
    """Write instance variables for a single class."""
    if not class_info.get('instance_variables'):
        return
        
    await md_file.write("##### Instance Variables\n\n")
    for instance_var in class_info['instance_variables']:
        await md_file.write(f"- **{instance_var['name']}** (line {instance_var['line_number']})\n")
    await md_file.write("\n")
```

##### Parameters

- **md_file**: Any
- **class_info**: Dict[Any] (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 3
- Cognitive Complexity: 3
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: write_class_base_classes

Error: Documentation generation failed.

```python
async def write_class_base_classes(md_file, class_info: Dict[str, Any]) -> None:
    """Write base classes for a single class."""
    if not class_info.get('base_classes'):
        return
        
    await md_file.write("##### Base Classes\n\n")
    for base_class in class_info['base_classes']:
        await md_file.write(f"- {base_class}\n")
    await md_file.write("\n")
```

##### Parameters

- **md_file**: Any
- **class_info**: Dict[Any] (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 3
- Cognitive Complexity: 3
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: write_functions_section

Error: Documentation generation failed.

```python
async def write_functions_section(md_file, functions: List[Dict[str, Any]]) -> None:
    """Write the functions section."""
    await md_file.write("### Functions\n\n")
    for func_info in functions:
        await write_function_details(md_file, func_info)
```

##### Parameters

- **md_file**: Any
- **functions**: List[Dict[Any]] (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 2
- Cognitive Complexity: 1
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: write_function_details

Error: Documentation generation failed.

```python
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
```

##### Parameters

- **md_file**: Any
- **func_info**: Dict[Any] (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 3
- Cognitive Complexity: 2
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: write_function_params

Error: Documentation generation failed.

```python
async def write_function_params(md_file, func_info: Dict[str, Any]) -> None:
    """Write parameters for a single function."""
    if not func_info.get('params'):
        return
        
    await md_file.write("##### Parameters\n\n")
    for param in func_info['params']:
        type_hint = " (typed)" if param.get('has_type_hint') else ""
        await md_file.write(f"- **{param['name']}**: {param['type']}{type_hint}\n")
    await md_file.write("\n")
```

##### Parameters

- **md_file**: Any
- **func_info**: Dict[Any] (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 3
- Cognitive Complexity: 3
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: write_function_return_type

Error: Documentation generation failed.

```python
async def write_function_return_type(md_file, func_info: Dict[str, Any]) -> None:
    """Write return type for a single function."""
    if not func_info.get('returns'):
        return
        
    await md_file.write("##### Return Type\n\n")
    returns = func_info['returns']
    type_hint = " (typed)" if returns.get('has_type_hint') else ""
    await md_file.write(f"{returns['type']}{type_hint}\n\n")
```

##### Parameters

- **md_file**: Any
- **func_info**: Dict[Any] (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 2
- Cognitive Complexity: 2
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: write_function_complexity_metrics

Error: Documentation generation failed.

```python
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
```

##### Parameters

- **md_file**: Any
- **func_info**: Dict[Any] (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 2
- Cognitive Complexity: 1
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: write_source_code_section

Error: Documentation generation failed.

```python
async def write_source_code_section(md_file, file_content: List[Dict[str, Any]]) -> None:
    """Write the source code section."""
    if not file_content or not file_content[0].get('content'):
        return
        
    await md_file.write("### Source Code\n\n")
    await md_file.write("```python\n")
    await md_file.write(file_content[0]['content'])
    await md_file.write("\n```\n")
```

##### Parameters

- **md_file**: Any
- **file_content**: List[Dict[Any]] (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 3
- Cognitive Complexity: 3
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

### Source Code

```python
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
```

---

## File: cloned_repo/metrics.py

### Summary

Found 3 classes and 0 functions | Total lines of code: 335 | Documentation coverage: 0.0%

### Changelog

- 2024-11-09T05:43:06.844007: Started code analysis
- 2024-11-09T05:43:06.872181: Analyzed class: CodeMetrics
- 2024-11-09T05:43:06.879655: Analyzed class: CognitiveComplexityVisitor
- 2024-11-09T05:43:06.887451: Analyzed class: HalsteadVisitor
- 2024-11-09T05:43:06.888708: Completed code analysis

### Classes

#### Class: CodeMetrics

##### Methods

###### __init__

```python
def __init__(self):
        self.total_functions = 0
        self.total_classes = 0
        self.total_lines = 0
        self.docstring_coverage = 0.0
        self.type_hint_coverage = 0.0
        self.avg_complexity = 0.0
        self.max_complexity = 0
        self.cognitive_complexity = 0
        self.halstead_metrics = defaultdict(float)
        self.type_hints_stats = defaultdict(int)
        self.quality_issues = []
        logger.debug("Initialized CodeMetrics instance.")
```

Parameters:
- self: Any

###### calculate_complexity

Calculate the cyclomatic complexity score for a function or method.
Args:
    node (ast.AST): The AST node representing a function or method.
Returns:
    int: The cyclomatic complexity score.

```python
def calculate_complexity(self, node: ast.AST) -> int:
        """
        Calculate the cyclomatic complexity score for a function or method.
        Args:
            node (ast.AST): The AST node representing a function or method.
        Returns:
            int: The cyclomatic complexity score.
        """
        name = getattr(node, 'name', 'unknown')
        complexity = 1  # Start with one for the function entry point
        try:
            for subnode in ast.walk(node):
                if isinstance(subnode, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp)):
                    complexity += 1
            logger.debug(f"Calculated complexity for node {name}: {complexity}")
            self.max_complexity = max(self.max_complexity, complexity)
        except Exception as e:
            logger.error(f"Error calculating complexity for node {name}: {e}")
        return complexity
```

Parameters:
- self: Any
- node: ast.AST (typed)

###### calculate_cognitive_complexity

Calculate the cognitive complexity score for a function or method.
This metric measures how difficult the code is to understand, considering:
- Nesting depth (loops, conditionals)
- Logical operators
- Recursion
- Multiple exit points

Args:
    node (ast.AST): The AST node representing a function or method.
Returns:
    int: The cognitive complexity score.

```python
def calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """
        Calculate the cognitive complexity score for a function or method.
        This metric measures how difficult the code is to understand, considering:
        - Nesting depth (loops, conditionals)
        - Logical operators
        - Recursion
        - Multiple exit points
        
        Args:
            node (ast.AST): The AST node representing a function or method.
        Returns:
            int: The cognitive complexity score.
        """
        name = getattr(node, 'name', 'unknown')
        try:
            cognitive_score = 0
            nesting_level = 0

            class CognitiveComplexityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.score = 0
                    self.depth = 0

                def visit_If(self, node):
                    self.score += (1 + self.depth)  # Base cost + nesting
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1

                def visit_For(self, node):
                    self.score += (1 + self.depth)
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1

                def visit_While(self, node):
                    self.score += (1 + self.depth)
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1

                def visit_BoolOp(self, node):
                    # Add complexity for boolean operations (and, or)
                    self.score += len(node.values) - 1
                    self.generic_visit(node)

                def visit_Try(self, node):
                    self.score += 1  # Base cost for try block
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1
                    # Add cost for each except handler
                    self.score += len(node.handlers)

                def visit_Return(self, node):
                    # Add complexity for multiple return statements
                    if self.depth > 0:
                        self.score += 1
                    self.generic_visit(node)

            visitor = CognitiveComplexityVisitor()
            visitor.visit(node)
            cognitive_score = visitor.score

            logger.debug(f"Calculated cognitive complexity for node {name}: {cognitive_score}")
            return cognitive_score

        except Exception as e:
            logger.error(f"Error calculating cognitive complexity for node {name}: {e}")
            return 0
```

Parameters:
- self: Any
- node: ast.AST (typed)

###### calculate_halstead_metrics

Calculate Halstead metrics for a function or method.
Halstead metrics include:
- Program Length (N): Total number of operators and operands
- Program Vocabulary (n): Number of unique operators and operands
- Program Volume (V): N * log2(n)
- Difficulty (D): Related to error proneness
- Effort (E): Mental effort required to implement

Args:
    node (ast.AST): The AST node representing a function or method.
Returns:
    Dict[str, float]: The Halstead metrics.

```python
def calculate_halstead_metrics(self, node: ast.AST) -> Dict[str, float]:
        """
        Calculate Halstead metrics for a function or method.
        Halstead metrics include:
        - Program Length (N): Total number of operators and operands
        - Program Vocabulary (n): Number of unique operators and operands
        - Program Volume (V): N * log2(n)
        - Difficulty (D): Related to error proneness
        - Effort (E): Mental effort required to implement
        
        Args:
            node (ast.AST): The AST node representing a function or method.
        Returns:
            Dict[str, float]: The Halstead metrics.
        """
        name = getattr(node, 'name', 'unknown')
        try:
            class HalsteadVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.operators = set()
                    self.operands = set()
                    self.total_operators = 0
                    self.total_operands = 0

                def visit_BinOp(self, node):
                    self.operators.add(type(node.op).__name__)
                    self.total_operators += 1
                    self.generic_visit(node)

                def visit_UnaryOp(self, node):
                    self.operators.add(type(node.op).__name__)
                    self.total_operators += 1
                    self.generic_visit(node)

                def visit_BoolOp(self, node):
                    self.operators.add(type(node.op).__name__)
                    self.total_operators += 1
                    self.generic_visit(node)

                def visit_Compare(self, node):
                    for op in node.ops:
                        self.operators.add(type(op).__name__)
                        self.total_operators += 1
                    self.generic_visit(node)

                def visit_Name(self, node):
                    self.operands.add(node.id)
                    self.total_operands += 1
                    self.generic_visit(node)

                def visit_Constant(self, node):
                    self.operands.add(str(node.value))
                    self.total_operands += 1
                    self.generic_visit(node)

            visitor = HalsteadVisitor()
            visitor.visit(node)

            # Calculate basic Halstead metrics
            n1 = len(visitor.operators)  # Number of unique operators
            n2 = len(visitor.operands)   # Number of unique operands
            N1 = visitor.total_operators # Total operators
            N2 = visitor.total_operands  # Total operands

            # Prevent division by zero and log(0)
            if n1 + n2 == 0:
                return {
                    "program_length": 0,
                    "vocabulary_size": 0,
                    "program_volume": 0,
                    "difficulty": 0,
                    "effort": 0
                }

            import math
            program_length = N1 + N2
            vocabulary_size = n1 + n2
            volume = program_length * math.log2(vocabulary_size) if vocabulary_size > 0 else 0
            difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
            effort = difficulty * volume

            metrics = {
                "program_length": program_length,
                "vocabulary_size": vocabulary_size,
                "program_volume": volume,
                "difficulty": difficulty,
                "effort": effort
            }

            logger.debug(f"Calculated Halstead metrics for node {name}: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating Halstead metrics for node {name}: {e}")
            return {
                "program_length": 0,
                "vocabulary_size": 0,
                "program_volume": 0,
                "difficulty": 0,
                "effort": 0
            }
```

Parameters:
- self: Any
- node: ast.AST (typed)

###### analyze_function_quality

Analyze function quality and add recommendations.
Args:
    function_info (Dict[str, Any]): The function details.

```python
def analyze_function_quality(self, function_info: Dict[str, Any]) -> None:
        """
        Analyze function quality and add recommendations.
        Args:
            function_info (Dict[str, Any]): The function details.
        """
        name = function_info.get('name', 'unknown')
        score = function_info.get('complexity_score', 0)
        logger.debug(f"Analyzing quality for function: {name}")
        if score > 10:
            msg = (
                f"Function '{name}' has high complexity ({score}). "
                "Consider breaking it down."
            )
            self.quality_issues.append(msg)
            logger.info(msg)
        if not function_info.get("docstring"):
            msg = f"Function '{name}' lacks a docstring."
            self.quality_issues.append(msg)
            logger.info(msg)
        params_without_types = [
            p["name"] for p in function_info.get("params", [])
            if not p.get("has_type_hint")
        ]
        if params_without_types:
            params_str = ", ".join(params_without_types)
            msg = (
                f"Function '{name}' has parameters without type hints: "
                f"{params_str}"
            )
            self.quality_issues.append(msg)
            logger.info(msg)
```

Parameters:
- self: Any
- function_info: Dict[Any] (typed)

###### analyze_class_quality

Analyze class quality and add recommendations.
Args:
    class_info (Dict[str, Any]): The class details.

```python
def analyze_class_quality(self, class_info: Dict[str, Any]) -> None:
        """
        Analyze class quality and add recommendations.
        Args:
            class_info (Dict[str, Any]): The class details.
        """
        name = class_info.get('name', 'unknown')
        logger.debug(f"Analyzing quality for class: {name}")
        if not class_info.get("docstring"):
            msg = f"Class '{name}' lacks a docstring."
            self.quality_issues.append(msg)
            logger.info(msg)
        method_count = len(class_info.get("methods", []))
        if method_count > 10:
            msg = (
                f"Class '{name}' has many methods ({method_count}). "
                "Consider splitting it."
            )
            self.quality_issues.append(msg)
            logger.info(msg)
```

Parameters:
- self: Any
- class_info: Dict[Any] (typed)

###### update_type_hint_stats

Update type hint statistics based on function information.
Args:
    function_info (Dict[str, Any]): The function details.

```python
def update_type_hint_stats(self, function_info: Dict[str, Any]) -> None:
        """
        Update type hint statistics based on function information.
        Args:
            function_info (Dict[str, Any]): The function details.
        """
        total_hints_possible = len(function_info.get("params", [])) + 1  # Including return type
        hints_present = sum(
            1 for p in function_info.get("params", []) if p.get("has_type_hint")
        )
        if function_info.get("return_type", {}).get("has_type_hint", False):
            hints_present += 1
        self.type_hints_stats["total_possible"] += total_hints_possible
        self.type_hints_stats["total_present"] += hints_present
        logger.debug(f"Updated type hint stats: {self.type_hints_stats}")
```

Parameters:
- self: Any
- function_info: Dict[Any] (typed)

###### calculate_final_metrics

Calculate final metrics after processing all items.
Args:
    all_items (List[Dict[str, Any]]): List of all functions and methods analyzed.

```python
def calculate_final_metrics(self, all_items: List[Dict[str, Any]]) -> None:
        """
        Calculate final metrics after processing all items.
        Args:
            all_items (List[Dict[str, Any]]): List of all functions and methods analyzed.
        """
        total_items = len(all_items)
        logger.debug(f"Calculating final metrics for {total_items} items.")
        if total_items > 0:
            items_with_doc = sum(1 for item in all_items if item.get("docstring"))
            self.docstring_coverage = (items_with_doc / total_items) * 100
            total_complexity = sum(
                item.get("complexity_score", 0)
                for item in all_items
            )
            self.avg_complexity = total_complexity / total_items if total_items else 0
        if self.type_hints_stats["total_possible"] > 0:
            self.type_hint_coverage = (
                self.type_hints_stats["total_present"] /
                self.type_hints_stats["total_possible"]
            ) * 100
        logger.info(
            f"Final metrics calculated: Docstring coverage: {self.docstring_coverage:.2f}%, "
            f"Type hint coverage: {self.type_hint_coverage:.2f}%, "
            f"Average complexity: {self.avg_complexity:.2f}, "
            f"Max complexity: {self.max_complexity}"
        )
```

Parameters:
- self: Any
- all_items: List[Dict[Any]] (typed)

###### get_summary

Generate a comprehensive summary of code metrics.
Returns:
    Dict[str, Any]: The summary of code metrics.

```python
def get_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of code metrics.
        Returns:
            Dict[str, Any]: The summary of code metrics.
        """
        summary = {
            "total_classes": self.total_classes,
            "total_functions": self.total_functions,
            "total_lines": self.total_lines,
            "docstring_coverage_percentage": round(self.docstring_coverage, 2),
            "type_hint_coverage_percentage": round(self.type_hint_coverage, 2),
            "average_complexity": round(self.avg_complexity, 2),
            "max_complexity": self.max_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "halstead_metrics": dict(self.halstead_metrics),
            "quality_recommendations": self.quality_issues
        }
        logger.debug(f"Generated summary: {summary}")
        return summary
```

Parameters:
- self: Any

##### Instance Variables

- **total_functions** (line 11)
- **total_classes** (line 12)
- **total_lines** (line 13)
- **docstring_coverage** (line 14)
- **type_hint_coverage** (line 15)
- **avg_complexity** (line 16)
- **max_complexity** (line 17)
- **cognitive_complexity** (line 18)
- **halstead_metrics** (line 19)
- **type_hints_stats** (line 20)
- **quality_issues** (line 21)

#### Class: CognitiveComplexityVisitor

##### Methods

###### __init__

```python
def __init__(self):
                    self.score = 0
                    self.depth = 0
```

Parameters:
- self: Any

###### visit_If

```python
def visit_If(self, node):
                    self.score += (1 + self.depth)  # Base cost + nesting
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1
```

Parameters:
- self: Any
- node: Any

###### visit_For

```python
def visit_For(self, node):
                    self.score += (1 + self.depth)
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1
```

Parameters:
- self: Any
- node: Any

###### visit_While

```python
def visit_While(self, node):
                    self.score += (1 + self.depth)
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1
```

Parameters:
- self: Any
- node: Any

###### visit_BoolOp

```python
def visit_BoolOp(self, node):
                    # Add complexity for boolean operations (and, or)
                    self.score += len(node.values) - 1
                    self.generic_visit(node)
```

Parameters:
- self: Any
- node: Any

###### visit_Try

```python
def visit_Try(self, node):
                    self.score += 1  # Base cost for try block
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1
                    # Add cost for each except handler
                    self.score += len(node.handlers)
```

Parameters:
- self: Any
- node: Any

###### visit_Return

```python
def visit_Return(self, node):
                    # Add complexity for multiple return statements
                    if self.depth > 0:
                        self.score += 1
                    self.generic_visit(node)
```

Parameters:
- self: Any
- node: Any

##### Instance Variables

- **score** (line 65)
- **depth** (line 66)

##### Base Classes

- ast.NodeVisitor

#### Class: HalsteadVisitor

##### Methods

###### __init__

```python
def __init__(self):
                    self.operators = set()
                    self.operands = set()
                    self.total_operators = 0
                    self.total_operands = 0
```

Parameters:
- self: Any

###### visit_BinOp

```python
def visit_BinOp(self, node):
                    self.operators.add(type(node.op).__name__)
                    self.total_operators += 1
                    self.generic_visit(node)
```

Parameters:
- self: Any
- node: Any

###### visit_UnaryOp

```python
def visit_UnaryOp(self, node):
                    self.operators.add(type(node.op).__name__)
                    self.total_operators += 1
                    self.generic_visit(node)
```

Parameters:
- self: Any
- node: Any

###### visit_BoolOp

```python
def visit_BoolOp(self, node):
                    self.operators.add(type(node.op).__name__)
                    self.total_operators += 1
                    self.generic_visit(node)
```

Parameters:
- self: Any
- node: Any

###### visit_Compare

```python
def visit_Compare(self, node):
                    for op in node.ops:
                        self.operators.add(type(op).__name__)
                        self.total_operators += 1
                    self.generic_visit(node)
```

Parameters:
- self: Any
- node: Any

###### visit_Name

```python
def visit_Name(self, node):
                    self.operands.add(node.id)
                    self.total_operands += 1
                    self.generic_visit(node)
```

Parameters:
- self: Any
- node: Any

###### visit_Constant

```python
def visit_Constant(self, node):
                    self.operands.add(str(node.value))
                    self.total_operands += 1
                    self.generic_visit(node)
```

Parameters:
- self: Any
- node: Any

##### Instance Variables

- **operators** (line 135)
- **operands** (line 136)
- **total_operators** (line 137)
- **total_operands** (line 138)

##### Base Classes

- ast.NodeVisitor

### Source Code

```python
import ast
from collections import defaultdict
from typing import Any, Dict, List
from core.logger import LoggerSetup

# Initialize a logger specifically for this module
logger = LoggerSetup.get_logger("metrics")

class CodeMetrics:
    def __init__(self):
        self.total_functions = 0
        self.total_classes = 0
        self.total_lines = 0
        self.docstring_coverage = 0.0
        self.type_hint_coverage = 0.0
        self.avg_complexity = 0.0
        self.max_complexity = 0
        self.cognitive_complexity = 0
        self.halstead_metrics = defaultdict(float)
        self.type_hints_stats = defaultdict(int)
        self.quality_issues = []
        logger.debug("Initialized CodeMetrics instance.")

    def calculate_complexity(self, node: ast.AST) -> int:
        """
        Calculate the cyclomatic complexity score for a function or method.
        Args:
            node (ast.AST): The AST node representing a function or method.
        Returns:
            int: The cyclomatic complexity score.
        """
        name = getattr(node, 'name', 'unknown')
        complexity = 1  # Start with one for the function entry point
        try:
            for subnode in ast.walk(node):
                if isinstance(subnode, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp)):
                    complexity += 1
            logger.debug(f"Calculated complexity for node {name}: {complexity}")
            self.max_complexity = max(self.max_complexity, complexity)
        except Exception as e:
            logger.error(f"Error calculating complexity for node {name}: {e}")
        return complexity

    def calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """
        Calculate the cognitive complexity score for a function or method.
        This metric measures how difficult the code is to understand, considering:
        - Nesting depth (loops, conditionals)
        - Logical operators
        - Recursion
        - Multiple exit points
        
        Args:
            node (ast.AST): The AST node representing a function or method.
        Returns:
            int: The cognitive complexity score.
        """
        name = getattr(node, 'name', 'unknown')
        try:
            cognitive_score = 0
            nesting_level = 0

            class CognitiveComplexityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.score = 0
                    self.depth = 0

                def visit_If(self, node):
                    self.score += (1 + self.depth)  # Base cost + nesting
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1

                def visit_For(self, node):
                    self.score += (1 + self.depth)
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1

                def visit_While(self, node):
                    self.score += (1 + self.depth)
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1

                def visit_BoolOp(self, node):
                    # Add complexity for boolean operations (and, or)
                    self.score += len(node.values) - 1
                    self.generic_visit(node)

                def visit_Try(self, node):
                    self.score += 1  # Base cost for try block
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1
                    # Add cost for each except handler
                    self.score += len(node.handlers)

                def visit_Return(self, node):
                    # Add complexity for multiple return statements
                    if self.depth > 0:
                        self.score += 1
                    self.generic_visit(node)

            visitor = CognitiveComplexityVisitor()
            visitor.visit(node)
            cognitive_score = visitor.score

            logger.debug(f"Calculated cognitive complexity for node {name}: {cognitive_score}")
            return cognitive_score

        except Exception as e:
            logger.error(f"Error calculating cognitive complexity for node {name}: {e}")
            return 0

    def calculate_halstead_metrics(self, node: ast.AST) -> Dict[str, float]:
        """
        Calculate Halstead metrics for a function or method.
        Halstead metrics include:
        - Program Length (N): Total number of operators and operands
        - Program Vocabulary (n): Number of unique operators and operands
        - Program Volume (V): N * log2(n)
        - Difficulty (D): Related to error proneness
        - Effort (E): Mental effort required to implement
        
        Args:
            node (ast.AST): The AST node representing a function or method.
        Returns:
            Dict[str, float]: The Halstead metrics.
        """
        name = getattr(node, 'name', 'unknown')
        try:
            class HalsteadVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.operators = set()
                    self.operands = set()
                    self.total_operators = 0
                    self.total_operands = 0

                def visit_BinOp(self, node):
                    self.operators.add(type(node.op).__name__)
                    self.total_operators += 1
                    self.generic_visit(node)

                def visit_UnaryOp(self, node):
                    self.operators.add(type(node.op).__name__)
                    self.total_operators += 1
                    self.generic_visit(node)

                def visit_BoolOp(self, node):
                    self.operators.add(type(node.op).__name__)
                    self.total_operators += 1
                    self.generic_visit(node)

                def visit_Compare(self, node):
                    for op in node.ops:
                        self.operators.add(type(op).__name__)
                        self.total_operators += 1
                    self.generic_visit(node)

                def visit_Name(self, node):
                    self.operands.add(node.id)
                    self.total_operands += 1
                    self.generic_visit(node)

                def visit_Constant(self, node):
                    self.operands.add(str(node.value))
                    self.total_operands += 1
                    self.generic_visit(node)

            visitor = HalsteadVisitor()
            visitor.visit(node)

            # Calculate basic Halstead metrics
            n1 = len(visitor.operators)  # Number of unique operators
            n2 = len(visitor.operands)   # Number of unique operands
            N1 = visitor.total_operators # Total operators
            N2 = visitor.total_operands  # Total operands

            # Prevent division by zero and log(0)
            if n1 + n2 == 0:
                return {
                    "program_length": 0,
                    "vocabulary_size": 0,
                    "program_volume": 0,
                    "difficulty": 0,
                    "effort": 0
                }

            import math
            program_length = N1 + N2
            vocabulary_size = n1 + n2
            volume = program_length * math.log2(vocabulary_size) if vocabulary_size > 0 else 0
            difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
            effort = difficulty * volume

            metrics = {
                "program_length": program_length,
                "vocabulary_size": vocabulary_size,
                "program_volume": volume,
                "difficulty": difficulty,
                "effort": effort
            }

            logger.debug(f"Calculated Halstead metrics for node {name}: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating Halstead metrics for node {name}: {e}")
            return {
                "program_length": 0,
                "vocabulary_size": 0,
                "program_volume": 0,
                "difficulty": 0,
                "effort": 0
            }

    def analyze_function_quality(self, function_info: Dict[str, Any]) -> None:
        """
        Analyze function quality and add recommendations.
        Args:
            function_info (Dict[str, Any]): The function details.
        """
        name = function_info.get('name', 'unknown')
        score = function_info.get('complexity_score', 0)
        logger.debug(f"Analyzing quality for function: {name}")
        if score > 10:
            msg = (
                f"Function '{name}' has high complexity ({score}). "
                "Consider breaking it down."
            )
            self.quality_issues.append(msg)
            logger.info(msg)
        if not function_info.get("docstring"):
            msg = f"Function '{name}' lacks a docstring."
            self.quality_issues.append(msg)
            logger.info(msg)
        params_without_types = [
            p["name"] for p in function_info.get("params", [])
            if not p.get("has_type_hint")
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
        """
        Analyze class quality and add recommendations.
        Args:
            class_info (Dict[str, Any]): The class details.
        """
        name = class_info.get('name', 'unknown')
        logger.debug(f"Analyzing quality for class: {name}")
        if not class_info.get("docstring"):
            msg = f"Class '{name}' lacks a docstring."
            self.quality_issues.append(msg)
            logger.info(msg)
        method_count = len(class_info.get("methods", []))
        if method_count > 10:
            msg = (
                f"Class '{name}' has many methods ({method_count}). "
                "Consider splitting it."
            )
            self.quality_issues.append(msg)
            logger.info(msg)

    def update_type_hint_stats(self, function_info: Dict[str, Any]) -> None:
        """
        Update type hint statistics based on function information.
        Args:
            function_info (Dict[str, Any]): The function details.
        """
        total_hints_possible = len(function_info.get("params", [])) + 1  # Including return type
        hints_present = sum(
            1 for p in function_info.get("params", []) if p.get("has_type_hint")
        )
        if function_info.get("return_type", {}).get("has_type_hint", False):
            hints_present += 1
        self.type_hints_stats["total_possible"] += total_hints_possible
        self.type_hints_stats["total_present"] += hints_present
        logger.debug(f"Updated type hint stats: {self.type_hints_stats}")

    def calculate_final_metrics(self, all_items: List[Dict[str, Any]]) -> None:
        """
        Calculate final metrics after processing all items.
        Args:
            all_items (List[Dict[str, Any]]): List of all functions and methods analyzed.
        """
        total_items = len(all_items)
        logger.debug(f"Calculating final metrics for {total_items} items.")
        if total_items > 0:
            items_with_doc = sum(1 for item in all_items if item.get("docstring"))
            self.docstring_coverage = (items_with_doc / total_items) * 100
            total_complexity = sum(
                item.get("complexity_score", 0)
                for item in all_items
            )
            self.avg_complexity = total_complexity / total_items if total_items else 0
        if self.type_hints_stats["total_possible"] > 0:
            self.type_hint_coverage = (
                self.type_hints_stats["total_present"] /
                self.type_hints_stats["total_possible"]
            ) * 100
        logger.info(
            f"Final metrics calculated: Docstring coverage: {self.docstring_coverage:.2f}%, "
            f"Type hint coverage: {self.type_hint_coverage:.2f}%, "
            f"Average complexity: {self.avg_complexity:.2f}, "
            f"Max complexity: {self.max_complexity}"
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of code metrics.
        Returns:
            Dict[str, Any]: The summary of code metrics.
        """
        summary = {
            "total_classes": self.total_classes,
            "total_functions": self.total_functions,
            "total_lines": self.total_lines,
            "docstring_coverage_percentage": round(self.docstring_coverage, 2),
            "type_hint_coverage_percentage": round(self.type_hint_coverage, 2),
            "average_complexity": round(self.avg_complexity, 2),
            "max_complexity": self.max_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "halstead_metrics": dict(self.halstead_metrics),
            "quality_recommendations": self.quality_issues
        }
        logger.debug(f"Generated summary: {summary}")
        return summary
```

---

## File: cloned_repo/api_interaction.py

### Summary

Found 0 classes and 4 functions | Total lines of code: 226 | Average function complexity: 4.00 | Maximum function complexity: 7 | Documentation coverage: 0.0%

### Changelog

- 2024-11-09T05:43:07.049089: Started code analysis
- 2024-11-09T05:43:07.049986: Analyzed function: get_service_headers
- 2024-11-09T05:43:07.050397: Analyzed function: get_azure_endpoint
- 2024-11-09T05:43:07.053073: Analyzed function: make_openai_request
- 2024-11-09T05:43:07.056141: Analyzed function: analyze_function_with_openai
- 2024-11-09T05:43:07.057076: Completed code analysis

### Functions

#### Function: get_service_headers

Error: Documentation generation failed.

```python
def get_service_headers(service: str) -> dict:
    logger.debug(f"Getting headers for service: {service}")
    headers = {"Content-Type": "application/json"}
    if service == "azure":
        headers["api-key"] = azure_api_key
    elif service == "openai":
        headers["Authorization"] = f"Bearer {openai_api_key}"
    else:
        logger.error(f"Unsupported service: {service}")
        raise ValueError(f"Unsupported service: {service}")
    logger.debug(f"Headers set: {headers}")
    return headers
```

##### Parameters

- **service**: str (typed)

##### Return Type

dict (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 3
- Cognitive Complexity: 3
- Is Async: No
- Is Generator: No
- Is Recursive: No

#### Function: get_azure_endpoint

Error: Documentation generation failed.

```python
def get_azure_endpoint() -> str:
    logger.debug(f"Azure endpoint retrieved: {azure_endpoint}")
    return azure_endpoint
```

##### Return Type

str (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 1
- Cognitive Complexity: 0
- Is Async: No
- Is Generator: No
- Is Recursive: No

#### Function: make_openai_request

Error: Documentation generation failed.

```python
async def make_openai_request(
    messages: list, functions: list, service: str, model_name: Optional[str] = None
) -> Dict[str, Any]:
    logger.info(f"Preparing to make request to {service} service")
    headers = get_service_headers(service)
    
    if service == "azure":
        endpoint = f"{get_azure_endpoint()}/openai/deployments/{azure_deployment_name}/chat/completions?api-version={azure_api_version}"
        model_name = azure_model_name
    else:
        endpoint = "https://api.openai.com/v1/chat/completions"

    payload = {
        "model": model_name,
        "messages": messages,
        "functions": functions,
        "function_call": "auto",
    }

    logger.debug(f"Using endpoint: {endpoint}")
    logger.debug(f"Using headers: {headers}")
    logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

    retries = 3
    base_backoff = 2
    
    for attempt in tqdm(range(1, retries + 1), desc="API Request Progress"):
        logger.info(f"Attempt {attempt} of {retries}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint, headers=headers, json=payload, timeout=30
                ) as response:
                    logger.debug(f"Response status: {response.status}")
                    if response.status == 200:
                        result = await response.json()
                        logger.debug(f"Received response: {json.dumps(result, indent=2)}")
                        return result
                    else:
                        error_msg = await response.text()
                        logger.warning(
                            f"Attempt {attempt}: API request failed with status {response.status}: {error_msg}"
                        )
                        sentry_sdk.capture_message(
                            f"Attempt {attempt}: API request failed with status {response.status}: {error_msg}"
                        )
        except aiohttp.ClientError as e:
            logger.error(f"Attempt {attempt}: Client error during API request: {e}")
            sentry_sdk.capture_exception(e)
        except asyncio.TimeoutError:
            logger.error(f"Attempt {attempt}: API request timed out.")
            sentry_sdk.capture_message("API request timed out.")
        except Exception as e:
            logger.error(f"Attempt {attempt}: Unexpected exception during API request: {e}")
            sentry_sdk.capture_exception(e)

        sleep_time = base_backoff ** attempt
        logger.debug(f"Retrying API request in {sleep_time} seconds (Attempt {attempt}/{retries})")
        await asyncio.sleep(sleep_time)

    logger.error("Exceeded maximum retries for API request.")
    return {"error": "Failed to get a successful response from the API."}
```

##### Parameters

- **messages**: list (typed)
- **functions**: list (typed)
- **service**: str (typed)
- **model_name**: Optional[str] (typed)

##### Return Type

Dict[Any] (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 5
- Cognitive Complexity: 10
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: analyze_function_with_openai

Error: Documentation generation failed.

```python
async def analyze_function_with_openai(
    function_details: Dict[str, Any], service: str
) -> Dict[str, Any]:
    function_name = function_details.get("name", "unknown")
    logger.info(f"Analyzing function: {function_name}")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates documentation.",
        },
        {
            "role": "user",
            "content": (
                f"Provide a detailed analysis for the following function:\n\n"
                f"{function_details.get('code', '')}"
            ),
        },
    ]

    try:
        # Load the function schema from the JSON file
        function_schema_path = os.path.join(os.path.dirname(__file__), 'function_schema.json')
        logger.debug(f"Loading function schema from {function_schema_path}")
        with open(function_schema_path, 'r', encoding='utf-8') as schema_file:
            function_schema = json.load(schema_file)
        logger.debug("Function schema loaded successfully")

        response = await make_openai_request(
            messages=messages,
            functions=[function_schema],
            service=service,
        )

        if "error" in response:
            logger.error(f"API returned an error: {response['error']}")
            sentry_sdk.capture_message(f"API returned an error: {response['error']}")
            return {
                "name": function_name,
                "complexity_score": function_details.get("complexity_score", "Unknown"),
                "summary": "Error during analysis.",
                "docstring": "Error: Documentation generation failed.",
                "changelog": "Error: Changelog generation failed.",
            }

        choices = response.get("choices", [])
        if not choices:
            error_msg = "Missing 'choices' in API response."
            logger.error(error_msg)
            raise KeyError(error_msg)

        response_message = choices[0].get("message", {})
        if "function_call" in response_message:
            function_args_str = response_message["function_call"].get("arguments", "{}")
            try:
                function_args = json.loads(function_args_str)
                logger.debug(f"Parsed function arguments: {function_args}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding error in function_call arguments: {e}")
                raise

            return {
                "name": function_name,
                "complexity_score": function_details.get("complexity_score", "Unknown"),
                "summary": function_args.get("summary", ""),
                "docstring": function_args.get("docstring", ""),
                "changelog": function_args.get("changelog", ""),
                "classes": function_args.get("classes", []),
                "functions": function_args.get("functions", []),
                "file_content": function_args.get("file_content", [])
            }

        error_msg = "Missing 'function_call' in API response message."
        logger.error(error_msg)
        raise KeyError(error_msg)

    except (KeyError, TypeError, json.JSONDecodeError) as e:
        logger.error(f"Error processing API response: {e}")
        sentry_sdk.capture_exception(e)
        return {
            "name": function_name,
            "complexity_score": function_details.get("complexity_score", "Unknown"),
            "summary": "Error during analysis.",
            "docstring": "Error: Documentation generation failed.",
            "changelog": "Error: Changelog generation failed.",
        }

    except Exception as e:
        logger.error(f"Unexpected error during function analysis: {e}")
        sentry_sdk.capture_exception(e)
        return {
            "name": function_name,
            "complexity_score": function_details.get("complexity_score", "Unknown"),
            "summary": "Error during analysis.",
            "docstring": "Error: Documentation generation failed.",
            "changelog": "Error: Changelog generation failed.",
        }
```

##### Parameters

- **function_details**: Dict[Any] (typed)
- **service**: str (typed)

##### Return Type

Dict[Any] (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 7
- Cognitive Complexity: 15
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

### Source Code

```python
import aiohttp
import asyncio
import json
import os
import sentry_sdk
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from core.logger import LoggerSetup
from extract.utils import validate_schema

# Initialize a logger specifically for this module
logger = LoggerSetup.get_logger("api_interaction")

# Load environment variables from .env file
load_dotenv()

# Determine which service to use
use_azure = os.getenv("USE_AZURE", "false").lower() == "true"

# Load environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
azure_api_key = os.getenv("AZURE_API_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")
azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")
azure_model_name = os.getenv("AZURE_MODEL_NAME", "gpt-4o-2024-08-06")
azure_api_version = os.getenv("AZURE_API_VERSION", "2024-08-01-preview")
sentry_dsn = os.getenv("SENTRY_DSN")

# Validate required environment variables based on the service
if use_azure:
    required_vars = {
        "AZURE_API_KEY": azure_api_key,
        "AZURE_ENDPOINT": azure_endpoint,
        "SENTRY_DSN": sentry_dsn
    }
else:
    required_vars = {
        "OPENAI_API_KEY": openai_api_key,
        "SENTRY_DSN": sentry_dsn
    }

for var_name, var_value in required_vars.items():
    if not var_value:
        logger.error(f"{var_name} is not set.")
        raise ValueError(f"{var_name} is not set.")
    else:
        logger.debug(f"{var_name} is set.")

def get_service_headers(service: str) -> dict:
    logger.debug(f"Getting headers for service: {service}")
    headers = {"Content-Type": "application/json"}
    if service == "azure":
        headers["api-key"] = azure_api_key
    elif service == "openai":
        headers["Authorization"] = f"Bearer {openai_api_key}"
    else:
        logger.error(f"Unsupported service: {service}")
        raise ValueError(f"Unsupported service: {service}")
    logger.debug(f"Headers set: {headers}")
    return headers

def get_azure_endpoint() -> str:
    logger.debug(f"Azure endpoint retrieved: {azure_endpoint}")
    return azure_endpoint

async def make_openai_request(
    messages: list, functions: list, service: str, model_name: Optional[str] = None
) -> Dict[str, Any]:
    logger.info(f"Preparing to make request to {service} service")
    headers = get_service_headers(service)
    
    if service == "azure":
        endpoint = f"{get_azure_endpoint()}/openai/deployments/{azure_deployment_name}/chat/completions?api-version={azure_api_version}"
        model_name = azure_model_name
    else:
        endpoint = "https://api.openai.com/v1/chat/completions"

    payload = {
        "model": model_name,
        "messages": messages,
        "functions": functions,
        "function_call": "auto",
    }

    logger.debug(f"Using endpoint: {endpoint}")
    logger.debug(f"Using headers: {headers}")
    logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

    retries = 3
    base_backoff = 2
    
    for attempt in tqdm(range(1, retries + 1), desc="API Request Progress"):
        logger.info(f"Attempt {attempt} of {retries}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint, headers=headers, json=payload, timeout=30
                ) as response:
                    logger.debug(f"Response status: {response.status}")
                    if response.status == 200:
                        result = await response.json()
                        logger.debug(f"Received response: {json.dumps(result, indent=2)}")
                        return result
                    else:
                        error_msg = await response.text()
                        logger.warning(
                            f"Attempt {attempt}: API request failed with status {response.status}: {error_msg}"
                        )
                        sentry_sdk.capture_message(
                            f"Attempt {attempt}: API request failed with status {response.status}: {error_msg}"
                        )
        except aiohttp.ClientError as e:
            logger.error(f"Attempt {attempt}: Client error during API request: {e}")
            sentry_sdk.capture_exception(e)
        except asyncio.TimeoutError:
            logger.error(f"Attempt {attempt}: API request timed out.")
            sentry_sdk.capture_message("API request timed out.")
        except Exception as e:
            logger.error(f"Attempt {attempt}: Unexpected exception during API request: {e}")
            sentry_sdk.capture_exception(e)

        sleep_time = base_backoff ** attempt
        logger.debug(f"Retrying API request in {sleep_time} seconds (Attempt {attempt}/{retries})")
        await asyncio.sleep(sleep_time)

    logger.error("Exceeded maximum retries for API request.")
    return {"error": "Failed to get a successful response from the API."}

async def analyze_function_with_openai(
    function_details: Dict[str, Any], service: str
) -> Dict[str, Any]:
    function_name = function_details.get("name", "unknown")
    logger.info(f"Analyzing function: {function_name}")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates documentation.",
        },
        {
            "role": "user",
            "content": (
                f"Provide a detailed analysis for the following function:\n\n"
                f"{function_details.get('code', '')}"
            ),
        },
    ]

    try:
        # Load the function schema from the JSON file
        function_schema_path = os.path.join(os.path.dirname(__file__), 'function_schema.json')
        logger.debug(f"Loading function schema from {function_schema_path}")
        with open(function_schema_path, 'r', encoding='utf-8') as schema_file:
            function_schema = json.load(schema_file)
        logger.debug("Function schema loaded successfully")

        response = await make_openai_request(
            messages=messages,
            functions=[function_schema],
            service=service,
        )

        if "error" in response:
            logger.error(f"API returned an error: {response['error']}")
            sentry_sdk.capture_message(f"API returned an error: {response['error']}")
            return {
                "name": function_name,
                "complexity_score": function_details.get("complexity_score", "Unknown"),
                "summary": "Error during analysis.",
                "docstring": "Error: Documentation generation failed.",
                "changelog": "Error: Changelog generation failed.",
            }

        choices = response.get("choices", [])
        if not choices:
            error_msg = "Missing 'choices' in API response."
            logger.error(error_msg)
            raise KeyError(error_msg)

        response_message = choices[0].get("message", {})
        if "function_call" in response_message:
            function_args_str = response_message["function_call"].get("arguments", "{}")
            try:
                function_args = json.loads(function_args_str)
                logger.debug(f"Parsed function arguments: {function_args}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding error in function_call arguments: {e}")
                raise

            return {
                "name": function_name,
                "complexity_score": function_details.get("complexity_score", "Unknown"),
                "summary": function_args.get("summary", ""),
                "docstring": function_args.get("docstring", ""),
                "changelog": function_args.get("changelog", ""),
                "classes": function_args.get("classes", []),
                "functions": function_args.get("functions", []),
                "file_content": function_args.get("file_content", [])
            }

        error_msg = "Missing 'function_call' in API response message."
        logger.error(error_msg)
        raise KeyError(error_msg)

    except (KeyError, TypeError, json.JSONDecodeError) as e:
        logger.error(f"Error processing API response: {e}")
        sentry_sdk.capture_exception(e)
        return {
            "name": function_name,
            "complexity_score": function_details.get("complexity_score", "Unknown"),
            "summary": "Error during analysis.",
            "docstring": "Error: Documentation generation failed.",
            "changelog": "Error: Changelog generation failed.",
        }

    except Exception as e:
        logger.error(f"Unexpected error during function analysis: {e}")
        sentry_sdk.capture_exception(e)
        return {
            "name": function_name,
            "complexity_score": function_details.get("complexity_score", "Unknown"),
            "summary": "Error during analysis.",
            "docstring": "Error: Documentation generation failed.",
            "changelog": "Error: Changelog generation failed.",
        }
```

---

## File: cloned_repo/files.py

### Summary

Error during extraction

### Changelog

- 2024-11-09T05:43:07.030160: Error during extraction: Additional properties are not allowed ('line_number' was unexpected)

Failed validating 'additionalProperties' in schema['properties']['classes']['items']['properties']['attributes']['items']:
    {'type': 'object',
     'properties': {'name': {'type': 'string'}, 'type': {'type': 'string'}},
     'required': ['name', 'type'],
     'additionalProperties': False}

On instance['classes'][0]['attributes'][3]:
    {'name': 'LOCK', 'type': 'Any', 'line_number': 31}

### Source Code

```python
import asyncio
import os
import json
import fnmatch
import time
import ast
import hashlib
import threading
import subprocess
import aiofiles
import shutil
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from core.logger import LoggerSetup
import sentry_sdk
from extract.code import extract_classes_and_functions_from_ast
from extract.utils import add_parent_info, validate_schema
from api_interaction import analyze_function_with_openai
from metrics import CodeMetrics

# Initialize logger for this module
logger = LoggerSetup.get_logger("files")

# Cache configuration
class CacheConfig:
    """Configuration constants for the caching system."""
    DIR = "cache"
    INDEX_FILE = os.path.join(DIR, "index.json")
    MAX_SIZE_MB = 500
    LOCK = threading.Lock()

def initialize_cache() -> None:
    """Initialize the cache directory and index file."""
    try:
        if not os.path.exists(CacheConfig.DIR):
            os.makedirs(CacheConfig.DIR)
            logger.info("Created cache directory.")
        if not os.path.exists(CacheConfig.INDEX_FILE):
            with open(CacheConfig.INDEX_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            logger.info("Initialized cache index file.")
    except OSError as e:
        logger.error(f"Error initializing cache: {e}")
        raise

def get_cache_path(key: str) -> str:
    """Generate a cache file path based on the key."""
    hashed_key = hashlib.sha256(key.encode()).hexdigest()
    cache_path = os.path.join(CacheConfig.DIR, f"{hashed_key}.json")
    logger.debug(f"Generated cache path for key {key}: {cache_path}")
    return cache_path

def load_cache_index() -> OrderedDict:
    """Load and sort the cache index by last access time."""
    with CacheConfig.LOCK:
        try:
            with open(CacheConfig.INDEX_FILE, 'r', encoding='utf-8') as f:
                index = json.load(f, object_pairs_hook=OrderedDict)
            logger.debug("Loaded cache index.")
            return OrderedDict(sorted(index.items(), key=lambda item: item[1]['last_access_time']))
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Error loading cache index: {e}")
            raise

def save_cache_index(index: OrderedDict) -> None:
    """Save the cache index to disk."""
    with CacheConfig.LOCK:
        try:
            with open(CacheConfig.INDEX_FILE, 'w', encoding='utf-8') as f:
                json.dump(index, f)
            logger.debug("Saved cache index.")
        except OSError as e:
            logger.error(f"Error saving cache index: {e}")
            raise

def cache_response(key: str, data: Dict[str, Any]) -> None:
    """Cache response data with the given key."""
    index = load_cache_index()
    cache_path = get_cache_path(key)
    with CacheConfig.LOCK:
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            index[key] = {
                'cache_path': cache_path,
                'last_access_time': time.time()
            }
            save_cache_index(index)
            logger.debug(f"Cached response for key {key}.")
        except OSError as e:
            logger.error(f"Error caching response for key {key}: {e}")
            raise

def get_cached_response(key: str) -> Dict[str, Any]:
    """Retrieve cached response based on the key."""
    index = load_cache_index()
    with CacheConfig.LOCK:
        cache_entry = index.get(key)
        if cache_entry:
            cache_path = cache_entry.get('cache_path')
            if cache_path and os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # Update last access time
                    cache_entry['last_access_time'] = time.time()
                    index.move_to_end(key)
                    save_cache_index(index)
                    logger.debug(f"Loaded cached response for key: {key}")
                    return data
                except (json.JSONDecodeError, OSError) as e:
                    logger.error(f"Error loading cached response for key {key}: {e}")
                    sentry_sdk.capture_exception(e)
            else:
                logger.warning(f"Cache file does not exist for key: {key}")
                del index[key]
                save_cache_index(index)
        else:
            logger.debug(f"No cached response found for key: {key}")
    return {}

def clear_cache(index: OrderedDict) -> None:
    """Evict least recently used cache entries if cache exceeds size limit."""
    total_size = 0
    with CacheConfig.LOCK:
        try:
            # Calculate total cache size
            for key, entry in index.items():
                cache_path = entry.get('cache_path')
                if cache_path and os.path.exists(cache_path):
                    total_size += os.path.getsize(cache_path)
            total_size_mb = total_size / (1024 * 1024)
            if total_size_mb > CacheConfig.MAX_SIZE_MB:
                logger.info("Cache size exceeded limit. Starting eviction process.")
                while total_size_mb > CacheConfig.MAX_SIZE_MB and index:
                    key, entry = index.popitem(last=False)
                    cache_path = entry.get('cache_path')
                    if cache_path and os.path.exists(cache_path):
                        file_size = os.path.getsize(cache_path)
                        os.remove(cache_path)
                        total_size -= file_size
                        total_size_mb = total_size / (1024 * 1024)
                        logger.debug(f"Removed cache file {cache_path}")
                save_cache_index(index)
                logger.info("Cache eviction completed.")
            else:
                logger.debug(f"Cache size within limit: {total_size_mb:.2f} MB")
        except OSError as e:
            logger.error(f"Error during cache cleanup: {e}")
            sentry_sdk.capture_exception(e)
            raise

async def clone_repo(repo_url: str, clone_dir: str) -> None:
    """Clone a GitHub repository into a specified directory."""
    logger.info(f"Cloning repository {repo_url} into {clone_dir}")
    try:
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)
            logger.info(f"Removed existing directory {clone_dir}")
            
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', repo_url, clone_dir],
            capture_output=True,
            text=True,
            timeout=60,
            check=True
        )
        
        if result.stderr:
            logger.warning(f"Git clone stderr: {result.stderr}")
            
        os.chmod(clone_dir, 0o755)
        logger.info(f"Successfully cloned repository {repo_url}")
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        logger.error(f"Error cloning repository {repo_url}: {e}")
        sentry_sdk.capture_exception(e)
        raise

def load_gitignore_patterns(repo_dir: str) -> List[str]:
    """Load .gitignore patterns from the repository directory."""
    gitignore_path = os.path.join(repo_dir, '.gitignore')
    try:
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            patterns = f.read().splitlines()
        logger.debug(f"Loaded .gitignore patterns from {gitignore_path}")
        return patterns
    except OSError as e:
        logger.error(f"Error loading .gitignore file: {e}")
        sentry_sdk.capture_exception(e)
        return []

def get_all_files(directory: str, exclude_patterns: Optional[List[str]] = None) -> List[str]:
    """Retrieve all Python files in the directory, excluding patterns."""
    exclude_patterns = exclude_patterns or []
    python_files: List[str] = []

    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if not any(fnmatch.fnmatch(file_path, pattern) for pattern in exclude_patterns):
                        python_files.append(file_path)
        logger.debug(f"Found {len(python_files)} Python files in {directory}")
        return python_files
    except Exception as e:
        logger.error(f"Error retrieving Python files: {e}")
        sentry_sdk.capture_exception(e)
        return []

async def process_file(filepath: str, service: str) -> Dict[str, Any]:
    """Read and parse a Python file."""
    if service not in ['azure', 'openai']:
        logger.error(f"Invalid service specified: {service}")
        return {
            "summary": "Error during extraction",
            "changelog": [{
                "change": f"Error: Invalid service '{service}' specified",
                "timestamp": datetime.now().isoformat()
            }],
            "classes": [],
            "functions": [],
            "file_content": [{"content": ""}]
        }

    try:
        # Read file content
        content = await read_file_content(filepath)
        logger.debug(f"Read {len(content)} characters from {filepath}")

        # Parse AST and add parent information
        try:
            tree = ast.parse(content)
            add_parent_info(tree)  # Add parent references to AST nodes
        except SyntaxError as e:
            logger.error(f"Syntax error in file {filepath}: {e}")
            return {
                "summary": "Error during extraction",
                "changelog": [{
                    "change": f"Syntax error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }],
                "classes": [],
                "functions": [],
                "file_content": [{"content": content}]
            }

        # Extract data using AST
        try:
            extracted_data = extract_classes_and_functions_from_ast(tree, content)
            
            # Validate extracted data
            try:
                validate_schema(extracted_data)
            except Exception as e:
                logger.error(f"Schema validation failed for {filepath}: {e}")
                sentry_sdk.capture_exception(e)
                return {
                    "summary": "Error during extraction",
                    "changelog": [{
                        "change": f"Schema validation failed: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }],
                    "classes": [],
                    "functions": [],
                    "file_content": [{"content": content}]
                }

            # Calculate additional metrics
            metrics = CodeMetrics()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity = metrics.calculate_complexity(node)
                    cognitive_complexity = metrics.calculate_cognitive_complexity(node)
                    halstead = metrics.calculate_halstead_metrics(node)
                    
                    # Update metrics in extracted data
                    for func in extracted_data.get("functions", []):
                        if func["name"] == node.name:
                            func["complexity_score"] = complexity
                            func["cognitive_complexity"] = cognitive_complexity
                            func["halstead_metrics"] = halstead

            logger.info(f"Successfully extracted data for {filepath}")
            return extracted_data

        except Exception as e:
            logger.error(f"Error extracting data from {filepath}: {e}")
            sentry_sdk.capture_exception(e)
            return {
                "summary": "Error during extraction",
                "changelog": [{
                    "change": f"Data extraction error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }],
                "classes": [],
                "functions": [],
                "file_content": [{"content": content}]
            }

    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return {
            "summary": "Error during extraction",
            "changelog": [{
                "change": "File not found",
                "timestamp": datetime.now().isoformat()
            }],
            "classes": [],
            "functions": [],
            "file_content": [{"content": ""}]
        }
    except UnicodeDecodeError:
        logger.error(f"Unicode decode error in file {filepath}")
        return {
            "summary": "Error during extraction",
            "changelog": [{
                "change": "Unicode decode error",
                "timestamp": datetime.now().isoformat()
            }],
            "classes": [],
            "functions": [],
            "file_content": [{"content": ""}]
        }
    except Exception as e:
        logger.error(f"Error processing file {filepath}: {e}")
        sentry_sdk.capture_exception(e)
        return {
            "summary": "Error during extraction",
            "changelog": [{
                "change": f"Unexpected error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }],
            "classes": [],
            "functions": [],
            "file_content": [{"content": ""}]
        }

async def read_file_content(filepath: str) -> str:
    """Read the content of a file asynchronously."""
    try:
        async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
            content = await f.read()
        logger.debug(f"Read content from {filepath}")
        return content
    except FileNotFoundError as e:
        logger.error(f"File not found: {filepath}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Unicode decode error in file {filepath}")
        raise
    except OSError as e:
        logger.error(f"OS error while reading file {filepath}: {e}")
        raise

async def analyze_and_update_functions(
    extracted_data: Dict[str, Any],
    tree: ast.AST,
    content: str,
    service: str
) -> str:
    """Analyze functions and update their docstrings."""
    for func in extracted_data.get("functions", []):
        analysis = await analyze_function_with_openai(func, service)
        content = update_function_docstring(content, tree, func, analysis)
    return content

def update_function_docstring(
    file_content: str,
    tree: ast.AST,
    function: Dict[str, Any],
    analysis: Dict[str, Any]
) -> str:
    """Update the docstring of a function."""
    new_docstring = analysis.get("docstring", "")
    if not new_docstring:
        return file_content

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function["name"]:
            file_content = insert_docstring(file_content, node, new_docstring)
            break
    return file_content

def insert_docstring(
    source: str,
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    docstring: str
) -> str:
    """Insert a docstring into a function or class."""
    lines = source.splitlines()
    start_line = node.body[0].lineno - 1
    indent = " " * (node.body[0].col_offset or 0)
    docstring_lines = [f'{indent}"""', f'{indent}{docstring}', f'{indent}"""']
    lines[start_line:start_line] = docstring_lines
    return "\n".join(lines)
```

---

## File: cloned_repo/cache.py

### Summary

Found 0 classes and 7 functions | Total lines of code: 159 | Average function complexity: 5.29 | Maximum function complexity: 12 | Documentation coverage: 100.0%

### Changelog

- 2024-11-09T05:43:07.115558: Started code analysis
- 2024-11-09T05:43:07.116304: Analyzed function: initialize_cache
- 2024-11-09T05:43:07.116928: Analyzed function: get_cache_path
- 2024-11-09T05:43:07.118276: Analyzed function: load_cache_index
- 2024-11-09T05:43:07.119117: Analyzed function: save_cache_index
- 2024-11-09T05:43:07.120137: Analyzed function: cache_response
- 2024-11-09T05:43:07.121643: Analyzed function: get_cached_response
- 2024-11-09T05:43:07.123595: Analyzed function: clear_cache
- 2024-11-09T05:43:07.124387: Completed code analysis

### Functions

#### Function: initialize_cache

Error: Documentation generation failed.

```python
def initialize_cache():
    """Initialize the cache directory and index."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        logger.info("Created cache directory.")
    if not os.path.exists(CACHE_INDEX_FILE):
        with open(CACHE_INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        logger.info("Initialized cache index file.")
```

##### Return Type

Any

##### Complexity Metrics

- Cyclomatic Complexity: 4
- Cognitive Complexity: 2
- Is Async: No
- Is Generator: No
- Is Recursive: No

#### Function: get_cache_path

Error: Documentation generation failed.

```python
def get_cache_path(key: str) -> str:
    """Generate a cache file path based on the key."""
    hashed_key = hashlib.sha256(key.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{hashed_key}.json")
    logger.debug(f"Generated cache path for key {key}: {cache_path}")
    return cache_path
```

##### Parameters

- **key**: str (typed)

##### Return Type

str (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 1
- Cognitive Complexity: 0
- Is Async: No
- Is Generator: No
- Is Recursive: No

#### Function: load_cache_index

Error: Documentation generation failed.

```python
def load_cache_index() -> OrderedDict:
    """Load the cache index, sorted by last access time."""
    with cache_lock:
        try:
            if os.path.exists(CACHE_INDEX_FILE):
                with open(CACHE_INDEX_FILE, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                    # Convert index_data to OrderedDict sorted by last_access_time
                    index = OrderedDict(sorted(index_data.items(), key=lambda item: item[1]['last_access_time']))
                    logger.debug("Loaded and sorted cache index.")
                    return index
            else:
                logger.debug("Cache index file not found. Initializing empty index.")
                return OrderedDict()
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed for cache index: {e}")
            sentry_sdk.capture_exception(e)
            return OrderedDict()
        except OSError as e:
            logger.error(f"OS error while loading cache index: {e}")
            sentry_sdk.capture_exception(e)
            return OrderedDict()
```

##### Return Type

OrderedDict (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 5
- Cognitive Complexity: 9
- Is Async: No
- Is Generator: No
- Is Recursive: No

#### Function: save_cache_index

Error: Documentation generation failed.

```python
def save_cache_index(index: OrderedDict) -> None:
    """Save the cache index."""
    with cache_lock:
        try:
            with open(CACHE_INDEX_FILE, 'w', encoding='utf-8') as f:
                json.dump(index, f)
            logger.debug("Saved cache index.")
        except OSError as e:
            logger.error(f"OS error while saving cache index: {e}")
            sentry_sdk.capture_exception(e)
```

##### Parameters

- **index**: OrderedDict (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 4
- Cognitive Complexity: 2
- Is Async: No
- Is Generator: No
- Is Recursive: No

#### Function: cache_response

Error: Documentation generation failed.

```python
def cache_response(key: str, data: Dict[str, Any]) -> None:
    """Cache the response data with the given key."""
    index = load_cache_index()
    cache_path = get_cache_path(key)
    with cache_lock:
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            index[key] = {
                'cache_path': cache_path,
                'last_access_time': time.time()
            }
            save_cache_index(index)
            logger.debug(f"Cached response for key: {key}")
            clear_cache(index)
        except OSError as e:
            logger.error(f"Failed to cache response for key {key}: {e}")
            sentry_sdk.capture_exception(e)
```

##### Parameters

- **key**: str (typed)
- **data**: Dict[Any] (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 4
- Cognitive Complexity: 2
- Is Async: No
- Is Generator: No
- Is Recursive: No

#### Function: get_cached_response

Error: Documentation generation failed.

```python
def get_cached_response(key: str) -> Dict[str, Any]:
    """Retrieve cached response based on the key."""
    index = load_cache_index()
    with cache_lock:
        cache_entry = index.get(key)
        if cache_entry:
            cache_path = cache_entry.get('cache_path')
            if cache_path and os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # Update last access time
                    cache_entry['last_access_time'] = time.time()
                    index.move_to_end(key)  # Move to end to reflect recent access
                    save_cache_index(index)
                    logger.debug(f"Loaded cached response for key: {key}")
                    return data
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decoding failed for cached response {key}: {e}")
                    sentry_sdk.capture_exception(e)
                except OSError as e:
                    logger.error(f"OS error while loading cached response for key {key}: {e}")
                    sentry_sdk.capture_exception(e)
            else:
                logger.warning(f"Cache file does not exist for key: {key}")
                # Remove invalid cache entry
                del index[key]
                save_cache_index(index)
        else:
            logger.debug(f"No cached response found for key: {key}")
    return {}
```

##### Parameters

- **key**: str (typed)

##### Return Type

Dict[Any] (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 7
- Cognitive Complexity: 8
- Is Async: No
- Is Generator: No
- Is Recursive: No

#### Function: clear_cache

Error: Documentation generation failed.

```python
def clear_cache(index: OrderedDict) -> None:
    """Evict least recently used cache entries if cache exceeds size limit."""
    total_size = 0
    with cache_lock:
        for key, entry in index.items():
            cache_path = entry.get('cache_path')
            if cache_path and os.path.exists(cache_path):
                try:
                    file_size = os.path.getsize(cache_path)
                    total_size += file_size
                except OSError as e:
                    logger.error(f"Error getting size for cache file {cache_path}: {e}")
                    sentry_sdk.capture_exception(e)
                    continue
        total_size_mb = total_size / (1024 * 1024)
        if total_size_mb > CACHE_MAX_SIZE_MB:
            logger.info("Cache size exceeded limit. Starting eviction process.")
            while total_size_mb > CACHE_MAX_SIZE_MB and index:
                # Pop the least recently used item
                key, entry = index.popitem(last=False)
                cache_path = entry.get('cache_path')
                if cache_path and os.path.exists(cache_path):
                    try:
                        file_size = os.path.getsize(cache_path)
                        os.remove(cache_path)
                        total_size -= file_size
                        total_size_mb = total_size / (1024 * 1024)
                        logger.debug(f"Removed cache file {cache_path} for key {key}")
                    except OSError as e:
                        logger.error(f"Error removing cache file {cache_path}: {e}")
                        sentry_sdk.capture_exception(e)
                else:
                    logger.debug(f"Cache file {cache_path} does not exist.")
            save_cache_index(index)
            logger.info("Cache eviction completed.")
        else:
            logger.debug(f"Cache size within limit: {total_size_mb:.2f} MB")
```

##### Parameters

- **index**: OrderedDict (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 12
- Cognitive Complexity: 16
- Is Async: No
- Is Generator: No
- Is Recursive: No

### Source Code

```python
import os
import json
import time
import hashlib
import threading
from collections import OrderedDict
from typing import Any, Dict
from core.logger import LoggerSetup
import sentry_sdk

# Initialize logger for this module
logger = LoggerSetup.get_logger("cache")
cache_lock = threading.Lock()

# Cache directory and configuration
CACHE_DIR = "cache"
CACHE_INDEX_FILE = os.path.join(CACHE_DIR, "index.json")
CACHE_MAX_SIZE_MB = 500

def initialize_cache():
    """Initialize the cache directory and index."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        logger.info("Created cache directory.")
    if not os.path.exists(CACHE_INDEX_FILE):
        with open(CACHE_INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        logger.info("Initialized cache index file.")

def get_cache_path(key: str) -> str:
    """Generate a cache file path based on the key."""
    hashed_key = hashlib.sha256(key.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{hashed_key}.json")
    logger.debug(f"Generated cache path for key {key}: {cache_path}")
    return cache_path

def load_cache_index() -> OrderedDict:
    """Load the cache index, sorted by last access time."""
    with cache_lock:
        try:
            if os.path.exists(CACHE_INDEX_FILE):
                with open(CACHE_INDEX_FILE, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                    # Convert index_data to OrderedDict sorted by last_access_time
                    index = OrderedDict(sorted(index_data.items(), key=lambda item: item[1]['last_access_time']))
                    logger.debug("Loaded and sorted cache index.")
                    return index
            else:
                logger.debug("Cache index file not found. Initializing empty index.")
                return OrderedDict()
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed for cache index: {e}")
            sentry_sdk.capture_exception(e)
            return OrderedDict()
        except OSError as e:
            logger.error(f"OS error while loading cache index: {e}")
            sentry_sdk.capture_exception(e)
            return OrderedDict()

def save_cache_index(index: OrderedDict) -> None:
    """Save the cache index."""
    with cache_lock:
        try:
            with open(CACHE_INDEX_FILE, 'w', encoding='utf-8') as f:
                json.dump(index, f)
            logger.debug("Saved cache index.")
        except OSError as e:
            logger.error(f"OS error while saving cache index: {e}")
            sentry_sdk.capture_exception(e)


def cache_response(key: str, data: Dict[str, Any]) -> None:
    """Cache the response data with the given key."""
    index = load_cache_index()
    cache_path = get_cache_path(key)
    with cache_lock:
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            index[key] = {
                'cache_path': cache_path,
                'last_access_time': time.time()
            }
            save_cache_index(index)
            logger.debug(f"Cached response for key: {key}")
            clear_cache(index)
        except OSError as e:
            logger.error(f"Failed to cache response for key {key}: {e}")
            sentry_sdk.capture_exception(e)

def get_cached_response(key: str) -> Dict[str, Any]:
    """Retrieve cached response based on the key."""
    index = load_cache_index()
    with cache_lock:
        cache_entry = index.get(key)
        if cache_entry:
            cache_path = cache_entry.get('cache_path')
            if cache_path and os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # Update last access time
                    cache_entry['last_access_time'] = time.time()
                    index.move_to_end(key)  # Move to end to reflect recent access
                    save_cache_index(index)
                    logger.debug(f"Loaded cached response for key: {key}")
                    return data
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decoding failed for cached response {key}: {e}")
                    sentry_sdk.capture_exception(e)
                except OSError as e:
                    logger.error(f"OS error while loading cached response for key {key}: {e}")
                    sentry_sdk.capture_exception(e)
            else:
                logger.warning(f"Cache file does not exist for key: {key}")
                # Remove invalid cache entry
                del index[key]
                save_cache_index(index)
        else:
            logger.debug(f"No cached response found for key: {key}")
    return {}

def clear_cache(index: OrderedDict) -> None:
    """Evict least recently used cache entries if cache exceeds size limit."""
    total_size = 0
    with cache_lock:
        for key, entry in index.items():
            cache_path = entry.get('cache_path')
            if cache_path and os.path.exists(cache_path):
                try:
                    file_size = os.path.getsize(cache_path)
                    total_size += file_size
                except OSError as e:
                    logger.error(f"Error getting size for cache file {cache_path}: {e}")
                    sentry_sdk.capture_exception(e)
                    continue
        total_size_mb = total_size / (1024 * 1024)
        if total_size_mb > CACHE_MAX_SIZE_MB:
            logger.info("Cache size exceeded limit. Starting eviction process.")
            while total_size_mb > CACHE_MAX_SIZE_MB and index:
                # Pop the least recently used item
                key, entry = index.popitem(last=False)
                cache_path = entry.get('cache_path')
                if cache_path and os.path.exists(cache_path):
                    try:
                        file_size = os.path.getsize(cache_path)
                        os.remove(cache_path)
                        total_size -= file_size
                        total_size_mb = total_size / (1024 * 1024)
                        logger.debug(f"Removed cache file {cache_path} for key {key}")
                    except OSError as e:
                        logger.error(f"Error removing cache file {cache_path}: {e}")
                        sentry_sdk.capture_exception(e)
                else:
                    logger.debug(f"Cache file {cache_path} does not exist.")
            save_cache_index(index)
            logger.info("Cache eviction completed.")
        else:
            logger.debug(f"Cache size within limit: {total_size_mb:.2f} MB")
```

---

## File: cloned_repo/monitoring.py

### Summary

Found 0 classes and 2 functions | Total lines of code: 37 | Average function complexity: 2.00 | Maximum function complexity: 2 | Documentation coverage: 50.0%

### Changelog

- 2024-11-09T05:43:07.190988: Started code analysis
- 2024-11-09T05:43:07.191499: Analyzed function: initialize_sentry
- 2024-11-09T05:43:07.191970: Analyzed function: capture_exception
- 2024-11-09T05:43:07.192067: Completed code analysis

### Functions

#### Function: initialize_sentry

Error: Documentation generation failed.

```python
def initialize_sentry():
    try:
        sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=1.0)
        logger.info("Sentry initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}")
        raise
```

##### Return Type

Any

##### Complexity Metrics

- Cyclomatic Complexity: 2
- Cognitive Complexity: 2
- Is Async: No
- Is Generator: No
- Is Recursive: No

#### Function: capture_exception

Error: Documentation generation failed.

```python
def capture_exception(exception: Exception):
    """
    Capture and report an exception to Sentry.

    Args:
        exception (Exception): The exception to capture.
    """
    try:
        sentry_sdk.capture_exception(exception)
    except Exception as e:
        logger.error(f"Failed to capture exception: {e}")
```

##### Parameters

- **exception**: Exception (typed)

##### Return Type

Any

##### Complexity Metrics

- Cyclomatic Complexity: 2
- Cognitive Complexity: 2
- Is Async: No
- Is Generator: No
- Is Recursive: No

### Source Code

```python
import sentry_sdk
import os
from core.logger import LoggerSetup
from dotenv import load_dotenv

# Initialize a logger specifically for this module
logger = LoggerSetup.get_logger("monitoring")

# Load environment variables from .env file
load_dotenv()

# Load environment variables and validate
sentry_dsn = os.getenv("SENTRY_DSN")

if not sentry_dsn:
    logger.error("SENTRY_DSN is not set.")
    raise ValueError("SENTRY_DSN is not set.")

def initialize_sentry():
    try:
        sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=1.0)
        logger.info("Sentry initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}")
        raise

def capture_exception(exception: Exception):
    """
    Capture and report an exception to Sentry.

    Args:
        exception (Exception): The exception to capture.
    """
    try:
        sentry_sdk.capture_exception(exception)
    except Exception as e:
        logger.error(f"Failed to capture exception: {e}")
```

---

## File: cloned_repo/main.py

### Summary

Found 0 classes and 6 functions | Total lines of code: 198 | Average function complexity: 6.50 | Maximum function complexity: 14 | Documentation coverage: 0.0%

### Changelog

- 2024-11-09T05:43:07.143880: Started code analysis
- 2024-11-09T05:43:07.144852: Analyzed function: validate_repo_url
- 2024-11-09T05:43:07.146173: Analyzed function: process_files_concurrently
- 2024-11-09T05:43:07.148449: Analyzed function: analyze_functions_concurrently
- 2024-11-09T05:43:07.151997: Analyzed function: main
- 2024-11-09T05:43:07.152423: Analyzed function: process_with_semaphore
- 2024-11-09T05:43:07.152837: Analyzed function: analyze_with_semaphore
- 2024-11-09T05:43:07.153655: Completed code analysis

### Functions

#### Function: validate_repo_url

Error: Documentation generation failed.

```python
def validate_repo_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.netloc not in ['github.com', 'www.github.com']:
            logger.debug("Invalid hostname: %s", parsed.netloc)
            return False
        path_parts = [p for p in parsed.path.split('/') if p]
        if len(path_parts) < 2:
            logger.debug("Invalid path format: %s", parsed.path)
            return False
        logger.info("Valid GitHub URL: %s", url)
        return True
    except ValueError as e:
        logger.error("URL validation error: %s", str(e))
        return False
```

##### Parameters

- **url**: str (typed)

##### Return Type

bool (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 4
- Cognitive Complexity: 10
- Is Async: No
- Is Generator: No
- Is Recursive: No

#### Function: process_files_concurrently

Error: Documentation generation failed.

```python
async def process_files_concurrently(files_list: List[str], service: str) -> Dict[str, Dict[str, Any]]:
    logger.info("Starting to process %d files", len(files_list))
    semaphore = asyncio.Semaphore(10)

    async def process_with_semaphore(filepath):
        async with semaphore:
            return await process_file(filepath, service)

    tasks = [process_with_semaphore(filepath) for filepath in files_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    processed_results = {}

    for filepath, result in zip(files_list, results):
        if isinstance(result, Exception):
            logger.error("Error processing file %s: %s", filepath, result)
            sentry_sdk.capture_exception(result)
            continue
        if result and ('classes' in result or 'functions' in result):
            processed_results[filepath] = result
        else:
            logger.warning("No valid data extracted for %s", filepath)

    logger.info("Completed processing files.")
    return processed_results
```

##### Parameters

- **files_list**: List[str] (typed)
- **service**: str (typed)

##### Return Type

Dict[Any] (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 6
- Cognitive Complexity: 7
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: analyze_functions_concurrently

Error: Documentation generation failed.

```python
async def analyze_functions_concurrently(results: Dict[str, Dict[str, Any]], service: str) -> None:
    logger.info("Starting function analysis using %s service", service)
    semaphore = asyncio.Semaphore(5)

    async def analyze_with_semaphore(func, service):
        async with semaphore:
            return await analyze_function_with_openai(func, service)

    tasks = []

    for analysis in results.values():
        functions = analysis.get("functions", [])
        for func in functions:
            if isinstance(func, dict) and func.get("name"):
                tasks.append(analyze_with_semaphore(func, service))

    if not tasks:
        logger.warning("No functions found to analyze")
        return

    analyzed_results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in analyzed_results:
        if isinstance(result, Exception):
            logger.error("Error analyzing function: %s", result)
            sentry_sdk.capture_exception(result)
            continue
        if isinstance(result, dict):
            func_name = result.get("name")
            if func_name:
                for analysis in results.values():
                    for func in analysis.get("functions", []):
                        if isinstance(func, dict) and func.get("name") == func_name:
                            func.update(result)

    logger.info("Completed function analysis.")
```

##### Parameters

- **results**: Dict[Any] (typed)
- **service**: str (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 14
- Cognitive Complexity: 33
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: main

Error: Documentation generation failed.

```python
async def main():
    parser = argparse.ArgumentParser(description="Analyze code and generate documentation.")
    parser.add_argument("input_path", help="Path to the input directory or repository URL")
    parser.add_argument("output_path", help="Path to the output directory for markdown files")
    parser.add_argument("--service", choices=["azure", "openai"], required=True, help="AI service to use")
    args = parser.parse_args()
    repo_dir = None
    summary_data = {
        'files_processed': 0,
        'errors_encountered': 0,
        'start_time': datetime.now(),
        'end_time': None
    }
    try:
        # Initialize monitoring (Sentry)
        initialize_sentry()

        # Load environment variables and validate
        openai_api_key = os.getenv("OPENAI_API_KEY")
        azure_api_key = os.getenv("AZURE_API_KEY")
        azure_endpoint = os.getenv("AZURE_ENDPOINT")
        sentry_dsn = os.getenv("SENTRY_DSN")

        if not openai_api_key:
            logger.error("OPENAI_API_KEY is not set.")
            raise ValueError("OPENAI_API_KEY is not set.")
        if not azure_api_key:
            logger.error("AZURE_API_KEY is not set.")
            raise ValueError("AZURE_API_KEY is not set.")
        if not azure_endpoint:
            logger.error("AZURE_ENDPOINT is not set.")
            raise ValueError("AZURE_ENDPOINT is not set.")
        if not sentry_dsn:
            logger.error("SENTRY_DSN is not set.")
            raise ValueError("SENTRY_DSN is not set.")

        # Initialize cache
        initialize_cache()

        input_path = args.input_path
        output_path = args.output_path

        if input_path.startswith(('http://', 'https://')):
            if not validate_repo_url(input_path):
                logger.error("Invalid GitHub repository URL: %s", input_path)
                sys.exit(1)
            repo_dir = 'cloned_repo'
            await clone_repo(input_path, repo_dir)
            input_path = repo_dir

        exclude_patterns = load_gitignore_patterns(input_path)
        python_files = get_all_files(input_path, exclude_patterns)

        if not python_files:
            logger.error("No Python files found to analyze")
            sys.exit(1)

        results = await process_files_concurrently(python_files, args.service)
        summary_data['files_processed'] = len(results)

        if not results:
            logger.error("No valid results from file processing")
            sys.exit(1)

        await analyze_functions_concurrently(results, args.service)
        await write_analysis_to_markdown(results, output_path)
        logger.info("Analysis complete. Documentation written to %s", output_path)
    except Exception as e:
        summary_data['errors_encountered'] += 1
        logger.error("Error during execution: %s", str(e))
        sentry_sdk.capture_exception(e)
        sys.exit(1)
    finally:
        summary_data['end_time'] = datetime.now()
        if repo_dir and os.path.exists(repo_dir):
            try:
                shutil.rmtree(repo_dir)
                logger.info("Cleaned up temporary repository files")
            except OSError as e:
                logger.error("Error cleaning up repository: %s", str(e))
        logger.info(
            "Summary: Files processed: %d, Errors: %d, Start: %s, End: %s, Duration: %s",
            summary_data['files_processed'],
            summary_data['errors_encountered'],
            summary_data['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
            summary_data['end_time'].strftime('%Y-%m-%d %H:%M:%S'),
            str(summary_data['end_time'] - summary_data['start_time'])
        )
```

##### Return Type

Any

##### Complexity Metrics

- Cyclomatic Complexity: 13
- Cognitive Complexity: 24
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: process_with_semaphore

Error: Documentation generation failed.

```python
async def process_with_semaphore(filepath):
        async with semaphore:
            return await process_file(filepath, service)
```

##### Parameters

- **filepath**: Any

##### Return Type

Any

##### Complexity Metrics

- Cyclomatic Complexity: 1
- Cognitive Complexity: 0
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

#### Function: analyze_with_semaphore

Error: Documentation generation failed.

```python
async def analyze_with_semaphore(func, service):
        async with semaphore:
            return await analyze_function_with_openai(func, service)
```

##### Parameters

- **func**: Any
- **service**: Any

##### Return Type

Any

##### Complexity Metrics

- Cyclomatic Complexity: 1
- Cognitive Complexity: 0
- Is Async: Yes
- Is Generator: No
- Is Recursive: No

### Source Code

```python
import argparse
import asyncio
import os
import sys
import shutil
from datetime import datetime
from typing import List, Dict, Any
from urllib.parse import urlparse
from dotenv import load_dotenv

from files import (
    clone_repo,
    load_gitignore_patterns,
    get_all_files,
    process_file
)
from docs import write_analysis_to_markdown
from api_interaction import analyze_function_with_openai
from monitoring import initialize_sentry
from core.logger import LoggerSetup
from cache import initialize_cache
import sentry_sdk

# Initialize logger for the main module
logger = LoggerSetup.get_logger("main")

# Load environment variables from .env file
load_dotenv()

def validate_repo_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.netloc not in ['github.com', 'www.github.com']:
            logger.debug("Invalid hostname: %s", parsed.netloc)
            return False
        path_parts = [p for p in parsed.path.split('/') if p]
        if len(path_parts) < 2:
            logger.debug("Invalid path format: %s", parsed.path)
            return False
        logger.info("Valid GitHub URL: %s", url)
        return True
    except ValueError as e:
        logger.error("URL validation error: %s", str(e))
        return False

async def process_files_concurrently(files_list: List[str], service: str) -> Dict[str, Dict[str, Any]]:
    logger.info("Starting to process %d files", len(files_list))
    semaphore = asyncio.Semaphore(10)

    async def process_with_semaphore(filepath):
        async with semaphore:
            return await process_file(filepath, service)

    tasks = [process_with_semaphore(filepath) for filepath in files_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    processed_results = {}

    for filepath, result in zip(files_list, results):
        if isinstance(result, Exception):
            logger.error("Error processing file %s: %s", filepath, result)
            sentry_sdk.capture_exception(result)
            continue
        if result and ('classes' in result or 'functions' in result):
            processed_results[filepath] = result
        else:
            logger.warning("No valid data extracted for %s", filepath)

    logger.info("Completed processing files.")
    return processed_results

async def analyze_functions_concurrently(results: Dict[str, Dict[str, Any]], service: str) -> None:
    logger.info("Starting function analysis using %s service", service)
    semaphore = asyncio.Semaphore(5)

    async def analyze_with_semaphore(func, service):
        async with semaphore:
            return await analyze_function_with_openai(func, service)

    tasks = []

    for analysis in results.values():
        functions = analysis.get("functions", [])
        for func in functions:
            if isinstance(func, dict) and func.get("name"):
                tasks.append(analyze_with_semaphore(func, service))

    if not tasks:
        logger.warning("No functions found to analyze")
        return

    analyzed_results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in analyzed_results:
        if isinstance(result, Exception):
            logger.error("Error analyzing function: %s", result)
            sentry_sdk.capture_exception(result)
            continue
        if isinstance(result, dict):
            func_name = result.get("name")
            if func_name:
                for analysis in results.values():
                    for func in analysis.get("functions", []):
                        if isinstance(func, dict) and func.get("name") == func_name:
                            func.update(result)

    logger.info("Completed function analysis.")

async def main():
    parser = argparse.ArgumentParser(description="Analyze code and generate documentation.")
    parser.add_argument("input_path", help="Path to the input directory or repository URL")
    parser.add_argument("output_path", help="Path to the output directory for markdown files")
    parser.add_argument("--service", choices=["azure", "openai"], required=True, help="AI service to use")
    args = parser.parse_args()
    repo_dir = None
    summary_data = {
        'files_processed': 0,
        'errors_encountered': 0,
        'start_time': datetime.now(),
        'end_time': None
    }
    try:
        # Initialize monitoring (Sentry)
        initialize_sentry()

        # Load environment variables and validate
        openai_api_key = os.getenv("OPENAI_API_KEY")
        azure_api_key = os.getenv("AZURE_API_KEY")
        azure_endpoint = os.getenv("AZURE_ENDPOINT")
        sentry_dsn = os.getenv("SENTRY_DSN")

        if not openai_api_key:
            logger.error("OPENAI_API_KEY is not set.")
            raise ValueError("OPENAI_API_KEY is not set.")
        if not azure_api_key:
            logger.error("AZURE_API_KEY is not set.")
            raise ValueError("AZURE_API_KEY is not set.")
        if not azure_endpoint:
            logger.error("AZURE_ENDPOINT is not set.")
            raise ValueError("AZURE_ENDPOINT is not set.")
        if not sentry_dsn:
            logger.error("SENTRY_DSN is not set.")
            raise ValueError("SENTRY_DSN is not set.")

        # Initialize cache
        initialize_cache()

        input_path = args.input_path
        output_path = args.output_path

        if input_path.startswith(('http://', 'https://')):
            if not validate_repo_url(input_path):
                logger.error("Invalid GitHub repository URL: %s", input_path)
                sys.exit(1)
            repo_dir = 'cloned_repo'
            await clone_repo(input_path, repo_dir)
            input_path = repo_dir

        exclude_patterns = load_gitignore_patterns(input_path)
        python_files = get_all_files(input_path, exclude_patterns)

        if not python_files:
            logger.error("No Python files found to analyze")
            sys.exit(1)

        results = await process_files_concurrently(python_files, args.service)
        summary_data['files_processed'] = len(results)

        if not results:
            logger.error("No valid results from file processing")
            sys.exit(1)

        await analyze_functions_concurrently(results, args.service)
        await write_analysis_to_markdown(results, output_path)
        logger.info("Analysis complete. Documentation written to %s", output_path)
    except Exception as e:
        summary_data['errors_encountered'] += 1
        logger.error("Error during execution: %s", str(e))
        sentry_sdk.capture_exception(e)
        sys.exit(1)
    finally:
        summary_data['end_time'] = datetime.now()
        if repo_dir and os.path.exists(repo_dir):
            try:
                shutil.rmtree(repo_dir)
                logger.info("Cleaned up temporary repository files")
            except OSError as e:
                logger.error("Error cleaning up repository: %s", str(e))
        logger.info(
            "Summary: Files processed: %d, Errors: %d, Start: %s, End: %s, Duration: %s",
            summary_data['files_processed'],
            summary_data['errors_encountered'],
            summary_data['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
            summary_data['end_time'].strftime('%Y-%m-%d %H:%M:%S'),
            str(summary_data['end_time'] - summary_data['start_time'])
        )

if __name__ == "__main__":
    asyncio.run(main())
```

---

## File: cloned_repo/extract/functions.py

### Summary

Found 1 classes and 0 functions | Total lines of code: 174 | Documentation coverage: 0.0%

### Changelog

- 2024-11-09T05:43:07.074044: Started code analysis
- 2024-11-09T05:43:07.092059: Analyzed class: FunctionExtractor
- 2024-11-09T05:43:07.092801: Completed code analysis

### Classes

#### Class: FunctionExtractor

##### Methods

###### __init__

```python
def __init__(self, node: ast.AST, content: str):
        super().__init__(node, content)
        self.metrics = CodeMetrics()
```

Parameters:
- self: Any
- node: ast.AST (typed)
- content: str (typed)

###### extract_details

```python
def extract_details(self) -> Dict[str, Any]:
        try:
            # Calculate all metrics first
            complexity_score = self.calculate_complexity()
            cognitive_score = self.calculate_cognitive_complexity()
            halstead_metrics = self.calculate_halstead_metrics()

            details = {
                "name": self.node.name,
                "docstring": self.get_docstring(),
                "params": self.extract_parameters(),
                "returns": self._extract_return_annotation(),
                "complexity_score": complexity_score,
                "cognitive_complexity": cognitive_score,
                "halstead_metrics": halstead_metrics,
                "line_number": self.node.lineno,
                "end_line_number": self.node.end_lineno,
                "code": self.get_source_segment(self.node),
                "is_async": self.is_async(),
                "is_generator": self.is_generator(),
                "is_recursive": self.is_recursive(),
                "summary": self._generate_summary(complexity_score, cognitive_score, halstead_metrics),
                "changelog": []
            }
            return details
        except Exception as e:
            logger.error(f"Error extracting function details: {e}")
            return self._get_empty_details()
```

Parameters:
- self: Any

###### extract_parameters

```python
def extract_parameters(self) -> List[Dict[str, Any]]:
        params = []
        try:
            for param in self.node.args.args:
                param_info = {
                    "name": param.arg,
                    "type": get_annotation(param.annotation),
                    "has_type_hint": param.annotation is not None
                }
                params.append(param_info)
        except Exception as e:
            logger.error(f"Error extracting parameters: {e}")
        return params
```

Parameters:
- self: Any

###### calculate_complexity

Calculate cyclomatic complexity.

```python
def calculate_complexity(self) -> int:
        """Calculate cyclomatic complexity."""
        return self.metrics.calculate_complexity(self.node)
```

Parameters:
- self: Any

###### calculate_cognitive_complexity

Calculate cognitive complexity.

```python
def calculate_cognitive_complexity(self) -> int:
        """Calculate cognitive complexity."""
        return self.metrics.calculate_cognitive_complexity(self.node)
```

Parameters:
- self: Any

###### calculate_halstead_metrics

Calculate Halstead metrics.

```python
def calculate_halstead_metrics(self) -> Dict[str, float]:
        """Calculate Halstead metrics."""
        return self.metrics.calculate_halstead_metrics(self.node)
```

Parameters:
- self: Any

###### _extract_return_annotation

Extract return type annotation.

```python
def _extract_return_annotation(self) -> Dict[str, Any]:
        """Extract return type annotation."""
        try:
            return {
                "type": get_annotation(self.node.returns),
                "has_type_hint": self.node.returns is not None
            }
        except Exception as e:
            logger.error(f"Error extracting return annotation: {e}")
            return {"type": "Any", "has_type_hint": False}
```

Parameters:
- self: Any

###### is_async

Check if the function is async.

```python
def is_async(self) -> bool:
        """Check if the function is async."""
        return isinstance(self.node, ast.AsyncFunctionDef)
```

Parameters:
- self: Any

###### is_generator

Check if the function is a generator.

```python
def is_generator(self) -> bool:
        """Check if the function is a generator."""
        try:
            for node in ast.walk(self.node):
                if isinstance(node, ast.Yield) or isinstance(node, ast.YieldFrom):
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking generator status: {e}")
            return False
```

Parameters:
- self: Any

###### is_recursive

Check if the function is recursive.

```python
def is_recursive(self) -> bool:
        """Check if the function is recursive."""
        try:
            function_name = self.node.name
            for node in ast.walk(self.node):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == function_name:
                        return True
            return False
        except Exception as e:
            logger.error(f"Error checking recursive status: {e}")
            return False
```

Parameters:
- self: Any

###### _generate_summary

Generate a comprehensive summary of the function.

```python
def _generate_summary(self, complexity: int, cognitive: int, halstead: Dict[str, float]) -> str:
        """Generate a comprehensive summary of the function."""
        parts = []
        try:
            # Basic function characteristics
            if self.node.returns:
                parts.append(f"Returns: {get_annotation(self.node.returns)}")
            
            if self.is_generator():
                parts.append("Generator function")
            
            if self.is_async():
                parts.append("Async function")
            
            if self.is_recursive():
                parts.append("Recursive function")
            
            # Complexity metrics
            parts.append(f"Cyclomatic Complexity: {complexity}")
            parts.append(f"Cognitive Complexity: {cognitive}")
            
            # Halstead metrics summary
            if halstead.get("program_volume", 0) > 0:
                parts.append(f"Volume: {halstead['program_volume']:.2f}")
            if halstead.get("difficulty", 0) > 0:
                parts.append(f"Difficulty: {halstead['difficulty']:.2f}")
            
            # Quality assessment
            if complexity > 10:
                parts.append("⚠️ High cyclomatic complexity")
            if cognitive > 15:
                parts.append("⚠️ High cognitive complexity")
            if halstead.get("difficulty", 0) > 20:
                parts.append("⚠️ High difficulty score")

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            parts.append("Error generating complete summary")
        
        return " | ".join(parts)
```

Parameters:
- self: Any
- complexity: int (typed)
- cognitive: int (typed)
- halstead: Dict[Any] (typed)

###### _get_empty_details

Return empty details structure matching schema.

```python
def _get_empty_details(self) -> Dict[str, Any]:
        """Return empty details structure matching schema."""
        return {
            "name": "",
            "docstring": "",
            "params": [],
            "returns": {"type": "Any", "has_type_hint": False},
            "complexity_score": 0,
            "cognitive_complexity": 0,
            "halstead_metrics": {
                "program_length": 0,
                "vocabulary_size": 0,
                "program_volume": 0,
                "difficulty": 0,
                "effort": 0
            },
            "line_number": 0,
            "end_line_number": 0,
            "code": "",
            "is_async": False,
            "is_generator": False,
            "is_recursive": False,
            "summary": "",
            "changelog": []
        }
```

Parameters:
- self: Any

##### Instance Variables

- **metrics** (line 13)

##### Base Classes

- BaseExtractor

### Source Code

```python
from typing import Dict, Any, List
import ast
from core.logger import LoggerSetup
from extract.base import BaseExtractor
from extract.utils import get_annotation
from metrics import CodeMetrics

logger = LoggerSetup.get_logger("extract.functions")

class FunctionExtractor(BaseExtractor):
    def __init__(self, node: ast.AST, content: str):
        super().__init__(node, content)
        self.metrics = CodeMetrics()

    def extract_details(self) -> Dict[str, Any]:
        try:
            # Calculate all metrics first
            complexity_score = self.calculate_complexity()
            cognitive_score = self.calculate_cognitive_complexity()
            halstead_metrics = self.calculate_halstead_metrics()

            details = {
                "name": self.node.name,
                "docstring": self.get_docstring(),
                "params": self.extract_parameters(),
                "returns": self._extract_return_annotation(),
                "complexity_score": complexity_score,
                "cognitive_complexity": cognitive_score,
                "halstead_metrics": halstead_metrics,
                "line_number": self.node.lineno,
                "end_line_number": self.node.end_lineno,
                "code": self.get_source_segment(self.node),
                "is_async": self.is_async(),
                "is_generator": self.is_generator(),
                "is_recursive": self.is_recursive(),
                "summary": self._generate_summary(complexity_score, cognitive_score, halstead_metrics),
                "changelog": []
            }
            return details
        except Exception as e:
            logger.error(f"Error extracting function details: {e}")
            return self._get_empty_details()

    def extract_parameters(self) -> List[Dict[str, Any]]:
        params = []
        try:
            for param in self.node.args.args:
                param_info = {
                    "name": param.arg,
                    "type": get_annotation(param.annotation),
                    "has_type_hint": param.annotation is not None
                }
                params.append(param_info)
        except Exception as e:
            logger.error(f"Error extracting parameters: {e}")
        return params

    def calculate_complexity(self) -> int:
        """Calculate cyclomatic complexity."""
        return self.metrics.calculate_complexity(self.node)

    def calculate_cognitive_complexity(self) -> int:
        """Calculate cognitive complexity."""
        return self.metrics.calculate_cognitive_complexity(self.node)

    def calculate_halstead_metrics(self) -> Dict[str, float]:
        """Calculate Halstead metrics."""
        return self.metrics.calculate_halstead_metrics(self.node)

    def _extract_return_annotation(self) -> Dict[str, Any]:
        """Extract return type annotation."""
        try:
            return {
                "type": get_annotation(self.node.returns),
                "has_type_hint": self.node.returns is not None
            }
        except Exception as e:
            logger.error(f"Error extracting return annotation: {e}")
            return {"type": "Any", "has_type_hint": False}

    def is_async(self) -> bool:
        """Check if the function is async."""
        return isinstance(self.node, ast.AsyncFunctionDef)

    def is_generator(self) -> bool:
        """Check if the function is a generator."""
        try:
            for node in ast.walk(self.node):
                if isinstance(node, ast.Yield) or isinstance(node, ast.YieldFrom):
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking generator status: {e}")
            return False

    def is_recursive(self) -> bool:
        """Check if the function is recursive."""
        try:
            function_name = self.node.name
            for node in ast.walk(self.node):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == function_name:
                        return True
            return False
        except Exception as e:
            logger.error(f"Error checking recursive status: {e}")
            return False

    def _generate_summary(self, complexity: int, cognitive: int, halstead: Dict[str, float]) -> str:
        """Generate a comprehensive summary of the function."""
        parts = []
        try:
            # Basic function characteristics
            if self.node.returns:
                parts.append(f"Returns: {get_annotation(self.node.returns)}")
            
            if self.is_generator():
                parts.append("Generator function")
            
            if self.is_async():
                parts.append("Async function")
            
            if self.is_recursive():
                parts.append("Recursive function")
            
            # Complexity metrics
            parts.append(f"Cyclomatic Complexity: {complexity}")
            parts.append(f"Cognitive Complexity: {cognitive}")
            
            # Halstead metrics summary
            if halstead.get("program_volume", 0) > 0:
                parts.append(f"Volume: {halstead['program_volume']:.2f}")
            if halstead.get("difficulty", 0) > 0:
                parts.append(f"Difficulty: {halstead['difficulty']:.2f}")
            
            # Quality assessment
            if complexity > 10:
                parts.append("⚠️ High cyclomatic complexity")
            if cognitive > 15:
                parts.append("⚠️ High cognitive complexity")
            if halstead.get("difficulty", 0) > 20:
                parts.append("⚠️ High difficulty score")

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            parts.append("Error generating complete summary")
        
        return " | ".join(parts)

    def _get_empty_details(self) -> Dict[str, Any]:
        """Return empty details structure matching schema."""
        return {
            "name": "",
            "docstring": "",
            "params": [],
            "returns": {"type": "Any", "has_type_hint": False},
            "complexity_score": 0,
            "cognitive_complexity": 0,
            "halstead_metrics": {
                "program_length": 0,
                "vocabulary_size": 0,
                "program_volume": 0,
                "difficulty": 0,
                "effort": 0
            },
            "line_number": 0,
            "end_line_number": 0,
            "code": "",
            "is_async": False,
            "is_generator": False,
            "is_recursive": False,
            "summary": "",
            "changelog": []
        }
```

---

## File: cloned_repo/extract/utils.py

### Summary

Found 0 classes and 5 functions | Total lines of code: 138 | Average function complexity: 4.20 | Maximum function complexity: 11 | Documentation coverage: 100.0%

### Changelog

- 2024-11-09T05:43:07.171605: Started code analysis
- 2024-11-09T05:43:07.172157: Analyzed function: add_parent_info
- 2024-11-09T05:43:07.173689: Analyzed function: get_annotation
- 2024-11-09T05:43:07.174697: Analyzed function: _load_schema
- 2024-11-09T05:43:07.176093: Analyzed function: validate_schema
- 2024-11-09T05:43:07.176882: Analyzed function: format_validation_error
- 2024-11-09T05:43:07.177381: Completed code analysis

### Functions

#### Function: add_parent_info

Error: Documentation generation failed.

```python
def add_parent_info(tree: ast.AST) -> None:
    """
    Add parent information to each node in the AST.
    
    This function traverses the AST and adds a 'parent' attribute to each node,
    which is needed for correctly identifying top-level functions vs methods.
    
    Args:
        tree (ast.AST): The AST to process
    """
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent
```

##### Parameters

- **tree**: ast.AST (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 3
- Cognitive Complexity: 3
- Is Async: No
- Is Generator: No
- Is Recursive: No

#### Function: get_annotation

Error: Documentation generation failed.

```python
def get_annotation(node: Optional[ast.AST]) -> str:
    """
    Convert AST annotation to string representation.
    
    Args:
        node (Optional[ast.AST]): The AST node containing type annotation
        
    Returns:
        str: String representation of the type annotation
    """
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
            # Handle Union types written with | operator (Python 3.10+)
            if isinstance(node.op, ast.BitOr):
                left = get_annotation(node.left)
                right = get_annotation(node.right)
                return f"Union[{left}, {right}]"
        else:
            return "Any"
    except Exception as e:
        logger.error(f"Error processing type annotation: {e}")
        return "Any"
```

##### Parameters

- **node**: Optional[ast.AST] (typed)

##### Return Type

str (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 11
- Cognitive Complexity: 49
- Is Async: No
- Is Generator: No
- Is Recursive: Yes

#### Function: _load_schema

Error: Documentation generation failed.

```python
def _load_schema() -> Dict[str, Any]:
    """
    Load the JSON schema from file with caching.
    
    Returns:
        Dict[str, Any]: The loaded schema
        
    Raises:
        FileNotFoundError: If schema file is not found
        json.JSONDecodeError: If schema file is invalid JSON
    """
    if 'schema' not in _schema_cache:
        schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'function_schema.json')
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
```

##### Return Type

Dict[Any] (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 4
- Cognitive Complexity: 4
- Is Async: No
- Is Generator: No
- Is Recursive: No

#### Function: validate_schema

Error: Documentation generation failed.

```python
def validate_schema(data: Dict[str, Any]) -> None:
    """
    Validate extracted data against schema.
    
    Args:
        data (Dict[str, Any]): The data to validate
        
    Raises:
        jsonschema.ValidationError: If validation fails
        jsonschema.SchemaError: If schema is invalid
        FileNotFoundError: If schema file is not found
        json.JSONDecodeError: If schema file is invalid JSON
    """
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
```

##### Parameters

- **data**: Dict[Any] (typed)

##### Return Type

None (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 2
- Cognitive Complexity: 4
- Is Async: No
- Is Generator: No
- Is Recursive: No

#### Function: format_validation_error

Error: Documentation generation failed.

```python
def format_validation_error(error: jsonschema.ValidationError) -> str:
    """
    Format a validation error into a human-readable message.
    
    Args:
        error (jsonschema.ValidationError): The validation error
        
    Returns:
        str: Formatted error message
    """
    path = ' -> '.join(str(p) for p in error.path) if error.path else 'root'
    return (
        f"Validation error at {path}:\n"
        f"Message: {error.message}\n"
        f"Failed value: {error.instance}\n"
        f"Schema path: {' -> '.join(str(p) for p in error.schema_path)}"
    )
```

##### Parameters

- **error**: jsonschema.ValidationError (typed)

##### Return Type

str (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 1
- Cognitive Complexity: 0
- Is Async: No
- Is Generator: No
- Is Recursive: No

### Source Code

```python
import ast
import json
import os
from typing import Optional, Dict, Any
import jsonschema
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger("extract.utils")

# Cache for the schema to avoid repeated file reads
_schema_cache: Dict[str, Any] = {}

def add_parent_info(tree: ast.AST) -> None:
    """
    Add parent information to each node in the AST.
    
    This function traverses the AST and adds a 'parent' attribute to each node,
    which is needed for correctly identifying top-level functions vs methods.
    
    Args:
        tree (ast.AST): The AST to process
    """
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent

def get_annotation(node: Optional[ast.AST]) -> str:
    """
    Convert AST annotation to string representation.
    
    Args:
        node (Optional[ast.AST]): The AST node containing type annotation
        
    Returns:
        str: String representation of the type annotation
    """
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
            # Handle Union types written with | operator (Python 3.10+)
            if isinstance(node.op, ast.BitOr):
                left = get_annotation(node.left)
                right = get_annotation(node.right)
                return f"Union[{left}, {right}]"
        else:
            return "Any"
    except Exception as e:
        logger.error(f"Error processing type annotation: {e}")
        return "Any"

def _load_schema() -> Dict[str, Any]:
    """
    Load the JSON schema from file with caching.
    
    Returns:
        Dict[str, Any]: The loaded schema
        
    Raises:
        FileNotFoundError: If schema file is not found
        json.JSONDecodeError: If schema file is invalid JSON
    """
    if 'schema' not in _schema_cache:
        schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'function_schema.json')
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
    """
    Validate extracted data against schema.
    
    Args:
        data (Dict[str, Any]): The data to validate
        
    Raises:
        jsonschema.ValidationError: If validation fails
        jsonschema.SchemaError: If schema is invalid
        FileNotFoundError: If schema file is not found
        json.JSONDecodeError: If schema file is invalid JSON
    """
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
    """
    Format a validation error into a human-readable message.
    
    Args:
        error (jsonschema.ValidationError): The validation error
        
    Returns:
        str: Formatted error message
    """
    path = ' -> '.join(str(p) for p in error.path) if error.path else 'root'
    return (
        f"Validation error at {path}:\n"
        f"Message: {error.message}\n"
        f"Failed value: {error.instance}\n"
        f"Schema path: {' -> '.join(str(p) for p in error.schema_path)}"
    )
```

---

## File: cloned_repo/extract/base.py

### Summary

Found 1 classes and 0 functions | Total lines of code: 57 | Documentation coverage: 100.0%

### Changelog

- 2024-11-09T05:43:07.202849: Started code analysis
- 2024-11-09T05:43:07.208282: Analyzed class: BaseExtractor
- 2024-11-09T05:43:07.208484: Completed code analysis

### Classes

#### Class: BaseExtractor

Base class for AST extractors.

##### Methods

###### __init__

```python
def __init__(self, node: ast.AST, content: str) -> None:
        if node is None:
            raise ValueError("AST node cannot be None")
        if not content:
            raise ValueError("Content cannot be empty")
        self.node = node
        self.content = content
        logger.debug(f"Initialized {self.__class__.__name__} for node type {type(node).__name__}")
```

Parameters:
- self: Any
- node: ast.AST (typed)
- content: str (typed)

###### extract_details

Extract details from the AST node.

```python
def extract_details(self) -> Dict[str, Any]:
        """Extract details from the AST node."""
        pass
```

Parameters:
- self: Any

###### get_docstring

Extract docstring from node.

```python
def get_docstring(self) -> str:
        """Extract docstring from node."""
        try:
            return ast.get_docstring(self.node) or ""
        except Exception as e:
            logger.error(f"Error extracting docstring: {e}")
            return ""
```

Parameters:
- self: Any

###### get_source_segment

Get source code segment for a node.

```python
def get_source_segment(self, node: ast.AST) -> str:
        """Get source code segment for a node."""
        try:
            return ast.get_source_segment(self.content, node) or ""
        except Exception as e:
            logger.error(f"Error getting source segment: {e}")
            return ""
```

Parameters:
- self: Any
- node: ast.AST (typed)

###### _get_empty_details

Return empty details structure matching schema.

```python
def _get_empty_details(self) -> Dict[str, Any]:
        """Return empty details structure matching schema."""
        return {
            "name": "",
            "docstring": "",
            "params": [],
            "returns": {"type": "None", "has_type_hint": False},
            "complexity_score": 0,
            "line_number": 0,
            "end_line_number": 0,
            "code": "",
            "is_async": False,
            "is_generator": False,
            "is_recursive": False,
            "summary": "",
            "changelog": ""
        }
```

Parameters:
- self: Any

##### Instance Variables

- **node** (line 16)
- **content** (line 17)
- **__class__** (line 18)

##### Base Classes

- ABC

### Source Code

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
import ast
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger("extract.base")

class BaseExtractor(ABC):
    """Base class for AST extractors."""
    
    def __init__(self, node: ast.AST, content: str) -> None:
        if node is None:
            raise ValueError("AST node cannot be None")
        if not content:
            raise ValueError("Content cannot be empty")
        self.node = node
        self.content = content
        logger.debug(f"Initialized {self.__class__.__name__} for node type {type(node).__name__}")

    @abstractmethod
    def extract_details(self) -> Dict[str, Any]:
        """Extract details from the AST node."""
        pass

    def get_docstring(self) -> str:
        """Extract docstring from node."""
        try:
            return ast.get_docstring(self.node) or ""
        except Exception as e:
            logger.error(f"Error extracting docstring: {e}")
            return ""

    def get_source_segment(self, node: ast.AST) -> str:
        """Get source code segment for a node."""
        try:
            return ast.get_source_segment(self.content, node) or ""
        except Exception as e:
            logger.error(f"Error getting source segment: {e}")
            return ""

    def _get_empty_details(self) -> Dict[str, Any]:
        """Return empty details structure matching schema."""
        return {
            "name": "",
            "docstring": "",
            "params": [],
            "returns": {"type": "None", "has_type_hint": False},
            "complexity_score": 0,
            "line_number": 0,
            "end_line_number": 0,
            "code": "",
            "is_async": False,
            "is_generator": False,
            "is_recursive": False,
            "summary": "",
            "changelog": ""
        }
```

---

## File: cloned_repo/extract/__init__.py

### Summary

Found 0 classes and 0 functions | Total lines of code: 12

### Changelog

- 2024-11-09T05:43:07.223005: Started code analysis
- 2024-11-09T05:43:07.223038: Completed code analysis

### Source Code

```python
from .base import BaseExtractor
from .classes import ClassExtractor
from .functions import FunctionExtractor
from .utils import add_parent_info, get_annotation

__all__ = [
    "BaseExtractor",
    "ClassExtractor",
    "FunctionExtractor",
    "add_parent_info",
    "get_annotation"
]
```

---

## File: cloned_repo/extract/classes.py

### Summary

Found 1 classes and 0 functions | Total lines of code: 114 | Documentation coverage: 0.0%

### Changelog

- 2024-11-09T05:43:07.233705: Started code analysis
- 2024-11-09T05:43:07.245185: Analyzed class: ClassExtractor
- 2024-11-09T05:43:07.245767: Completed code analysis

### Classes

#### Class: ClassExtractor

##### Methods

###### extract_details

```python
def extract_details(self) -> Dict[str, Any]:
        try:
            details = {
                "name": self.node.name,
                "docstring": self.get_docstring(),
                "methods": self.extract_methods(),
                "attributes": self.extract_attributes(),
                "instance_variables": self.extract_instance_variables(),
                "base_classes": self.extract_base_classes(),
                "summary": self._generate_summary(),
                "changelog": []
            }
            return details
        except Exception as e:
            logger.error(f"Error extracting class details: {e}")
            return self._get_empty_details()
```

Parameters:
- self: Any

###### extract_methods

```python
def extract_methods(self) -> List[Dict[str, Any]]:
        methods = []
        try:
            for node in self.node.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    extractor = FunctionExtractor(node, self.content)
                    method_details = extractor.extract_details()
                    methods.append(method_details)
        except Exception as e:
            logger.error(f"Error extracting methods: {e}")
        return methods
```

Parameters:
- self: Any

###### extract_attributes

```python
def extract_attributes(self) -> List[Dict[str, Any]]:
        attributes = []
        try:
            for node in self.node.body:
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    attributes.append({
                        "name": node.target.id,
                        "type": get_annotation(node.annotation),  # Use the imported function
                        "line_number": node.lineno
                    })
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            attributes.append({
                                "name": target.id,
                                "type": "Any",
                                "line_number": node.lineno
                            })
        except Exception as e:
            logger.error(f"Error extracting attributes: {e}")
        return attributes
```

Parameters:
- self: Any

###### extract_instance_variables

```python
def extract_instance_variables(self) -> List[Dict[str, Any]]:
        instance_vars = []
        try:
            for node in self.node.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
                    for sub_node in ast.walk(node):
                        if isinstance(sub_node, ast.Attribute) and isinstance(sub_node.value, ast.Name):
                            if sub_node.value.id == "self":
                                instance_vars.append({
                                    "name": sub_node.attr,
                                    "line_number": sub_node.lineno
                                })
        except Exception as e:
            logger.error(f"Error extracting instance variables: {e}")
        return instance_vars
```

Parameters:
- self: Any

###### extract_base_classes

```python
def extract_base_classes(self) -> List[str]:
        base_classes = []
        try:
            for base in self.node.bases:
                if isinstance(base, ast.Name):
                    base_classes.append(base.id)
                elif isinstance(base, ast.Attribute):
                    parts = []
                    node = base
                    while isinstance(node, ast.Attribute):
                        parts.append(node.attr)
                        node = node.value
                    if isinstance(node, ast.Name):
                        parts.append(node.id)
                        base_classes.append(".".join(reversed(parts)))
        except Exception as e:
            logger.error(f"Error extracting base classes: {e}")
        return base_classes
```

Parameters:
- self: Any

###### _generate_summary

```python
def _generate_summary(self) -> str:
        parts = []
        try:
            if self.node.bases:
                base_classes = self.extract_base_classes()
                parts.append(f"Inherits from: {', '.join(base_classes)}")
            
            method_count = len(self.extract_methods())
            attr_count = len(self.extract_attributes())
            instance_var_count = len(self.extract_instance_variables())
            
            parts.append(f"Methods: {method_count}")
            parts.append(f"Attributes: {attr_count}")
            parts.append(f"Instance Variables: {instance_var_count}")
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
        
        return " | ".join(parts)
```

Parameters:
- self: Any

##### Base Classes

- BaseExtractor

### Source Code

```python
from typing import Dict, Any, List
from core.logger import LoggerSetup
from extract.base import BaseExtractor
from extract.functions import FunctionExtractor
from extract.utils import get_annotation  # Import the get_annotation function
import ast

logger = LoggerSetup.get_logger("extract.classes")

class ClassExtractor(BaseExtractor):
    def extract_details(self) -> Dict[str, Any]:
        try:
            details = {
                "name": self.node.name,
                "docstring": self.get_docstring(),
                "methods": self.extract_methods(),
                "attributes": self.extract_attributes(),
                "instance_variables": self.extract_instance_variables(),
                "base_classes": self.extract_base_classes(),
                "summary": self._generate_summary(),
                "changelog": []
            }
            return details
        except Exception as e:
            logger.error(f"Error extracting class details: {e}")
            return self._get_empty_details()

    def extract_methods(self) -> List[Dict[str, Any]]:
        methods = []
        try:
            for node in self.node.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    extractor = FunctionExtractor(node, self.content)
                    method_details = extractor.extract_details()
                    methods.append(method_details)
        except Exception as e:
            logger.error(f"Error extracting methods: {e}")
        return methods

    def extract_attributes(self) -> List[Dict[str, Any]]:
        attributes = []
        try:
            for node in self.node.body:
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    attributes.append({
                        "name": node.target.id,
                        "type": get_annotation(node.annotation),  # Use the imported function
                        "line_number": node.lineno
                    })
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            attributes.append({
                                "name": target.id,
                                "type": "Any",
                                "line_number": node.lineno
                            })
        except Exception as e:
            logger.error(f"Error extracting attributes: {e}")
        return attributes

    def extract_instance_variables(self) -> List[Dict[str, Any]]:
        instance_vars = []
        try:
            for node in self.node.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
                    for sub_node in ast.walk(node):
                        if isinstance(sub_node, ast.Attribute) and isinstance(sub_node.value, ast.Name):
                            if sub_node.value.id == "self":
                                instance_vars.append({
                                    "name": sub_node.attr,
                                    "line_number": sub_node.lineno
                                })
        except Exception as e:
            logger.error(f"Error extracting instance variables: {e}")
        return instance_vars

    def extract_base_classes(self) -> List[str]:
        base_classes = []
        try:
            for base in self.node.bases:
                if isinstance(base, ast.Name):
                    base_classes.append(base.id)
                elif isinstance(base, ast.Attribute):
                    parts = []
                    node = base
                    while isinstance(node, ast.Attribute):
                        parts.append(node.attr)
                        node = node.value
                    if isinstance(node, ast.Name):
                        parts.append(node.id)
                        base_classes.append(".".join(reversed(parts)))
        except Exception as e:
            logger.error(f"Error extracting base classes: {e}")
        return base_classes

    def _generate_summary(self) -> str:
        parts = []
        try:
            if self.node.bases:
                base_classes = self.extract_base_classes()
                parts.append(f"Inherits from: {', '.join(base_classes)}")
            
            method_count = len(self.extract_methods())
            attr_count = len(self.extract_attributes())
            instance_var_count = len(self.extract_instance_variables())
            
            parts.append(f"Methods: {method_count}")
            parts.append(f"Attributes: {attr_count}")
            parts.append(f"Instance Variables: {instance_var_count}")
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
        
        return " | ".join(parts)
```

---

## File: cloned_repo/extract/code.py

### Summary

Found 0 classes and 1 functions | Total lines of code: 133 | Average function complexity: 10.00 | Maximum function complexity: 10 | Documentation coverage: 100.0%

### Changelog

- 2024-11-09T05:43:07.261717: Started code analysis
- 2024-11-09T05:43:07.266018: Analyzed function: extract_classes_and_functions_from_ast
- 2024-11-09T05:43:07.266588: Completed code analysis

### Functions

#### Function: extract_classes_and_functions_from_ast

Error: Documentation generation failed.

```python
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

        # Add initial changelog entry
        result["changelog"].append({
            "change": "Started code analysis",
            "timestamp": datetime.now().isoformat()
        })

        # Extract classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                try:
                    extractor = ClassExtractor(node, content)
                    class_info = extractor.extract_details()
                    result["classes"].append(class_info)
                    metrics.total_classes += 1
                    
                    # Add changelog entry for class
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
                if not isinstance(getattr(node, 'parent', None), ast.ClassDef):  # Only top-level functions
                    try:
                        extractor = FunctionExtractor(node, content)
                        func_info = extractor.extract_details()
                        result["functions"].append(func_info)
                        metrics.total_functions += 1
                        
                        # Add changelog entry for function
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

        # Calculate overall metrics
        metrics.total_lines = len(content.splitlines())
        
        # Generate comprehensive summary
        summary_parts = [
            f"Found {len(result['classes'])} classes and {len(result['functions'])} functions",
            f"Total lines of code: {metrics.total_lines}",
        ]

        # Add complexity information if available
        if result["functions"]:
            avg_complexity = sum(f.get("complexity_score", 0) for f in result["functions"]) / len(result["functions"])
            max_complexity = max((f.get("complexity_score", 0) for f in result["functions"]), default=0)
            summary_parts.extend([
                f"Average function complexity: {avg_complexity:.2f}",
                f"Maximum function complexity: {max_complexity}"
            ])

        # Add docstring coverage information
        functions_with_docs = sum(1 for f in result["functions"] if f.get("docstring"))
        classes_with_docs = sum(1 for c in result["classes"] if c.get("docstring"))
        total_items = len(result["functions"]) + len(result["classes"])
        if total_items > 0:
            doc_coverage = ((functions_with_docs + classes_with_docs) / total_items) * 100
            summary_parts.append(f"Documentation coverage: {doc_coverage:.1f}%")

        result["summary"] = " | ".join(summary_parts)
        
        # Add final changelog entry
        result["changelog"].append({
            "change": "Completed code analysis",
            "timestamp": datetime.now().isoformat()
        })
        
        # Validate against schema
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
```

##### Parameters

- **tree**: ast.AST (typed)
- **content**: str (typed)

##### Return Type

Dict[Any] (typed)

##### Complexity Metrics

- Cyclomatic Complexity: 10
- Cognitive Complexity: 26
- Is Async: No
- Is Generator: No
- Is Recursive: No

### Source Code

```python
# extract/code.py
import ast
from datetime import datetime
from typing import Dict, Any, List
from core.logger import LoggerSetup
from extract.base import BaseExtractor
from extract.utils import validate_schema
from extract.classes import ClassExtractor
from extract.functions import FunctionExtractor
from metrics import CodeMetrics

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

        # Add initial changelog entry
        result["changelog"].append({
            "change": "Started code analysis",
            "timestamp": datetime.now().isoformat()
        })

        # Extract classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                try:
                    extractor = ClassExtractor(node, content)
                    class_info = extractor.extract_details()
                    result["classes"].append(class_info)
                    metrics.total_classes += 1
                    
                    # Add changelog entry for class
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
                if not isinstance(getattr(node, 'parent', None), ast.ClassDef):  # Only top-level functions
                    try:
                        extractor = FunctionExtractor(node, content)
                        func_info = extractor.extract_details()
                        result["functions"].append(func_info)
                        metrics.total_functions += 1
                        
                        # Add changelog entry for function
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

        # Calculate overall metrics
        metrics.total_lines = len(content.splitlines())
        
        # Generate comprehensive summary
        summary_parts = [
            f"Found {len(result['classes'])} classes and {len(result['functions'])} functions",
            f"Total lines of code: {metrics.total_lines}",
        ]

        # Add complexity information if available
        if result["functions"]:
            avg_complexity = sum(f.get("complexity_score", 0) for f in result["functions"]) / len(result["functions"])
            max_complexity = max((f.get("complexity_score", 0) for f in result["functions"]), default=0)
            summary_parts.extend([
                f"Average function complexity: {avg_complexity:.2f}",
                f"Maximum function complexity: {max_complexity}"
            ])

        # Add docstring coverage information
        functions_with_docs = sum(1 for f in result["functions"] if f.get("docstring"))
        classes_with_docs = sum(1 for c in result["classes"] if c.get("docstring"))
        total_items = len(result["functions"]) + len(result["classes"])
        if total_items > 0:
            doc_coverage = ((functions_with_docs + classes_with_docs) / total_items) * 100
            summary_parts.append(f"Documentation coverage: {doc_coverage:.1f}%")

        result["summary"] = " | ".join(summary_parts)
        
        # Add final changelog entry
        result["changelog"].append({
            "change": "Completed code analysis",
            "timestamp": datetime.now().isoformat()
        })
        
        # Validate against schema
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
```

---

## File: cloned_repo/core/logger.py

### Summary

Found 1 classes and 0 functions | Total lines of code: 52 | Documentation coverage: 0.0%

### Changelog

- 2024-11-09T05:43:07.279155: Started code analysis
- 2024-11-09T05:43:07.283080: Analyzed class: LoggerSetup
- 2024-11-09T05:43:07.283243: Completed code analysis

### Classes

#### Class: LoggerSetup

##### Methods

###### get_logger

Get a logger for a specific module with optional console logging.

Args:
    module_name (str): The name of the module for which to set up the logger.
    console_logging (bool): If True, also log to console.

Returns:
    logging.Logger: Configured logger for the module.

```python
def get_logger(module_name: str, console_logging: bool = False) -> logging.Logger:
        """
        Get a logger for a specific module with optional console logging.

        Args:
            module_name (str): The name of the module for which to set up the logger.
            console_logging (bool): If True, also log to console.

        Returns:
            logging.Logger: Configured logger for the module.
        """
        logger = logging.getLogger(module_name)
        if not logger.handlers:  # Avoid adding handlers multiple times
            logger.setLevel(logging.DEBUG)
            LoggerSetup._add_file_handler(logger, module_name)
            if console_logging:
                LoggerSetup._add_console_handler(logger)
        return logger
```

Parameters:
- module_name: str (typed)
- console_logging: bool (typed)

###### _add_file_handler

Add a rotating file handler to the logger.

```python
def _add_file_handler(logger: logging.Logger, module_name: str) -> None:
        """
        Add a rotating file handler to the logger.
        """
        log_dir = os.path.join("logs", module_name)
        os.makedirs(log_dir, exist_ok=True)
        handler = RotatingFileHandler(
            os.path.join(log_dir, f"{module_name}.log"),
            maxBytes=10**6,  # 1 MB
            backupCount=5
        )
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
```

Parameters:
- logger: logging.Logger (typed)
- module_name: str (typed)

###### _add_console_handler

Add a console handler to the logger.

```python
def _add_console_handler(logger: logging.Logger) -> None:
        """
        Add a console handler to the logger.
        """
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(console_handler)
```

Parameters:
- logger: logging.Logger (typed)

### Source Code

```python
import logging
import os
from logging.handlers import RotatingFileHandler

class LoggerSetup:
    @staticmethod
    def get_logger(module_name: str, console_logging: bool = False) -> logging.Logger:
        """
        Get a logger for a specific module with optional console logging.

        Args:
            module_name (str): The name of the module for which to set up the logger.
            console_logging (bool): If True, also log to console.

        Returns:
            logging.Logger: Configured logger for the module.
        """
        logger = logging.getLogger(module_name)
        if not logger.handlers:  # Avoid adding handlers multiple times
            logger.setLevel(logging.DEBUG)
            LoggerSetup._add_file_handler(logger, module_name)
            if console_logging:
                LoggerSetup._add_console_handler(logger)
        return logger

    @staticmethod
    def _add_file_handler(logger: logging.Logger, module_name: str) -> None:
        """
        Add a rotating file handler to the logger.
        """
        log_dir = os.path.join("logs", module_name)
        os.makedirs(log_dir, exist_ok=True)
        handler = RotatingFileHandler(
            os.path.join(log_dir, f"{module_name}.log"),
            maxBytes=10**6,  # 1 MB
            backupCount=5
        )
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)

    @staticmethod
    def _add_console_handler(logger: logging.Logger) -> None:
        """
        Add a console handler to the logger.
        """
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(console_handler)
```

---

## File: cloned_repo/core/__init__.py

### Summary

Found 0 classes and 0 functions | Total lines of code: 4

### Changelog

- 2024-11-09T05:43:07.294548: Started code analysis
- 2024-11-09T05:43:07.294568: Completed code analysis

### Source Code

```python
# core/__init__.py
from .logger import LoggerSetup

__all__ = ['Settings', 'LoggerSetup']

```

---

