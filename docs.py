import os
from typing import Any, Dict, List
from logging_utils import setup_logger
import aiofiles

# Initialize logger for this module
logger = setup_logger("docs")

def escape_markdown(text: str) -> str:
    """Escape markdown special characters in a string."""
    markdown_special_chars = '\\`*_{}[]()#+-.!'
    for char in markdown_special_chars:
        text = text.replace(char, '\\' + char)
    return text

def create_complexity_indicator(complexity: int) -> str:
    """Create a visual indicator for code complexity."""
    if complexity <= 5:
        indicator = "🟢"
    elif complexity <= 10:
        indicator = "🟡"
    else:
        indicator = "🔴"
    logger.debug(f"Complexity {complexity} has indicator {indicator}")
    return indicator

async def write_analysis_to_markdown(results: Dict[str, Dict[str, Any]], output_dir: str, repo_dir: str) -> None:
    """Write analysis results to separate markdown files."""
    logger.info("Writing analysis results to directory: %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    index_file_path = os.path.join(output_dir, "README.md")

    try:
        async with aiofiles.open(index_file_path, 'w', encoding='utf-8') as index_file:
            if not results:
                logger.error("No analysis results to write")
                await index_file.write("# ⚠️ No Analysis Results\n\nNo valid code analysis data was found.")
                return

            await index_file.write("# 📊 Code Analysis Report\n\n")
            await index_file.write("## 📑 Table of Contents\n\n")

            valid_results = {k: v for k, v in results.items() if v and isinstance(v, dict)}
            logger.debug(f"Valid results count: {len(valid_results)}")

            if not valid_results:
                logger.warning("No valid analysis results after filtering")
                await index_file.write("\n## ⚠️ Warning\nNo valid analysis results found after filtering.")
                return

            for filepath in valid_results:
                rel_path = os.path.relpath(filepath, repo_dir)
                file_slug = rel_path.replace('/', '_').replace('.', '_').replace(' ', '_')
                md_filename = f"{file_slug}.md"
                await index_file.write(f"- [📄 {rel_path}]({md_filename})\n")
                analysis = valid_results[filepath]
                md_file_path = os.path.join(output_dir, md_filename)
                await write_individual_markdown(md_file_path, rel_path, analysis)

            await index_file.write(f"\n## 📈 Summary\n\n")
            await index_file.write(f"- Total files analyzed: {len(valid_results)}\n")

        logger.info("Successfully wrote analysis to markdown files.")

    except OSError as e:
        logger.error("Error writing markdown files: %s", str(e))
        raise

async def write_individual_markdown(md_file_path: str, rel_path: str, analysis: Dict[str, Any]) -> None:
    """Write an individual markdown file for a single file's analysis."""
    logger.info(f"Writing analysis for {rel_path} to {md_file_path}")
    try:
        async with aiofiles.open(md_file_path, 'w', encoding='utf-8') as md_file:
            await md_file.write(f"# 📄 {rel_path}\n\n")
            await write_module_overview(md_file, analysis)
            await write_classes_analysis(md_file, analysis)
            await write_functions_analysis(md_file, analysis)
            await write_source_code(md_file, analysis)
            await md_file.write("---\n\n")
        logger.info(f"Successfully wrote analysis to {md_file_path}")
    except OSError as e:
        logger.error(f"Error writing file {md_file_path}: {e}")
        raise

async def write_module_overview(md_file, analysis: Dict[str, Any]) -> None:
    """Write the module overview section."""
    module_info = analysis.get("module", {})
    if module_info:
        await md_file.write("<details>\n<summary><h2>📦 Module Overview</h2></summary>\n\n")
        docstring = module_info.get("docstring", "")
        if docstring:
            docstring = escape_markdown(docstring)
            await md_file.write(f"```text\n{docstring}\n```\n\n")
        await write_imports(md_file, module_info)
        await write_global_variables_and_constants(md_file, module_info)
        await md_file.write("</details>\n\n")
    logger.debug("Wrote module overview")

async def write_imports(md_file, module_info: Dict[str, Any]) -> None:
    """Write the imports section."""
    imports = module_info.get("imports", [])
    if imports:
        await md_file.write("### 📥 Imports\n\n")
        await md_file.write("| Import | Type | Alias |\n")
        await md_file.write("|:-------|:-----|:------|\n")
        for imp in imports:
            if imp["type"] == "import":
                await md_file.write(f"| `{escape_markdown(imp['module'])}` | Direct | `{escape_markdown(imp.get('alias', '-'))}` |\n")
            else:
                await md_file.write(f"| `{escape_markdown(imp['module'] + '.' + imp['name'])}` | From | `{escape_markdown(imp.get('alias', '-'))}` |\n")
        await md_file.write("\n")
    logger.debug("Wrote imports section")

async def write_global_variables_and_constants(md_file, module_info: Dict[str, Any]) -> None:
    """Write the global variables and constants section."""
    for section, items, icon in [
        ("Global Variables", module_info.get("global_variables", []), "🌍"),
        ("Constants", module_info.get("constants", []), "🔒")
    ]:
        if items:
            await md_file.write(f"### {icon} {section}\n\n")
            await md_file.write("| Name | Line Number |\n")
            await md_file.write("|:-----|:------------|\n")
            for item in items:
                await md_file.write(f"| `{escape_markdown(item['name'])}` | {item['line_number']} |\n")
            await md_file.write("\n")
    logger.debug("Wrote global variables and constants")

async def write_classes_analysis(md_file, analysis: Dict[str, Any]) -> None:
    """Write the classes analysis section."""
    classes = analysis.get("classes", [])
    if classes:
        await md_file.write("## 🔷 Classes\n\n")
        for cls in classes:
            await md_file.write(f"<details>\n<summary><h3>📘 {escape_markdown(cls['name'])}</h3></summary>\n\n")
            docstring = cls.get("docstring", "")
            if docstring:
                docstring = escape_markdown(docstring)
                await md_file.write(f"```text\n{docstring}\n```\n\n")
            bases = cls.get("base_classes", [])
            if bases:
                bases_str = ', '.join([escape_markdown(base) for base in bases])
                await md_file.write(f"**Inherits from:** `{bases_str}`\n\n")
            await write_class_components(md_file, cls)
            await md_file.write("</details>\n\n")
    logger.debug("Wrote classes analysis")

async def write_class_components(md_file, cls: Dict[str, Any]) -> None:
    """Write the components of a class."""
    for section, items, icon in [
        ("Attributes", cls.get("attributes", []), "📝"),
        ("Instance Variables", cls.get("instance_variables", []), "🔹"),
        ("Methods", cls.get("methods", []), "⚙️")
    ]:
        if items:
            await md_file.write(f"#### {icon} {section}\n\n")
            if section == "Methods":
                await write_methods_table(md_file, items)
            else:
                await write_attributes_or_instance_variables_table(md_file, section, items)
            await md_file.write("\n")
    logger.debug(f"Wrote class components for {cls['name']}")

async def write_methods_table(md_file, methods: List[Dict[str, Any]]) -> None:
    """Write the methods table."""
    await md_file.write("| Method | Type | Complexity | Description |\n")
    await md_file.write("|:-------|:-----|:-----------|:------------|\n")
    for method in methods:
        method_type = (
            "🔧 Static" if method.get("is_static") else
            "📊 Class" if method.get("is_class_method") else
            "📍 Property" if method.get("is_property") else
            "📌 Abstract" if method.get("is_abstract") else
            "⚡ Instance"
        )
        complexity = method.get("complexity_score", 0)
        indicator = create_complexity_indicator(complexity)
        summary = method.get("summary", "").split("\n")[0]
        await md_file.write(
            f"| `{escape_markdown(method['name'])}` | {method_type} | "
            f"{complexity} {indicator} | {escape_markdown(summary)} |\n"
        )
    logger.debug("Wrote methods table")

async def write_attributes_or_instance_variables_table(md_file, section: str, items: List[Dict[str, Any]]) -> None:
    """Write the attributes or instance variables table."""
    headers = {
        "Attributes": ["Name", "Type"],
        "Instance Variables": ["Name", "Line Number"]
    }
    cols = headers[section]
    await md_file.write(f"| {' | '.join(cols)} |\n")
    await md_file.write("|" + "|".join(":--" for _ in cols) + "|\n")
    for item in items:
        values = [f"`{escape_markdown(item['name'])}`"] + [f"`{escape_markdown(str(item.get(k.lower().replace(' ', '_'), '')))}`" for k in cols[1:]]
        await md_file.write(f"| {' | '.join(values)} |\n")
    logger.debug(f"Wrote {section.lower()} table")

async def write_functions_analysis(md_file, analysis: Dict[str, Any]) -> None:
    """Write the functions analysis section."""
    functions = analysis.get("functions", [])
    if functions:
        await md_file.write("## ⚡ Functions\n\n")
        for func in functions:
            await md_file.write(f"<details>\n<summary><h3>📘 {escape_markdown(func['name'])}</h3></summary>\n\n")
            docstring = func.get("docstring", "")
            if docstring:
                docstring = escape_markdown(docstring)
                await md_file.write(f"```text\n{docstring}\n```\n\n")
            await write_function_metadata(md_file, func)
            await write_function_parameters(md_file, func)
            await write_function_return_type(md_file, func)
            await write_function_complexity_metrics(md_file, func)
            await md_file.write("</details>\n\n")
    logger.debug("Wrote functions analysis")

async def write_function_metadata(md_file, func: Dict[str, Any]) -> None:
    """Write function metadata with badges."""
    await md_file.write("#### 📊 Overview\n\n")
    await md_file.write(f"- 📍 Lines: `{func['line_number']}-{func['end_line_number']}`\n")
    await md_file.write(f"- 🔄 Async: `{func['is_async']}`\n")
    await md_file.write(f"- ⚡ Generator: `{func['is_generator']}`\n")
    await md_file.write(f"- 🔁 Recursive: `{func['is_recursive']}`\n\n")
    logger.debug(f"Wrote metadata for function {func['name']}")

async def write_function_parameters(md_file, func: Dict[str, Any]) -> None:
    """Write the function parameters section."""
    params = func.get("params", [])
    if params:
        await md_file.write("#### 📥 Parameters\n\n")
        await md_file.write("| Parameter | Type | Type Hint |\n")
        await md_file.write("|:----------|:-----|:----------|\n")
        for param in params:
            hint = "✅" if param['has_type_hint'] else "❌"
            await md_file.write(f"| `{escape_markdown(param['name'])}` | `{escape_markdown(param['type'])}` | {hint} |\n")
        await md_file.write("\n")
    logger.debug(f"Wrote parameters for function {func['name']}")

async def write_function_return_type(md_file, func: Dict[str, Any]) -> None:
    """Write the function return type section."""
    ret_type = func.get("return_type", {})
    if ret_type:
        await md_file.write("#### 📤 Return Type\n\n")
        hint = "✅" if ret_type.get('has_type_hint') else "❌"
        await md_file.write(f"- Type: `{escape_markdown(ret_type.get('type', ''))}`\n")
        await md_file.write(f"- Type Hint: {hint}\n\n")
    logger.debug(f"Wrote return type for function {func['name']}")

async def write_function_complexity_metrics(md_file, func: Dict[str, Any]) -> None:
    """Write the function complexity metrics section."""
    metrics = func.get("complexity_metrics", {})
    if metrics:
        await md_file.write("#### 📊 Complexity Metrics\n\n")
        cyclo = metrics.get('cyclomatic', 0)
        cogn = metrics.get('cognitive', 0)
        indicator = create_complexity_indicator(cyclo)
        await md_file.write(f"- Cyclomatic: {cyclo} {indicator}\n")
        await md_file.write(f"- Cognitive: {cogn}\n")
        halstead = metrics.get("halstead", {})
        if halstead:
            await md_file.write("- Halstead:\n")
            for metric, value in halstead.items():
                await md_file.write(f"  - {escape_markdown(metric.title())}: {value:.2f}\n")
        await md_file.write("\n")
    logger.debug(f"Wrote complexity metrics for function {func['name']}")

async def write_source_code(md_file, analysis: Dict[str, Any]) -> None:
    """Write the source code section, escaping special markdown characters."""
    if not analysis:
        return

    await md_file.write("<details>\n<summary><h2>📝 Source Code</h2></summary>\n\n")
    await md_file.write("```python\n")

    file_content = analysis.get('file_content', [])
    if file_content and isinstance(file_content[0], dict):
        content = file_content[0].get('content', '')
        # Escape backticks and other markdown characters
        content = escape_markdown(content)
        await md_file.write(content)

    await md_file.write("\n```\n\n")
    await md_file.write("</details>\n\n")
    logger.debug("Wrote source code section")