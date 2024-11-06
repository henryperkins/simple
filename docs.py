import os
from typing import Any, Dict, List
from tqdm import tqdm
from logging_utils import setup_logger, generate_summary

# Initialize logger for this module
logger = setup_logger("docs")

def create_complexity_indicator(complexity: int) -> str:
    """Create a visual indicator for code complexity."""
    indicator = "🟢" if complexity <= 5 else "🟡" if complexity <= 10 else "🔴"
    logger.debug(f"Complexity {complexity} has indicator {indicator}")
    return indicator

def write_analysis_to_markdown(results: Dict[str, Dict[str, Any]], output_file_path: str, repo_dir: str) -> None:
    """Write comprehensive analysis results to a markdown file."""
    logger.info("Writing analysis results to %s", output_file_path)
    
    try:
        with open(output_file_path, 'w', encoding='utf-8') as md_file:
            if not results:
                logger.error("No analysis results to write")
                md_file.write("# ⚠️ No Analysis Results\n\nNo valid code analysis data was found.")
                return

            write_markdown_header(md_file)
            
            valid_results = {k: v for k, v in results.items() if v and isinstance(v, dict)}
            logger.debug(f"Valid results count: {len(valid_results)}")
            
            if not valid_results:
                logger.warning("No valid analysis results after filtering")
                md_file.write("\n## ⚠️ Warning\nNo valid analysis results found after filtering.")
                return
                
            write_table_of_contents(md_file, valid_results, repo_dir)
            write_analysis_details(md_file, valid_results, repo_dir)
            
            md_file.write(f"\n## 📈 Summary\n\n")
            md_file.write(f"- Total files analyzed: {len(valid_results)}\n")
            
        logger.info("Successfully wrote analysis to markdown.")
        
    except OSError as e:
        logger.error("Error writing markdown file: %s", str(e))
        raise

def write_markdown_header(md_file) -> None:
    """Write the header for the markdown file."""
    md_file.write("# 📊 Code Analysis Report\n\n")
    md_file.write("## 📑 Table of Contents\n\n")
    logger.debug("Wrote markdown header")

def write_table_of_contents(md_file, results: Dict[str, Dict[str, Any]], repo_dir: str) -> None:
    """Write the table of contents for the markdown file."""
    for filepath in results:
        rel_path = os.path.relpath(filepath, repo_dir)
        anchor = rel_path.replace('/', '-').replace('.', '-').replace(' ', '-')
        md_file.write(f"- [📄 {rel_path}](#{anchor})\n")
    md_file.write("\n---\n\n")
    logger.debug("Wrote table of contents")

def write_analysis_details(md_file, results: Dict[str, Dict[str, Any]], repo_dir: str) -> None:
    """Write detailed analysis for each file."""
    for filepath, analysis in results.items():
        rel_path = os.path.relpath(filepath, repo_dir)
        md_file.write(f"\n## 📄 {rel_path}\n\n")
        
        if 'metrics' in analysis:
            md_file.write("### 📊 Metrics\n\n")
            metrics = analysis['metrics']
            md_file.write(f"- Lines of Code: {metrics.get('loc', 'N/A')}\n")
            md_file.write(f"- Complexity: {create_complexity_indicator(metrics.get('complexity', 0))}\n\n")
        
        if 'docstrings' in analysis:
            md_file.write("### 📝 Documentation\n\n")
            for doc in analysis['docstrings']:
                md_file.write(f"#### {doc.get('name', 'Unknown')}\n")
                md_file.write(f"{doc.get('content', 'No documentation available')}\n\n")
    logger.debug("Wrote analysis details")

def write_file_section(md_file, rel_path: str, anchor: str, analysis: Dict[str, Any]) -> None:
    """Write the section for a single file in the markdown file."""
    md_file.write(f"# 📄 {rel_path}\n\n")
    write_module_overview(md_file, analysis)
    write_classes_analysis(md_file, analysis)
    write_functions_analysis(md_file, analysis)
    write_source_code(md_file, analysis)
    md_file.write("---\n\n")
    logger.debug(f"Wrote file section for {rel_path}")

def write_module_overview(md_file, analysis: Dict[str, Any]) -> None:
    """Write the module overview section."""
    if module_info := analysis.get("module", {}):
        md_file.write("<details>\n<summary><h2>📦 Module Overview</h2></summary>\n\n")
        if docstring := module_info.get("docstring"):
            md_file.write(f"```text\n{docstring}\n```\n\n")
        write_imports(md_file, module_info)
        write_global_variables_and_constants(md_file, module_info)
        md_file.write("</details>\n\n")
    logger.debug("Wrote module overview")

def write_imports(md_file, module_info: Dict[str, Any]) -> None:
    """Write the imports section."""
    if imports := module_info.get("imports"):
        md_file.write("### 📥 Imports\n\n")
        md_file.write("| Import | Type | Alias |\n")
        md_file.write("|:-------|:-----|:------|\n")
        for imp in imports:
            if imp["type"] == "import":
                md_file.write(f"| `{imp['module']}` | Direct | `{imp.get('alias', '-')}` |\n")
            else:
                md_file.write(f"| `{imp['module']}.{imp['name']}` | From | `{imp.get('alias', '-')}` |\n")
        md_file.write("\n")
    logger.debug("Wrote imports section")

def write_global_variables_and_constants(md_file, module_info: Dict[str, Any]) -> None:
    """Write the global variables and constants section."""
    for section, items, icon in [
        ("Global Variables", module_info.get("global_variables"), "🌍"),
        ("Constants", module_info.get("constants"), "🔒")
    ]:
        if items:
            md_file.write(f"### {icon} {section}\n\n")
            md_file.write("| Name | Line Number |\n")
            md_file.write("|:-----|:------------|\n")
            for item in items:
                md_file.write(f"| `{item['name']}` | {item['line_number']} |\n")
            md_file.write("\n")
    logger.debug("Wrote global variables and constants")

def write_classes_analysis(md_file, analysis: Dict[str, Any]) -> None:
    """Write the classes analysis section."""
    if classes := analysis.get("classes"):
        md_file.write("## 🔷 Classes\n\n")
        for cls in classes:
            md_file.write(f"<details>\n<summary><h3>📘 {cls['name']}</h3></summary>\n\n")
            if docstring := cls.get("docstring"):
                md_file.write(f"```text\n{docstring}\n```\n\n")
            if bases := cls.get("base_classes"):
                md_file.write(f"**Inherits from:** `{', '.join(bases)}`\n\n")
            write_class_components(md_file, cls)
            md_file.write("</details>\n\n")
    logger.debug("Wrote classes analysis")

def write_class_components(md_file, cls: Dict[str, Any]) -> None:
    """Write the components of a class."""
    for section, items, icon in [
        ("Attributes", cls.get("attributes"), "📝"),
        ("Instance Variables", cls.get("instance_variables"), "🔹"),
        ("Methods", cls.get("methods"), "⚙️")
    ]:
        if items:
            md_file.write(f"#### {icon} {section}\n\n")
            if section == "Methods":
                write_methods_table(md_file, items)
            else:
                write_attributes_or_instance_variables_table(md_file, section, items)
            md_file.write("\n")
    logger.debug(f"Wrote class components for {cls['name']}")

def write_methods_table(md_file, methods: List[Dict[str, Any]]) -> None:
    """Write the methods table."""
    md_file.write("| Method | Type | Complexity | Description |\n")
    md_file.write("|:-------|:-----|:-----------|:------------|\n")
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
        md_file.write(
            f"| `{method['name']}` | {method_type} | "
            f"{complexity} {indicator} | {summary} |\n"
        )
    logger.debug("Wrote methods table")

def write_attributes_or_instance_variables_table(md_file, section: str, items: List[Dict[str, Any]]) -> None:
    """Write the attributes or instance variables table."""
    headers = {
        "Attributes": ["Name", "Type"],
        "Instance Variables": ["Name", "Line Number"]
    }
    cols = headers[section]
    md_file.write(f"| {' | '.join(cols)} |\n")
    md_file.write("|:" + "|:".join("-" * len(col) for col in cols) + "|\n")
    for item in items:
        values = [f"`{item['name']}`"] + [str(item[k.lower().replace(' ', '_')]) for k in cols[1:]]
        md_file.write(f"| {' | '.join(values)} |\n")
    logger.debug(f"Wrote {section.lower()} table")

def write_functions_analysis(md_file, analysis: Dict[str, Any]) -> None:
    """Write the functions analysis section."""
    if functions := analysis.get("functions"):
        md_file.write("## ⚡ Functions\n\n")
        for func in functions:
            md_file.write(f"<details>\n<summary><h3>📘 {func['name']}</h3></summary>\n\n")
            if docstring := func.get("docstring"):
                md_file.write(f"```text\n{docstring}\n```\n\n")
            write_function_metadata(md_file, func)
            write_function_parameters(md_file, func)
            write_function_return_type(md_file, func)
            write_function_complexity_metrics(md_file, func)
            md_file.write("</details>\n\n")
    logger.debug("Wrote functions analysis")

def write_function_metadata(md_file, func: Dict[str, Any]) -> None:
    """Write function metadata with badges."""
    md_file.write("#### 📊 Overview\n\n")
    md_file.write(f"- 📍 Lines: `{func['line_number']}-{func['end_line_number']}`\n")
    md_file.write(f"- 🔄 Async: `{func['is_async']}`\n")
    md_file.write(f"- ⚡ Generator: `{func['is_generator']}`\n")
    md_file.write(f"- 🔁 Recursive: `{func['is_recursive']}`\n\n")
    logger.debug(f"Wrote metadata for function {func['name']}")

def write_function_parameters(md_file, func: Dict[str, Any]) -> None:
    """Write the function parameters section."""
    if params := func.get("params"):
        md_file.write("#### 📥 Parameters\n\n")
        md_file.write("| Parameter | Type | Type Hint |\n")
        md_file.write("|:----------|:-----|:---------|\n")
        for param in params:
            hint = "✅" if param['has_type_hint'] else "❌"
            md_file.write(f"| `{param['name']}` | `{param['type']}` | {hint} |\n")
        md_file.write("\n")
    logger.debug(f"Wrote parameters for function {func['name']}")

def write_function_return_type(md_file, func: Dict[str, Any]) -> None:
    """Write the function return type section."""
    if ret_type := func.get("return_type"):
        md_file.write("#### 📤 Return Type\n\n")
        hint = "✅" if ret_type['has_type_hint'] else "❌"
        md_file.write(f"- Type: `{ret_type['type']}`\n")
        md_file.write(f"- Type Hint: {hint}\n\n")
    logger.debug(f"Wrote return type for function {func['name']}")

def write_function_complexity_metrics(md_file, func: Dict[str, Any]) -> None:
    """Write the function complexity metrics section."""
    if metrics := func.get("complexity_metrics"):
        md_file.write("#### 📊 Complexity Metrics\n\n")
        cyclo = metrics['cyclomatic']
        cogn = metrics['cognitive']
        indicator = create_complexity_indicator(cyclo)
        md_file.write(f"- Cyclomatic: {cyclo} {indicator}\n")
        md_file.write(f"- Cognitive: {cogn}\n")
        if halstead := metrics.get("halstead"):
            md_file.write("- Halstead:\n")
            for metric, value in halstead.items():
                md_file.write(f"  - {metric.title()}: {value:.2f}\n")
        md_file.write("\n")
    logger.debug(f"Wrote complexity metrics for function {func['name']}")

def write_source_code(md_file, analysis: Dict[str, Any]) -> None:
    """Write the source code section."""
    if not analysis:
        return
    
    md_file.write("<details>\n<summary><h2>📝 Source Code</h2></summary>\n\n")
    md_file.write("```python\n")
    
    file_content = analysis.get('file_content', [])
    if file_content and isinstance(file_content[0], dict):
        content = file_content[0].get('content', '')
        md_file.write(content)
    
    md_file.write("\n```\n\n")
    md_file.write("</details>\n\n")
    logger.debug("Wrote source code section")
