import sys
import logging
import asyncio
from code_extractor import extract_metadata_from_file
from project_context import build_project_context, summarize_global_context_for_function
from prompt_generator import generate_function_prompt
from ai_client import get_enriched_docstring
from response_parser import parse_and_validate_response, clean_response
from docstring_integrator import integrate_docstring_into_file
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


async def enrich_docstrings_in_file(file_path: str):
    logger.info("Starting docstring enrichment process for the entire module.")

    # Step 1: Extract metadata from the file
    logger.info(f"Extracting metadata from {file_path}.")
    extraction_result = extract_metadata_from_file(file_path)

    # Debugging: Print all extracted function and method names
    all_functions = extraction_result.functions
    for class_meta in extraction_result.classes:
        all_functions.extend(class_meta.methods)

    extracted_function_names = [f.name for f in all_functions]
    logger.info(f"Extracted functions and methods: {extracted_function_names}")

    # Step 2: Build project context
    logger.info("Building project context.")
    project_root = "."  # Assuming current directory is the project root
    project_context, _ = build_project_context(project_root)

    # Step 3: Enrich docstrings for each function and method
    for function_meta in all_functions:
        logger.info(f"Processing function/method: {function_meta.name}")

        # Summarize global context for the current function/method
        global_context = summarize_global_context_for_function(
            function_meta.name, project_context
        )

        # Generate the prompt
        logger.info("Generating prompt for AI.")
        prompt = generate_function_prompt(
            function_meta, extraction_result.module_metadata, global_context
        )

        # Get enriched docstring from AI
        logger.info("Sending prompt to AI for docstring generation.")
        raw_response = await get_enriched_docstring(prompt)

        # Debugging: Print the raw AI response
        logger.info(f"Raw AI response for {function_meta.name}: {raw_response}")

        # Clean the response before parsing
        cleaned_response = clean_response(raw_response)

        # Parse and validate the AI's response
        logger.info("Parsing and validating AI response.")
        try:
            if not cleaned_response.strip():
                raise ValueError("Empty response from AI")

            # Parse and validate the JSON response
            docstring_data = parse_and_validate_response(cleaned_response)
        except ValueError as e:
            logger.error(
                f"Failed to parse and validate AI response for {function_meta.name}: {e}"
            )
            continue

        # Integrate the enriched docstring into the file
        logger.info(
            f"Integrating enriched docstring into the source file for {function_meta.name}."
        )
        integrate_docstring_into_file(file_path, docstring_data, function_meta.name)

    logger.info(
        "Docstring enrichment process completed successfully for the entire module."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python enrich_docstrings.py <file_path>")
        print(f"Received arguments: {sys.argv}")
        sys.exit(1)

    file_path = sys.argv[1]

    asyncio.run(enrich_docstrings_in_file(file_path))
