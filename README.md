# Simple Extract

Simple Extract is a Python project designed to analyze and document code using OpenAI's API. The project processes Python files, extracts classes and functions, and generates comprehensive documentation.

## Project Structure

- **api_interaction.py**: Handles interactions with OpenAI and Azure OpenAI APIs, including making requests and caching responses.
- **cache.py**: Implements a Least Recently Used (LRU) cache for storing API responses.
- **code_extraction.py**: Extracts class and function details from the Abstract Syntax Tree (AST) of Python files.
- **config.py**: Manages configuration settings and environment variables for OpenAI and Azure OpenAI services.
- **file_processing.py**: Processes Python files, including cloning repositories, reading files, and extracting code details.
- **main.py**: The main entry point for the documentation generator, handling argument parsing and orchestrating the analysis process.
- **monitoring.py**: Initializes and configures Sentry for error monitoring and performance tracking.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/henryperkins/simple.git
    cd simple
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a `.env` file in the project root.
    - Add the following environment variables:
      ```
      AZURE_OPENAI_API_KEY=your_azure_openai_api_key
      AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
      AZURE_OPENAI_DEPLOYMENT_NAME=your_azure_openai_deployment_name
      OPENAI_API_KEY=your_openai_api_key
      ```

## Usage

To analyze a GitHub repository or local directory and generate documentation, run the following command:

```bash
python main.py <input_path> <output_file> --service <service>
```

- `<input_path>`: GitHub repository URL or local directory path to analyze.
- `<output_file>`: File to save the generated Markdown documentation.
- `--service`: The AI service to use (`openai` or `azure`). Default is `openai`.

Example:

```bash
python main.py https://github.com/henryperkins/simple cloned_repo/output.md --service openai
```

## License

This project is licensed under the MIT License.
Feel free to customize this `README.md` as needed, especially the installation and usage sections. If you have any additional details or instructions, you can include them in the relevant sections.
