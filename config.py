# config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_service_headers(service):
    """Get the appropriate headers for the specified service.

    Args:
        service (str): The service to use ('openai' or 'azure').

    Returns:
        dict: A dictionary containing the headers for the service.

    Raises:
        ValueError: If an invalid service is specified.
    """
    if service == "azure":
        return {
            "api-key": AZURE_OPENAI_API_KEY,
            "Content-Type": "application/json"
        }
    elif service == "openai":
        return {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
    else:
        raise ValueError("Invalid service specified")

def get_azure_endpoint():
    """Construct the Azure OpenAI endpoint URL.

    Returns:
        str: The constructed endpoint URL.
    """
    return f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"