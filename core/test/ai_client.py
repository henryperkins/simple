import os
import asyncio
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up Azure OpenAI client
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
api_key = os.getenv('AZURE_OPENAI_KEY')
api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
deployment_id = os.getenv('AZURE_OPENAI_DEPLOYMENT_ID')

# Initialize the Async client
client = AsyncAzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version
)

async def get_enriched_docstring(prompt: str) -> str:
    """
    Send a prompt to the Azure OpenAI API and return the raw AI response.

    Args:
        prompt (str): The prompt to send to the AI model.

    Returns:
        str: The raw response from the AI model.
    """
    try:
        # Configure API call parameters
        response = await client.chat.completions.create(
            model=deployment_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error during API call: {e}")
        return "An error occurred with the Azure OpenAI API."

# Example usage
if __name__ == "__main__":
    async def main():
        prompt = "Your generated prompt here"
        docstring = await get_enriched_docstring(prompt)
        print(docstring)

    asyncio.run(main())