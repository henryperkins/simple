import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

try:
    response = client.chat.completions.create(
        model=os.getenv(
            "AZURE_DEPLOYMENT_NAME"
        ),  # Use "model" instead of "engine"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"},
        ],
    )
    print(response.choices[0].message.content)  # Access the response content
except Exception as e:
    print(f"An error occurred: {e}")
