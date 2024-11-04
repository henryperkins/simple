# config.py

import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class Config:
    """Configuration settings for the application."""

    # OpenAI API configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Azure OpenAI configuration
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    # Sentry configuration
    SENTRY_DSN = os.getenv("SENTRY_DSN")
    SENTRY_ENVIRONMENT = os.getenv("SENTRY_ENVIRONMENT", "production")
    SENTRY_RELEASE = os.getenv("SENTRY_RELEASE", "my-project-name@1.0.0")

    # Cache configuration
    CACHE_MAX_SIZE_MB = int(os.getenv("CACHE_MAX_SIZE_MB", 500))

    @staticmethod
    def validate():
        """Validate the configuration settings."""
        required_vars = [
            "OPENAI_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "SENTRY_DSN"
        ]
        missing_vars = [var for var in required_vars if not getattr(Config, var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Validate configuration on import
Config.validate()
