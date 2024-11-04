import os
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    """Configuration settings for the application."""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    SENTRY_DSN = os.getenv("SENTRY_DSN")
    SENTRY_ENVIRONMENT = os.getenv("SENTRY_ENVIRONMENT", "production")
    SENTRY_RELEASE = os.getenv("SENTRY_RELEASE", "1.0.0")
    CACHE_MAX_SIZE_MB = int(os.getenv("CACHE_MAX_SIZE_MB", 500))

    @staticmethod
    def validate():
        """Validate that all required environment variables are set."""
        required_vars = [
            "OPENAI_API_KEY",
            "OPENAI_MODEL_NAME",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "SENTRY_DSN"
        ]
        missing_vars = [var for var in required_vars if not getattr(Config, var)]
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise EnvironmentError(error_msg)
        logger.info("All required environment variables are set.")

    @staticmethod
    def get_service_headers(service: str) -> dict:
        """
        Get the appropriate headers for the specified AI service.

        Args:
            service (str): The AI service to use ('azure' or 'openai').

        Returns:
            dict: A dictionary of headers.
        
        Raises:
            ValueError: If an invalid service is specified.
        """
        if service == "azure":
            return {
                "api-key": Config.AZURE_OPENAI_API_KEY,
                "Content-Type": "application/json"
            }
        elif service == "openai":
            return {
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
        else:
            error_msg = "Invalid service specified"
            logger.error(error_msg)
            raise ValueError(error_msg)

    @staticmethod
    def get_azure_endpoint() -> str:
        """
        Construct the Azure OpenAI endpoint URL.

        Returns:
            str: The constructed endpoint URL.
        """
        endpoint = f"{Config.AZURE_OPENAI_ENDPOINT}/openai/deployments/{Config.AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={Config.AZURE_OPENAI_API_VERSION}"
        logger.debug(f"Constructed Azure endpoint: {endpoint}")
        return endpoint