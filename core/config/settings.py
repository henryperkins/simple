import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from core.logger import LoggerSetup

# Load environment variables from .env file
load_dotenv()

class Settings:
    def __init__(self):
        self.logger = LoggerSetup.get_logger("settings")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.azure_api_key = os.getenv("AZURE_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")
        self.azure_model_name = os.getenv("AZURE_MODEL_NAME", "gpt-4o-2024-08-06")
        self.azure_api_version = os.getenv("AZURE_API_VERSION", "2024-08-01-preview")
        self.sentry_dsn = os.getenv("SENTRY_DSN")
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.release_version = os.getenv("RELEASE_VERSION", "1.0.0")

        # Log whether keys are set (without exposing their values)
        self.logger.debug(f"OPENAI_API_KEY set: {'Yes' if self.openai_api_key else 'No'}")
        self.logger.debug(f"AZURE_API_KEY set: {'Yes' if self.azure_api_key else 'No'}")

        # Validate essential environment variables
        if not self.openai_api_key:
            self.logger.error("OPENAI_API_KEY is not set.")
            raise ValueError("OPENAI_API_KEY is not set.")
        if not self.azure_api_key:
            self.logger.error("AZURE_API_KEY is not set.")
            raise ValueError("AZURE_API_KEY is not set.")
        if not self.azure_endpoint:
            self.logger.error("AZURE_ENDPOINT is not set.")
            raise ValueError("AZURE_ENDPOINT is not set.")
        if not self.sentry_dsn:
            self.logger.error("SENTRY_DSN is not set.")
            raise ValueError("SENTRY_DSN is not set.")

    def get_service_headers(self, service: str) -> dict:
        """
        Get headers required for a specific service.

        Args:
            service (str): The service name ('azure' or 'openai').

        Returns:
            dict: Headers with authorization for the specified service.

        Raises:
            ValueError: If the service is unsupported or required keys are not set.
        """
        headers = {"Content-Type": "application/json"}
    
        if service == "azure":
            self.logger.debug(f"Checking Azure API key (present: {'Yes' if self.azure_api_key else 'No'})")
            if not self.azure_api_key:
                self.logger.error("AZURE_API_KEY is not set when creating headers")
                raise ValueError("AZURE_API_KEY is not set")
            headers["api-key"] = self.azure_api_key
        elif service == "openai":
            if not self.openai_api_key:
                self.logger.error("OPENAI_API_KEY is not set when creating headers")
                raise ValueError("OPENAI_API_KEY is not set")
            headers["Authorization"] = f"Bearer {self.openai_api_key}"
        else:
            self.logger.error(f"Unsupported service: {service}")
            raise ValueError(f"Unsupported service: {service}")
        
        return headers

    def get_azure_endpoint(self) -> str:
        """
        Retrieve the endpoint URL for Azure-based requests.

        Returns:
            str: The Azure endpoint URL.

        Raises:
            ValueError: If AZURE_ENDPOINT is not set.
        """
        if not self.azure_endpoint:
            self.logger.error("AZURE_ENDPOINT is not set.")
            raise ValueError("AZURE_ENDPOINT is not set.")
        self.logger.debug(f"Azure endpoint retrieved: {self.azure_endpoint}")
        return self.azure_endpoint
