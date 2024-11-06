import os
from typing import Optional, Dict
from logging_utils import setup_logger
from dotenv import load_dotenv

# Initialize a logger specifically for the config module
logger = setup_logger("config")

class Config:
    """
    Configuration class to manage access to environment variables and settings.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Class variables for configuration settings
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "").rstrip('/')
    AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
    AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "")
    SENTRY_DSN = os.getenv("SENTRY_DSN", "")

    @classmethod
    def get_service_headers(cls, service: str) -> Dict[str, str]:
        logger.debug(f"Fetching headers for service: {service}")
        headers = {"Content-Type": "application/json"}
        if service == "azure":
            if not cls.AZURE_API_KEY:
                logger.error("AZURE_API_KEY is not set.")
                raise ValueError("AZURE_API_KEY is not set.")
            headers["api-key"] = cls.AZURE_API_KEY
        elif service == "openai":
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                logger.error("OPENAI_API_KEY is not set.")
                raise ValueError("OPENAI_API_KEY is not set.")
            headers["Authorization"] = f"Bearer {openai_api_key}"
        else:
            logger.error(f"Unsupported service: {service}")
            raise ValueError(f"Unsupported service: {service}")
        logger.debug(f"Generated headers for {service}: {headers}")
        return headers

    @classmethod
    def get_azure_endpoint(cls) -> str:
        if not cls.AZURE_ENDPOINT:
            logger.error("AZURE_ENDPOINT is not set. Please configure AZURE_ENDPOINT.")
            raise ValueError("AZURE_ENDPOINT is not set.")
        logger.debug(f"Azure endpoint retrieved: {cls.AZURE_ENDPOINT}")
        return cls.AZURE_ENDPOINT

    @classmethod
    def load_environment(cls) -> None:
        """
        Ensure that all required environment variables are loaded.
        """
        required_vars = ["OPENAI_MODEL_NAME", "SENTRY_DSN"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
        else:
            logger.info("All required environment variables are set.")

    @classmethod
    def get_variable(cls, var_name: str) -> Optional[str]:
        """
        Retrieve a specific environment variable by name, with logging.
        """
        value = os.getenv(var_name)
        if value is None:
            logger.warning(f"Environment variable {var_name} is not set.")
        else:
            logger.debug(f"Retrieved environment variable {var_name}.")
        return value
