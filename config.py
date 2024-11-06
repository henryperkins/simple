import os
from typing import Optional, Dict
from logging_utils import setup_logger

# Initialize a logger specifically for the config module
logger = setup_logger("config")

class Config:
    """Configuration class to manage access to environment variables and settings."""
    
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
    SENTRY_DSN = os.getenv("SENTRY_DSN", "")
    
    @classmethod
    def get_service_headers(cls, service: str) -> Dict[str, str]:
        """Get headers required for a specific service."""
        logger.debug(f"Fetching headers for service: {service}")
        if service == "azure":
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cls.AZURE_API_KEY}"
            }
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"
            }
        
        logger.debug(f"Generated headers for {service}: {headers}")
        return headers
    
    @classmethod
    def get_azure_endpoint(cls) -> str:
        """Retrieve the endpoint URL for Azure-based requests."""
        if not cls.AZURE_ENDPOINT:
            logger.warning("Azure endpoint is not set. Please configure AZURE_ENDPOINT.")
        logger.debug(f"Azure endpoint retrieved: {cls.AZURE_ENDPOINT}")
        return cls.AZURE_ENDPOINT or "https://default.azure.endpoint"
    
    @classmethod
    def load_environment(cls) -> None:
        """Ensure that all required environment variables are loaded."""
        required_vars = ["OPENAI_MODEL_NAME", "AZURE_ENDPOINT", "AZURE_API_KEY", "SENTRY_DSN"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
        else:
            logger.info("All required environment variables are set.")
    
    @classmethod
    def get_variable(cls, var_name: str) -> Optional[str]:
        """Retrieve a specific environment variable by name, with logging."""
        value = os.getenv(var_name)
        if value is None:
            logger.warning(f"Environment variable {var_name} is not set.")
        else:
            logger.debug(f"Retrieved environment variable {var_name}: {value}")
        return value