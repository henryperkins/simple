import os
from typing import Dict, Optional
from dotenv import load_dotenv
from core.logging.setup import LoggerSetup

class Settings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self.logger = LoggerSetup.get_logger("config")
        load_dotenv()
        self.openai_model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-2024-08-06")
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT", "").rstrip('/')
        self.azure_api_key = os.getenv("AZURE_API_KEY", "")
        self.azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME", "")
        self.azure_api_version = os.getenv("AZURE_API_VERSION", "2024-08-01-preview")
        self.sentry_dsn = os.getenv("SENTRY_DSN", "")
        self.environment = os.getenv("ENVIRONMENT", "production")
        self.release_version = os.getenv("RELEASE_VERSION", "unknown")

    def get_service_headers(self, service: str) -> Dict[str, str]:
        self.logger.debug(f"Fetching headers for service: {service}")
        headers = {"Content-Type": "application/json"}
        if service == "azure":
            if not self.azure_api_key:
                self.logger.error("AZURE_API_KEY is not set.")
                raise ValueError("AZURE_API_KEY is not set.")
            headers["api-key"] = self.azure_api_key
        elif service == "openai":
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                self.logger.error("OPENAI_API_KEY is not set.")
                raise ValueError("OPENAI_API_KEY is not set.")
            headers["Authorization"] = f"Bearer {openai_api_key}"
        else:
            self.logger.error(f"Unsupported service: {service}")
            raise ValueError(f"Unsupported service: {service}")
        self.logger.debug(f"Generated headers for {service}: {headers}")
        return headers

    def get_azure_endpoint(self) -> str:
        if not self.azure_endpoint:
            self.logger.error("AZURE_ENDPOINT is not set.")
            raise ValueError("AZURE_ENDPOINT is not set.")
        self.logger.debug(f"Azure endpoint retrieved: {self.azure_endpoint}")
        return self.azure_endpoint