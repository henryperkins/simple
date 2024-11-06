import logging
import os
from logging.handlers import RotatingFileHandler

class LoggerSetup:
    @staticmethod
    def get_logger(module_name: str, console_logging: bool = False) -> logging.Logger:
        """
        Get a logger for a specific module with optional console logging.

        Args:
            module_name (str): The name of the module for which to set up the logger.
            console_logging (bool): If True, also log to console.

        Returns:
            logging.Logger: Configured logger for the module.
        """
        logger = logging.getLogger(module_name)
        if not logger.handlers:  # Avoid adding handlers multiple times
            logger.setLevel(logging.DEBUG)
            LoggerSetup._add_file_handler(logger, module_name)
            if console_logging:
                LoggerSetup._add_console_handler(logger)
        return logger

    @staticmethod
    def _add_file_handler(logger: logging.Logger, module_name: str) -> None:
        """
        Add a rotating file handler to the logger.

        Args:
            logger (logging.Logger): The logger to which the handler is added.
            module_name (str): The name of the module for log file naming.
        """
        log_dir = os.path.join("logs", module_name)
        os.makedirs(log_dir, exist_ok=True)
        handler = RotatingFileHandler(
            os.path.join(log_dir, f"{module_name}.log"),
            maxBytes=10**6,  # 1 MB
            backupCount=5
        )
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)

    @staticmethod
    def _add_console_handler(logger: logging.Logger) -> None:
        """
        Add a console handler to the logger.

        Args:
            logger (logging.Logger): The logger to which the handler is added.
        """
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(console_handler)

    @staticmethod
    def generate_summary():
        """
        Generates a summary log file with critical information from all module logs.
        """
        log_dir = "logs"
        summary_log_path = os.path.join(log_dir, "summary.log")
        try:
            with open(summary_log_path, 'w', encoding='utf-8') as summary_file:
                for module_name in os.listdir(log_dir):
                    module_log_dir = os.path.join(log_dir, module_name)
                    module_log_path = os.path.join(module_log_dir, f"{module_name}.log")
                    if os.path.exists(module_log_path):
                        try:
                            with open(module_log_path, 'r', encoding='utf-8') as log_file:
                                for line in log_file:
                                    if "ERROR" in line or "WARNING" in line:
                                        summary_file.write(line)
                        except OSError as e:
                            print(f"Error reading log file {module_log_path}: {e}")
            print(f"Summary log generated at: {summary_log_path}")
        except OSError as e:
            print(f"Error writing summary log file: {e}")