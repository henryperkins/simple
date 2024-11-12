# main.py

import argparse
import asyncio
import os
import sys
from datetime import datetime
from urllib.parse import urlparse
from dotenv import load_dotenv

from core.logger import LoggerSetup
from utils import ensure_directory
from files import create_analyzer
from docs import write_analysis_to_markdown
from monitoring import initialize_sentry
import sentry_sdk

logger = LoggerSetup.get_logger("main")
load_dotenv()

class CodeAnalysisRunner:
    """Manages the code analysis process."""

    def __init__(self):
        self.analyzer = create_analyzer()
        self.summary_data = {
            'files_processed': 0,
            'errors_encountered': 0,
            'start_time': datetime.now(),
            'end_time': None
        }

    @staticmethod
    def validate_repo_url(url: str) -> bool:
        """Validate GitHub repository URL."""
        try:
            parsed = urlparse(url)
            return (parsed.netloc in ['github.com', 'www.github.com'] and
                    len([p for p in parsed.path.split('/') if p]) >= 2)
        except ValueError as e:
            logger.error(f"URL validation error: {e}")
            return False

    @staticmethod
    def validate_environment(service: str) -> None:
        """Validate required environment variables."""
        required_vars = {
            "azure": ["AZURE_OPENAI_API_KEY", "AZURE_ENDPOINT"],
            "openai": ["OPENAI_API_KEY"],
            "claude": ["ANTHROPIC_API_KEY"]
        }

        missing_vars = [var for var in required_vars[service] if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

    async def run_analysis(self, input_path: str, output_path: str, service: str) -> None:
        """Run the code analysis process."""
        logger.info("Starting code analysis process")
        try:
            # Initialize services
            initialize_sentry()
            self.validate_environment(service)
            ensure_directory(output_path)

            # Process input path
            if input_path.startswith(('http://', 'https://')):
                if not self.validate_repo_url(input_path):
                    raise ValueError(f"Invalid GitHub repository URL: {input_path}")
                logger.debug(f"Analyzing repository: {input_path}")
                results = await self.analyzer.analyze_repository(input_path, output_path, service)
            else:
                logger.debug(f"Analyzing directory: {input_path}")
                results = await self.analyzer.analyze_directory(input_path, service)

            if not results:
                raise ValueError("No valid results from analysis")

            # Generate documentation
            logger.debug("Generating markdown documentation")
            await write_analysis_to_markdown(results, output_path)
            self.summary_data['files_processed'] = len(results)
            logger.info(f"Analysis complete. Documentation written to {output_path}")

        except Exception as e:
            self.summary_data['errors_encountered'] += 1
            logger.error(f"Error during execution: {e}")
            sentry_sdk.capture_exception(e)
            raise
        finally:
            self.summary_data['end_time'] = datetime.now()
            self._log_summary()

    def _log_summary(self) -> None:
        """Log analysis summary."""
        duration = self.summary_data['end_time'] - self.summary_data['start_time']
        logger.info(
            "Summary: Files processed: %d, Errors: %d, Duration: %s",
            self.summary_data['files_processed'],
            self.summary_data['errors_encountered'],
            str(duration)
        )

async def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Analyze code and generate documentation.")
    parser.add_argument("input_path", help="Path to the input directory or repository URL")
    parser.add_argument("output_path", help="Path to the output directory for markdown files")
    parser.add_argument(
        "--service",
        choices=["azure", "openai", "claude"],
        required=True,
        help="AI service to use"
    )
    args = parser.parse_args()

    try:
        runner = CodeAnalysisRunner()
        await runner.run_analysis(args.input_path, args.output_path, args.service)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
