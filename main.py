"""
Entry point for local development.
"""

import sys
import os

# Disable ChromaDB telemetry early to avoid warnings
os.environ["CHROMA_TELEMETRY_ENABLED"] = "FALSE"

from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__, config.log_level)


def main():
    """Main entry point"""
    logger.info(f"Starting RAG application in {config.environment} environment...")
    config.validate()
    
    if config.is_local() or config.is_docker():
        # Run local API server
        import uvicorn
        from src.api import app
        
        logger.info(f"Starting API server at http://{config.api_host}:{config.api_port}")
        uvicorn.run(
            app,
            host=config.api_host,
            port=config.api_port,
            reload=config.debug
        )
    else:
        logger.error(f"Unsupported environment for main(): {config.environment}")
        sys.exit(1)


if __name__ == "__main__":
    main()
