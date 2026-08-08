"""
Atlas AI Financial Assistant - Structured Logging Configuration
"""

import sys
import logging
from app.core.config import settings


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def setup_logger(name: str = "atlas_ai") -> logging.Logger:
    """Configures and returns a structured logger with UTF-8 safety."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        level = logging.DEBUG if settings.DEBUG else logging.INFO
        logger.setLevel(level)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(name)s:%(filename)s:%(lineno)d] | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        
    return logger


logger = setup_logger()
