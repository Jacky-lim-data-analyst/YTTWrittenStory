"""
Project-wide custom logger

Goals:
- Single, consistent logging entry point
- Human-readable local logs

Usage:
```
from utils.logging import get_logger

logger = get_logger(__name__)
```
"""

import logging
import sys
from datetime import datetime
from typing import Optional

LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG
}

class SimpleFormatter(logging.Formatter):
    """
    Opinionated formatter:
    2025-12-31 14:03:12 | INFO | chunker | message
    """
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname.ljust(7)
        name = record.name
        message = record.getMessage()

        return f"{timestamp} | {level} | {name} | {message}"
    
def get_logger(
    name: str,
    level: str = "INFO",
    stream: Optional[object] = None
) -> logging.Logger:
    """
    Returns a configured logger instance
    
    - Prevents duplicate handlers
    - Uses stdout by default
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level = LOG_LEVELS.get(level.upper(), logging.INFO)
    logger.setLevel(log_level)

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(SimpleFormatter())

    logger.addHandler(handler)
    logger.propagate = False

    return logger 