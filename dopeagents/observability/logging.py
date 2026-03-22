"""Rich-based colored logging for DopeAgents.

Provides a logger factory with environment-aware color support:
- Development: Colorful output via Rich
- Production: Plain text, no ANSI codes
- Files: Always plain text (Rich disables for non-TTY)

Usage:
    from dopeagents.observability.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Starting agent workflow")
    logger.debug("Detailed diagnostic info", extra={"workflow_id": "abc"})
"""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

from dopeagents.config import get_config, reset_config

# ── Module-level logger cache ──────────────────────────────────────

_loggers_cache: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """Get a logger with Rich support, configured from environment.

    Args:
        name: Logger name, typically __name__ from calling module.

    Returns:
        logging.Logger configured with Rich handler if colors enabled,
        plain StreamHandler otherwise.

    Example:
        logger = get_logger(__name__)
        logger.info("Agent started")
    """
    # Return cached logger if already created
    if name in _loggers_cache:
        return _loggers_cache[name]

    config = get_config()
    logger = logging.getLogger(name)

    # Set log level from config
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return _loggers_cache.setdefault(name, logger)

    # Create appropriate handler based on color config
    handler: logging.Handler
    if config.log_color:
        # Rich handler with colors for development
        handler = RichHandler(
            console=Console(file=sys.stderr),
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
        )
        formatter = logging.Formatter(
            fmt="%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        # Plain text handler for production
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Cache for reuse
    _loggers_cache[name] = logger

    return logger


def reset_logging() -> None:
    """Reset all cached loggers and config.

    Useful for testing or when reconfiguring DopeAgentsConfig.
    Clears both the logger cache and the global config singleton.
    """
    for logger in _loggers_cache.values():
        logger.handlers.clear()
    _loggers_cache.clear()
    reset_config()
